#!/usr/bin/env python3
"""
eval/compare.py — automated speech quality comparison between two WAV files.

Usage:
    python eval/compare.py --ref path/to/ref.wav --hyp path/to/hyp.wav
    python eval/compare.py --ref ref.wav --hyp hyp.wav --json out.json

Metrics:
    PESQ (wideband, ≥ 3.0)   — speech quality (requires 16 kHz)
    STOI (≥ 0.85)             — speech intelligibility (16 kHz)
    MCD  (≤ 5.0 dB)           — mel-cepstral distortion
    SI-SNR (≥ 15 dB)          — signal-level fidelity
    RTF  (≤ 1.0 CUDA, ≤ 3.0 CPU) — passed in explicitly, not computed here

Thresholds align with the project evaluation spec (AGENTS.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Optional metric imports (fail gracefully) ────────────────────────────────

def _try_import(module: str):
    try:
        import importlib
        return importlib.import_module(module)
    except ImportError:
        return None


THRESHOLDS = {
    "pesq": (">=", 3.0),
    "stoi": (">=", 0.85),
    "mcd":  ("<=", 5.0),
    "si_snr": (">=", 15.0),
}


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV file; returns (float32 mono [-1,1], sample_rate)."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except ImportError:
        import scipy.io.wavfile as wavfile
        sr, audio = wavfile.read(str(path))
        if audio.dtype.kind == "i":
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=src_sr, target_sr=tgt_sr)
    except ImportError:
        pass
    # scipy fallback
    from scipy.signal import resample_poly
    import math
    g = math.gcd(src_sr, tgt_sr)
    return resample_poly(audio, tgt_sr // g, src_sr // g).astype(np.float32)


def align_length(ref: np.ndarray, hyp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim or zero-pad hyp to match ref length."""
    n = len(ref)
    if len(hyp) > n:
        hyp = hyp[:n]
    elif len(hyp) < n:
        hyp = np.pad(hyp, (0, n - len(hyp)))
    return ref, hyp


# ── Per-metric computation ───────────────────────────────────────────────────

def compute_pesq(ref16: np.ndarray, hyp16: np.ndarray, sr: int = 16000) -> Optional[float]:
    pesq_mod = _try_import("pesq")
    if pesq_mod is None:
        return None
    try:
        return float(pesq_mod.pesq(sr, ref16, hyp16, "wb"))
    except Exception as exc:
        print(f"  [PESQ] error: {exc}", file=sys.stderr)
        return None


def compute_stoi(ref16: np.ndarray, hyp16: np.ndarray, sr: int = 16000) -> Optional[float]:
    pystoi_mod = _try_import("pystoi")
    if pystoi_mod is None:
        return None
    try:
        return float(pystoi_mod.stoi(ref16, hyp16, sr, extended=False))
    except Exception as exc:
        print(f"  [STOI] error: {exc}", file=sys.stderr)
        return None


def compute_mcd(ref_path: str | Path, hyp_path: str | Path) -> Optional[float]:
    pymcd_mod = _try_import("pymcd.metrics")
    if pymcd_mod is None:
        return None
    try:
        calculator = pymcd_mod.Calculate_MCD(MCD_mode="plain")
        return float(calculator.calculate_mcd(str(ref_path), str(hyp_path)))
    except Exception as exc:
        print(f"  [MCD] error: {exc}", file=sys.stderr)
        return None


def compute_si_snr(ref: np.ndarray, hyp: np.ndarray) -> Optional[float]:
    try:
        import torch
        from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
        metric = ScaleInvariantSignalNoiseRatio()
        val = metric(
            torch.tensor(hyp).unsqueeze(0),
            torch.tensor(ref).unsqueeze(0),
        )
        return float(val.item())
    except Exception:
        pass
    # Pure numpy fallback
    try:
        ref_f = ref - ref.mean()
        hyp_f = hyp - hyp.mean()
        dot = float(np.dot(ref_f, hyp_f))
        ref_sq = float(np.dot(ref_f, ref_f)) + 1e-8
        s_target = (dot / ref_sq) * ref_f
        e_noise = hyp_f - s_target
        ratio = np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8)
        return float(10.0 * np.log10(ratio + 1e-8))
    except Exception as exc:
        print(f"  [SI-SNR] error: {exc}", file=sys.stderr)
        return None


# ── Pass/fail helpers ────────────────────────────────────────────────────────

def passes_threshold(name: str, value: Optional[float]) -> Optional[bool]:
    if value is None:
        return None
    op, thr = THRESHOLDS[name]
    if op == ">=":
        return value >= thr
    elif op == "<=":
        return value <= thr
    return None


def _status(name: str, value: Optional[float]) -> str:
    p = passes_threshold(name, value)
    if p is None:
        return "N/A"
    return "PASS" if p else "FAIL"


# ── Main comparison ──────────────────────────────────────────────────────────

def compare(
    ref_path: str | Path,
    hyp_path: str | Path,
    rtf: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """
    Compare two WAV files and return a dict of metrics.

    Args:
        ref_path: path to reference (PyTorch baseline) WAV
        hyp_path: path to hypothesis (exported model) WAV
        rtf: real-time factor (wall-clock / audio duration); caller computes this
        verbose: print a table to stdout

    Returns:
        dict with keys: pesq, stoi, mcd, si_snr, rtf, duration_ref, duration_hyp,
                         pesq_pass, stoi_pass, mcd_pass, si_snr_pass, all_pass
    """
    ref_audio, ref_sr = load_audio(ref_path)
    hyp_audio, hyp_sr = load_audio(hyp_path)

    duration_ref = len(ref_audio) / ref_sr
    duration_hyp = len(hyp_audio) / hyp_sr

    # Resample both to 16 kHz for PESQ / STOI
    ref16 = resample(ref_audio, ref_sr, 16000)
    hyp16 = resample(hyp_audio, hyp_sr, 16000)
    ref16, hyp16 = align_length(ref16, hyp16)

    # Also align at native SR for SI-SNR
    ref_nat, hyp_nat = align_length(
        resample(ref_audio, ref_sr, ref_sr),
        resample(hyp_audio, hyp_sr, ref_sr),
    )

    pesq_val = compute_pesq(ref16, hyp16)
    stoi_val = compute_stoi(ref16, hyp16)
    mcd_val  = compute_mcd(ref_path, hyp_path)
    si_snr_val = compute_si_snr(ref_nat, hyp_nat)

    results: dict = {
        "ref": str(ref_path),
        "hyp": str(hyp_path),
        "duration_ref": round(duration_ref, 3),
        "duration_hyp": round(duration_hyp, 3),
        "pesq":   round(pesq_val, 4) if pesq_val is not None else None,
        "stoi":   round(stoi_val, 4) if stoi_val is not None else None,
        "mcd":    round(mcd_val, 4)  if mcd_val  is not None else None,
        "si_snr": round(si_snr_val, 4) if si_snr_val is not None else None,
        "rtf":    round(rtf, 4)      if rtf       is not None else None,
    }

    # Pass / fail flags
    results["pesq_pass"]   = passes_threshold("pesq",   pesq_val)
    results["stoi_pass"]   = passes_threshold("stoi",   stoi_val)
    results["mcd_pass"]    = passes_threshold("mcd",    mcd_val)
    results["si_snr_pass"] = passes_threshold("si_snr", si_snr_val)

    results["all_pass"] = all(
        v is True
        for v in [
            results["pesq_pass"],
            results["stoi_pass"],
            results["mcd_pass"],
            results["si_snr_pass"],
        ]
        if v is not None
    )

    if verbose:
        print("\n── Speech Quality Comparison ──────────────────────────────")
        print(f"  REF : {ref_path}  ({duration_ref:.2f}s)")
        print(f"  HYP : {hyp_path}  ({duration_hyp:.2f}s)")
        if rtf is not None:
            rtf_thr = 1.0 if rtf <= 1.0 else 3.0
            print(f"  RTF : {rtf:.3f}  (threshold ≤ {rtf_thr})")
        print(f"  {'Metric':<12} {'Value':>10}  {'Threshold':>12}  {'Status':>6}")
        print(f"  {'-'*50}")
        rows = [
            ("PESQ (WB)",  results["pesq"],   "≥ 3.0",  results["pesq_pass"]),
            ("STOI",       results["stoi"],   "≥ 0.85", results["stoi_pass"]),
            ("MCD (dB)",   results["mcd"],    "≤ 5.0",  results["mcd_pass"]),
            ("SI-SNR (dB)",results["si_snr"], "≥ 15.0", results["si_snr_pass"]),
        ]
        for name, val, thr, passed in rows:
            val_str = f"{val:.3f}" if val is not None else "n/a"
            status  = ("PASS" if passed else "FAIL") if passed is not None else "N/A"
            print(f"  {name:<12} {val_str:>10}  {thr:>12}  {status:>6}")
        print(f"\n  Overall: {'ALL PASS' if results['all_pass'] else 'SOME FAILED'}")
        print("───────────────────────────────────────────────────────────\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Speech quality comparison (ref vs hyp)")
    parser.add_argument("--ref", required=True, help="Reference WAV (PyTorch baseline)")
    parser.add_argument("--hyp", required=True, help="Hypothesis WAV (exported model)")
    parser.add_argument("--rtf", type=float, default=None, help="Real-time factor (optional)")
    parser.add_argument("--json", default=None, help="Save JSON results to this path")
    parser.add_argument("--quiet", action="store_true", help="Suppress table output")
    args = parser.parse_args()

    results = compare(args.ref, args.hyp, rtf=args.rtf, verbose=not args.quiet)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {out}")

    sys.exit(0 if results["all_pass"] else 1)


if __name__ == "__main__":
    main()
