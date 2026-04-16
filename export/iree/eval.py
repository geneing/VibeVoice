#!/usr/bin/env python3
"""
export/iree/eval.py — evaluate IREE export against the PyTorch baseline.

For every (sentence, speaker) pair in eval/test_corpus.json:
  1. Generate reference WAV using PyTorch baseline (demo/realtime_inference.py)
  2. Generate hypothesis WAV using IREE components (infer.py)
  3. Run eval/compare.py metrics on each pair
  4. Print a summary table and save results to eval/samples/iree/results.json

Usage:
    uv run python export/iree/eval.py
    uv run python export/iree/eval.py --backend vulkan
    uv run python export/iree/eval.py --skip_ref    # if ref WAVs already exist
    uv run python export/iree/eval.py --sentence s1 --speaker en-Carter_man
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT  = Path(__file__).resolve().parents[2]
EXPORT_DIR = Path(__file__).parent
EVAL_DIR   = REPO_ROOT / "eval"
SAMPLES_DIR = EVAL_DIR / "samples" / "iree"
CORPUS_PATH = EVAL_DIR / "test_corpus.json"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPORT_DIR))

MODEL_PATH  = REPO_ROOT / "model"
VOICES_DIR  = REPO_ROOT / "demo" / "voices" / "streaming_model"


# ── Baseline generation ───────────────────────────────────────────────────────

def generate_ref(
    text:        str,
    speaker:     str,
    out_path:    Path,
    model_path:  str,
    verbose:     bool = False,
) -> tuple[str, float]:
    """
    Run the PyTorch baseline and save to out_path.
    Returns (wav_path, rtf).
    """
    from demo.realtime_inference import generate_speech

    voice_path = str(VOICES_DIR / f"{speaker}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    generate_speech(
        text=text,
        model_path=model_path,
        voice_sample_path=voice_path,
        output_path=str(out_path),
        device="cpu",
        cfg_scale=1.5,
    )
    elapsed = time.time() - t0

    # Estimate audio duration from WAV
    try:
        import soundfile as sf
        info = sf.info(str(out_path))
        rtf = elapsed / info.duration
    except Exception:
        rtf = float("nan")

    if verbose:
        print(f"    [ref] {out_path.name}  elapsed={elapsed:.1f}s  rtf={rtf:.2f}")
    return str(out_path), rtf


# ── IREE hypothesis generation ────────────────────────────────────────────────

def generate_hyp(
    text:       str,
    speaker:    str,
    out_path:   Path,
    backend:    str,
    verbose:    bool = False,
) -> tuple[str, float]:
    """
    Run IREE inference and save to out_path.
    Returns (wav_path, rtf).
    """
    from infer import generate_speech as iree_generate

    voice_path = str(VOICES_DIR / f"{speaker}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    iree_generate(
        text=text,
        voice_path=voice_path,
        output_path=str(out_path),
        vmfb_dir=EXPORT_DIR,
        backend=backend,
        cfg_scale=1.5,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    try:
        import soundfile as sf
        info = sf.info(str(out_path))
        rtf = elapsed / info.duration
    except Exception:
        rtf = float("nan")

    if verbose:
        print(f"    [hyp] {out_path.name}  elapsed={elapsed:.1f}s  rtf={rtf:.2f}")
    return str(out_path), rtf


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_eval(
    backend:      str  = "cpu",
    skip_ref:     bool = False,
    sentence_ids: Optional[list[str]] = None,
    speakers:     Optional[list[str]] = None,
    model_path:   str  = str(MODEL_PATH),
) -> list[dict]:
    """Run evaluation for all (sentence, speaker) pairs. Returns list of result dicts."""
    from eval.compare import compare as compare_wavs

    corpus = json.loads(CORPUS_PATH.read_text())
    all_sentences = corpus["sentences"]
    all_speakers  = corpus["speakers"]

    if sentence_ids:
        all_sentences = [s for s in all_sentences if s["id"] in sentence_ids]
    if speakers:
        all_speakers = [s for s in all_speakers if s in speakers]

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total = len(all_sentences) * len(all_speakers)
    done  = 0

    print(f"\nEvaluating {len(all_sentences)} sentences × {len(all_speakers)} speakers "
          f"= {total} pairs  (backend={backend})\n")

    for speaker in all_speakers:
        for sentence in all_sentences:
            sid   = sentence["id"]
            text  = sentence["text"]
            desc  = sentence.get("desc", "")
            done += 1
            tag   = f"{speaker}_{sid}"

            print(f"[{done}/{total}]  {speaker}  |  {sid} ({desc})")
            print(f"  {text}")

            ref_path = SAMPLES_DIR / f"{tag}_ref.wav"
            hyp_path = SAMPLES_DIR / f"{tag}_hyp.wav"

            # ── Reference ────────────────────────────────────────────────────
            ref_rtf: Optional[float] = None
            if skip_ref and ref_path.exists():
                print(f"  [ref] skipped (already exists)")
            else:
                try:
                    _, ref_rtf = generate_ref(
                        text, speaker, ref_path, model_path, verbose=True
                    )
                except Exception as exc:
                    print(f"  [ref] FAILED: {exc}")
                    results.append({
                        "speaker": speaker, "sentence_id": sid,
                        "error_ref": str(exc),
                    })
                    continue

            # ── Hypothesis ───────────────────────────────────────────────────
            hyp_rtf: Optional[float] = None
            hyp_ok = False
            try:
                _, hyp_rtf = generate_hyp(text, speaker, hyp_path, backend, verbose=True)
                hyp_ok = True
            except Exception as exc:
                print(f"  [hyp] FAILED: {exc}")
                results.append({
                    "speaker": speaker, "sentence_id": sid,
                    "ref": str(ref_path), "error_hyp": str(exc),
                    "ref_rtf": ref_rtf,
                })
                continue

            # ── Metrics ──────────────────────────────────────────────────────
            try:
                metrics = compare_wavs(
                    ref_path=ref_path,
                    hyp_path=hyp_path,
                    rtf=hyp_rtf,
                    verbose=True,
                )
                metrics["speaker"]     = speaker
                metrics["sentence_id"] = sid
                metrics["desc"]        = desc
                metrics["ref_rtf"]     = ref_rtf
                results.append(metrics)
            except Exception as exc:
                print(f"  [metrics] FAILED: {exc}")
                results.append({
                    "speaker": speaker, "sentence_id": sid,
                    "ref": str(ref_path), "hyp": str(hyp_path),
                    "error_metrics": str(exc),
                    "hyp_rtf": hyp_rtf,
                })

    return results


def _print_summary(results: list[dict]) -> bool:
    """Print final summary table. Returns True if all automated tests pass."""
    ok_results = [r for r in results if "pesq" in r or "stoi" in r]
    err_results = [r for r in results if "error_ref" in r
                   or "error_hyp" in r or "error_metrics" in r]

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║              IREE Export Evaluation Summary                  ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    metrics_names = ["pesq", "stoi", "mcd", "si_snr"]
    if ok_results:
        # Per-row
        header = f"  {'pair':<32} {'PESQ':>6} {'STOI':>6} {'MCD':>6} {'SI-SNR':>7}  {'pass':>4}"
        print(header)
        print("  " + "─" * 68)
        all_pass = True
        for r in ok_results:
            tag   = f"{r.get('speaker','?')}_{r.get('sentence_id','?')}"
            vals  = [r.get(m) for m in metrics_names]
            flags = [r.get(f"{m}_pass") for m in metrics_names]
            row   = f"  {tag:<32}"
            for v in vals:
                row += f" {v:>6.3f}" if v is not None else f" {'n/a':>6}"
            ok = all(f is True for f in flags if f is not None)
            row += f"  {'✓' if ok else '✗':>4}"
            if not ok:
                all_pass = False
            print(row)

        # Aggregate
        print("  " + "─" * 68)
        for m in metrics_names:
            vals = [r[m] for r in ok_results if r.get(m) is not None]
            if vals:
                import statistics
                print(f"  {m:<10}  mean={statistics.mean(vals):>6.3f}  "
                      f"min={min(vals):>6.3f}  max={max(vals):>6.3f}")
    else:
        all_pass = False
        print("  No successful metric comparisons.")

    if err_results:
        print(f"\n  {len(err_results)} pair(s) encountered errors (see results.json).")
        all_pass = False

    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    if all_pass:
        print("  ALL AUTOMATED METRICS PASSED.")
    else:
        print("  SOME METRICS FAILED OR ERRORED — review results and listen to WAVs.")

    print(f"\n  WAV files saved in: {SAMPLES_DIR}")
    print("  ⚠  Listening tests are MANDATORY before declaring this export production-ready.")
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate IREE VibeVoice export against PyTorch baseline"
    )
    parser.add_argument("--backend", choices=["cpu", "vulkan"], default="cpu",
                        help="IREE backend to use for hypothesis generation")
    parser.add_argument("--skip_ref", action="store_true",
                        help="Skip reference generation if _ref.wav already exists")
    parser.add_argument("--sentence", dest="sentence_ids", nargs="+",
                        help="Run only these sentence IDs (e.g. s1 s2)")
    parser.add_argument("--speaker", dest="speakers", nargs="+",
                        help="Run only these speakers")
    parser.add_argument("--model_path", default=str(MODEL_PATH),
                        help="Path to local model directory")
    args = parser.parse_args()

    results = run_eval(
        backend=args.backend,
        skip_ref=args.skip_ref,
        sentence_ids=args.sentence_ids,
        speakers=args.speakers,
        model_path=args.model_path,
    )

    # Save results
    results_path = SAMPLES_DIR / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {results_path}")

    all_pass = _print_summary(results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
