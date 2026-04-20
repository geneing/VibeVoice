#!/usr/bin/env python3
"""
export/iree/infer.py — inference orchestration using IREE .vmfb components.

Mirrors the generate_speech() API from demo/realtime_inference.py.
All five components are called via IREE runtime; no PyTorch at inference time.
Windowing, EOS detection, DPM-Solver++ scheduling, and CFG are in Python/numpy.

Usage:
    uv run python infer.py \
        --text "Hello world" \
        --voice ../../demo/voices/streaming_model/en-Carter_man.pt \
        --output output_iree.wav \
        --backend cpu          # or: vulkan

    uv run python export/iree/infer.py --help
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[2]
EXPORT_DIR = Path(__file__).parent
MODEL_PATH = REPO_ROOT / "model"

sys.path.insert(0, str(REPO_ROOT))

# ── Constants (from nenad102_onnx/config.json) ────────────────────────────────
HIDDEN_SIZE   = 896
LATENT_DIM    = 64
N_LM_LAYERS   = 4
N_TTS_LAYERS  = 20
N_KV_HEADS    = 2
HEAD_DIM      = 64
SAMPLE_RATE   = 24000
TEXT_WINDOW   = 5
SPEECH_WINDOW = 6
CFG_SCALE_DEF = 1.5

# Diffusion schedule: 5-step v-prediction DPM-Solver++
# Timestep indices from the DDPM 1000-step training schedule
DDPM_TIMESTEPS = [999, 799, 599, 400, 200]


# ── DPM-Solver++ scheduler (pure numpy, ported from nenad102_onnx) ────────────

def _sigma_to_alpha_sigma(sigma: float):
    alpha = 1.0 / math.sqrt(1.0 + sigma ** 2)
    return alpha, sigma * alpha


def _sigma_to_lambda(sigma: float) -> float:
    alpha, sigma_t = _sigma_to_alpha_sigma(sigma)
    if sigma_t < 1e-10:
        return 20.0
    return math.log(alpha / sigma_t)


def _compute_sigmas(timesteps: list[int], betas: np.ndarray) -> list[float]:
    """Compute sigma schedule for given timestep indices from DDPM betas."""
    alphas_cumprod = np.cumprod(1.0 - betas)
    sigmas = []
    for t in timesteps:
        alpha_bar = float(alphas_cumprod[t])
        sigmas.append(math.sqrt((1.0 - alpha_bar) / alpha_bar))
    sigmas.append(0.0)  # final sigma = 0
    return sigmas


def dpm_solver_step(
    sample:     np.ndarray,  # (2, 64)  current noisy latent (2 copies for CFG)
    m_list:     list,
    step_idx:   int,
    sigmas:     list[float],
    v_pred:     np.ndarray,  # (2, 64) velocity prediction (after CFG)
) -> np.ndarray:
    """One DPM-Solver++ midpoint update (v-prediction, lower_order_final)."""
    sample = sample.astype(np.float64)

    sig      = sigmas[step_idx]
    sig_next = sigmas[step_idx + 1]
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma(sig)
    alpha_t,  sigma_t  = _sigma_to_alpha_sigma(sig_next)
    lam_t  = _sigma_to_lambda(sig_next)
    lam_s0 = _sigma_to_lambda(sig)

    # Convert v-prediction to x0 prediction
    x0 = alpha_s0 * sample - sigma_s0 * v_pred.astype(np.float64)
    m_list.append(x0.copy())

    h = lam_t - lam_s0
    if step_idx == 0 or step_idx == 4:
        # First-order update (Euler)
        sample = (sigma_t / sigma_s0) * sample - alpha_t * np.expm1(-h) * x0
    else:
        # Second-order update (midpoint)
        sig_prev = sigmas[step_idx - 1]
        lam_s1   = _sigma_to_lambda(sig_prev)
        h_0 = lam_s0 - lam_s1
        r0  = h_0 / h
        D1  = (1.0 / r0) * (m_list[-1] - m_list[-2])
        sample = (
            (sigma_t / sigma_s0) * sample
            - alpha_t * np.expm1(-h) * m_list[-1]
            - 0.5 * alpha_t * np.expm1(-h) * D1
        )
    return sample.astype(np.float32)


# ── IREE runtime helpers ──────────────────────────────────────────────────────

IREE_DRIVER_MAP = {
    "cpu":    "local-task",
    "vulkan": "vulkan",
}


def _load_vmfb(path: str | Path, driver: str):
    """Load a .vmfb and return a callable module."""
    import iree.runtime as ireert
    with open(str(path), "rb") as f:
        bytecode = f.read()
    config = ireert.Config(driver)
    return ireert.load_vm_module(
        ireert.VmModule.copy_buffer(config.vm_instance, bytecode),
        config,
    )


def _to_np(t) -> np.ndarray:
    """Convert torch.Tensor to numpy, handling device/dtype."""
    import torch
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


def _call(module, *args: np.ndarray) -> tuple[np.ndarray, ...]:
    """Call an IREE module.main() and return float32 numpy outputs as a tuple.

    Outputs are always cast to float32 so that downstream Python (DPM solver,
    CFG math, EOS logistic) works regardless of whether the module is fp16.
    """
    result = module.main(*args)
    # iree.runtime may return a single array or a tuple
    if isinstance(result, (list, tuple)):
        return tuple(np.array(r, dtype=np.float32) if np.array(r).dtype.kind == 'f'
                     else np.array(r) for r in result)
    arr = np.array(result)
    return (arr.astype(np.float32) if arr.dtype.kind == 'f' else arr,)


def _cast_fp(arr: np.ndarray, fp16: bool) -> np.ndarray:
    """Cast a float array to float16 or float32 depending on fp16 flag."""
    if arr.dtype.kind != 'f':
        return arr  # integers pass through unchanged
    return arr.astype(np.float16 if fp16 else np.float32)


# ── KV cache helpers ──────────────────────────────────────────────────────────

def _extract_kv_from_cache(past_kv, n_layers: int) -> tuple[list, list]:
    """
    Extract flat numpy KV arrays from a DynamicCache (torch) or tuple-of-tuples.
    Returns (keys, values) each a list of n_layers float32 numpy arrays.
    """
    import torch
    keys:   list[np.ndarray] = []
    values: list[np.ndarray] = []
    try:
        # DynamicCache
        for i in range(n_layers):
            keys.append(_to_np(past_kv.key_cache[i]))
            values.append(_to_np(past_kv.value_cache[i]))
    except AttributeError:
        # Tuple-of-tuples
        for i in range(n_layers):
            k, v = past_kv[i]
            keys.append(_to_np(k))
            values.append(_to_np(v))
    return keys, values


def _flat_kv(*keys_values_interleaved: np.ndarray) -> list[np.ndarray]:
    """Build interleaved [k0, v0, k1, v1, ...] list from alternating lists."""
    return list(keys_values_interleaved)


def _kv_concat_new(
    old_keys:   list[np.ndarray],
    old_values: list[np.ndarray],
    new_kv_flat: tuple[np.ndarray, ...],
    n_layers:   int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    The IREE wrapper returns the FULL updated KV (old + new concatenated).
    Just return the new flat KV as the updated state.
    """
    new_keys   = [new_kv_flat[2*i]   for i in range(n_layers)]
    new_values = [new_kv_flat[2*i+1] for i in range(n_layers)]
    return new_keys, new_values


# ── Core inference class ──────────────────────────────────────────────────────

class IREEVibeVoiceInference:
    """
    Inference engine that calls IREE .vmfb components in the same sequence as
    the PyTorch generate() method in modeling_vibevoice_streaming_inference.py.
    """

    def __init__(
        self,
        vmfb_dir: str | Path,
        backend: str = "cpu",
        fp16: bool = True,
    ) -> None:
        driver = IREE_DRIVER_MAP.get(backend, "local-task")
        d = Path(vmfb_dir)

        # Vulkan FP16 artefacts use the suffix _vulkan_fp16 (set by export.py).
        # CPU artefacts are always float32 regardless of the fp16 flag.
        is_fp16_vulkan = (backend == "vulkan" and fp16)
        suffix = f"{backend}_fp16" if is_fp16_vulkan else backend
        self._fp16 = is_fp16_vulkan   # used to cast inputs before IREE calls

        print(f"Loading IREE components (backend={backend}, fp16={is_fp16_vulkan}) …")
        self._text_lm   = _load_vmfb(d / f"text_lm_{suffix}.vmfb",            driver)
        self._tts_lm    = _load_vmfb(d / f"tts_lm_{suffix}.vmfb",             driver)
        self._connector = _load_vmfb(d / f"acoustic_connector_{suffix}.vmfb",  driver)
        self._diff_head = _load_vmfb(d / f"diffusion_head_{suffix}.vmfb",      driver)
        self._vocoder   = _load_vmfb(d / f"vocoder_{suffix}.vmfb",             driver)
        print("  All components loaded.")

        # Load DPM betas from nenad102_onnx (pre-computed)
        betas_path = REPO_ROOT / "nenad102_onnx" / "betas.npy"
        if betas_path.exists():
            betas = np.load(str(betas_path))
        else:
            # Cosine schedule fallback (matches DPMSolverMultistepScheduler defaults)
            betas = self._cosine_betas(1000)
        self._sigmas = _compute_sigmas(DDPM_TIMESTEPS, betas)

    @staticmethod
    def _cosine_betas(n: int, max_beta: float = 0.999) -> np.ndarray:
        """Cosine beta schedule (matches VibeVoice training config)."""
        betas = []
        for i in range(n):
            t1 = i / n
            t2 = (i + 1) / n
            b = min(1 - (math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2)
                      / (math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2), max_beta)
            betas.append(b)
        return np.array(betas, dtype=np.float64)

    # ── Text-LM step ──────────────────────────────────────────────────────────

    def _run_text_lm(
        self,
        input_ids:      np.ndarray,   # (1, seq) int64
        cache_position: np.ndarray,   # (seq,) int64
        lm_keys:        list[np.ndarray],
        lm_values:      list[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Returns (last_hidden_state, new_lm_keys, new_lm_values)."""
        flat_kv = [_cast_fp(a, self._fp16) for pair in zip(lm_keys, lm_values) for a in pair]
        outs = _call(self._text_lm, input_ids, cache_position, *flat_kv)
        hidden = outs[0]
        new_keys   = [outs[1 + 2*i] for i in range(N_LM_LAYERS)]
        new_values = [outs[2 + 2*i] for i in range(N_LM_LAYERS)]
        return hidden, new_keys, new_values

    # ── TTS-LM step ───────────────────────────────────────────────────────────

    def _run_tts_lm(
        self,
        lm_hidden:      np.ndarray,   # (1, seq, hidden)
        tts_mask:       np.ndarray,   # (1, seq) int64
        cache_position: np.ndarray,   # (seq,) int64
        tts_keys:       list[np.ndarray],
        tts_values:     list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Returns (last_hidden_state, eos_logit, new_keys, new_values)."""
        flat_kv = [_cast_fp(a, self._fp16) for pair in zip(tts_keys, tts_values) for a in pair]
        outs = _call(self._tts_lm, _cast_fp(lm_hidden, self._fp16), tts_mask, cache_position, *flat_kv)
        hidden    = outs[0]   # (1, seq, hidden)
        eos_logit = outs[1]   # (1, 1)
        new_keys   = [outs[2 + 2*i] for i in range(N_TTS_LAYERS)]
        new_values = [outs[3 + 2*i] for i in range(N_TTS_LAYERS)]
        return hidden, eos_logit, new_keys, new_values

    # ── Diffusion-head step ───────────────────────────────────────────────────

    def _run_diffusion_step(
        self,
        noisy:     np.ndarray,   # (2, 64)
        timestep:  int,
        condition: np.ndarray,   # (2, 896)
    ) -> np.ndarray:             # (2, 64)
        t = np.array([float(timestep), float(timestep)], dtype=np.float32)
        outs = _call(
            self._diff_head,
            _cast_fp(noisy, self._fp16),
            _cast_fp(t, self._fp16),
            _cast_fp(condition, self._fp16),
        )
        return outs[0]

    # ── Sample one speech latent via DPM-Solver++ with CFG ────────────────────

    def _sample_speech_token(
        self,
        pos_cond: np.ndarray,   # (1, 896)
        neg_cond: np.ndarray,   # (1, 896)
        cfg_scale: float,
    ) -> np.ndarray:             # (1, 64)
        # Concatenate pos + neg → batch=2 for the diffusion head
        condition = np.concatenate([pos_cond, neg_cond], axis=0)  # (2, 896)
        # Initialise noise
        speech = np.random.randn(2, LATENT_DIM).astype(np.float32)

        m_list: list = []
        for step_idx, t in enumerate(DDPM_TIMESTEPS):
            # The wrapper always receives batch=2 duplicates: [half, half]
            half     = speech[:1]                              # (1, 64)
            combined = np.concatenate([half, half], axis=0)   # (2, 64)

            v_pred_raw = self._run_diffusion_step(combined, t, condition)  # (2, 64)

            # CFG: split into cond / uncond
            cond_eps,  uncond_eps  = v_pred_raw[:1], v_pred_raw[1:]
            guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            v_pred = np.concatenate([guided_eps, guided_eps], axis=0)  # (2, 64)

            speech = dpm_solver_step(speech, m_list, step_idx, self._sigmas, v_pred)

        return speech[:1].astype(np.float32)  # positive half only: (1, 64)

    # ── Acoustic connector ────────────────────────────────────────────────────

    def _run_connector(self, latent: np.ndarray) -> np.ndarray:
        """latent: (1, 1, 64) → embed: (1, 1, 896)"""
        outs = _call(self._connector, _cast_fp(latent, self._fp16))
        return outs[0]

    # ── Vocoder ───────────────────────────────────────────────────────────────

    def _run_vocoder(self, all_latents: list[np.ndarray]) -> np.ndarray:
        """
        Decode a list of (1, 64) latents to a waveform.
        Concatenates into (1, N, 64) and calls the vocoder once.
        Returns a flat float32 waveform.
        """
        latent_seq = np.concatenate(all_latents, axis=0)           # (N, 64)
        latent_seq = latent_seq[np.newaxis, :, :]                   # (1, N, 64)
        outs = _call(self._vocoder, _cast_fp(latent_seq, self._fp16))
        waveform = outs[0].flatten().astype(np.float32)
        return waveform

    # ── Load voice preset ─────────────────────────────────────────────────────

    def _load_voice(self, voice_path: str | Path) -> dict:
        """
        Load a .pt voice preset and extract flat KV numpy arrays.

        Returns a dict with:
            lm_keys, lm_values                     (list of arrays, N_LM_LAYERS)
            tts_keys, tts_values                   (N_TTS_LAYERS)
            neg_tts_keys, neg_tts_values           (N_TTS_LAYERS)
            tts_last_hidden                        (1, ?, 896)
            neg_tts_last_hidden                    (1, ?, 896)
            lm_kv_seq_len, tts_kv_seq_len          (int)
        """
        import torch
        preset = torch.load(str(voice_path), map_location="cpu", weights_only=False)

        lm_keys,      lm_values      = _extract_kv_from_cache(
            preset["lm"].past_key_values, N_LM_LAYERS)
        tts_keys,     tts_values     = _extract_kv_from_cache(
            preset["tts_lm"].past_key_values, N_TTS_LAYERS)
        neg_tts_keys, neg_tts_values = _extract_kv_from_cache(
            preset["neg_tts_lm"].past_key_values, N_TTS_LAYERS)

        tts_last_hidden     = _to_np(preset["tts_lm"].last_hidden_state)
        neg_tts_last_hidden = _to_np(preset["neg_tts_lm"].last_hidden_state)

        return {
            "lm_keys":           lm_keys,
            "lm_values":         lm_values,
            "tts_keys":          tts_keys,
            "tts_values":        tts_values,
            "neg_tts_keys":      neg_tts_keys,
            "neg_tts_values":    neg_tts_values,
            "tts_last_hidden":   tts_last_hidden,
            "neg_tts_last_hidden": neg_tts_last_hidden,
            "lm_kv_seq_len":     lm_keys[0].shape[2] if lm_keys else 0,
            "tts_kv_seq_len":    tts_keys[0].shape[2] if tts_keys else 0,
        }

    # ── Tokenise input text ───────────────────────────────────────────────────

    def _tokenise(self, text: str, voice_path: str | Path) -> np.ndarray:
        """
        Tokenise text using the VibeVoice processor and return tts_text_ids.
        Returns int64 numpy array of shape (1, N_tokens).
        """
        import torch
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )
        model_path = str(MODEL_PATH)
        proc = VibeVoiceStreamingProcessor.from_pretrained(model_path)
        preset = torch.load(str(voice_path), map_location="cpu", weights_only=False)
        inputs = proc.process_input_with_cached_prompt(
            text=text,
            cached_prompt=preset,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        # tts_text_ids are the text token IDs that the TTS LM will process
        tts_text_ids = inputs.get("tts_text_ids", None)
        if tts_text_ids is None:
            raise ValueError("Processor did not return tts_text_ids")
        return tts_text_ids.numpy().astype(np.int64)

    # ── Main generation loop ──────────────────────────────────────────────────

    def generate(
        self,
        text:            str,
        voice_path:      str | Path,
        cfg_scale:       float = CFG_SCALE_DEF,
        max_speech_tokens: int = 512,
        verbose:         bool  = True,
    ) -> np.ndarray:
        """
        Generate speech from text using IREE components.

        Returns float32 numpy waveform at SAMPLE_RATE Hz.
        """
        import torch
        # Re-use the processor only for tokenisation; no PyTorch inference.
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        voice = self._load_voice(voice_path)

        # Tokenise
        proc = VibeVoiceStreamingProcessor.from_pretrained(str(MODEL_PATH))
        preset = torch.load(str(voice_path), map_location="cpu", weights_only=False)
        inputs = proc.process_input_with_cached_prompt(
            text=text,
            cached_prompt=preset,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        tts_text_ids = inputs["tts_text_ids"].numpy().astype(np.int64)  # (1, N)
        total_tokens = tts_text_ids.shape[1]

        if verbose:
            print(f"  {total_tokens} text tokens")

        # Initialise running KV state from voice preset
        lm_keys      = [k.copy() for k in voice["lm_keys"]]
        lm_values    = [v.copy() for v in voice["lm_values"]]
        tts_keys     = [k.copy() for k in voice["tts_keys"]]
        tts_values   = [v.copy() for v in voice["tts_values"]]
        neg_tts_keys   = [k.copy() for k in voice["neg_tts_keys"]]
        neg_tts_values = [v.copy() for v in voice["neg_tts_values"]]

        tts_last_hidden     = voice["tts_last_hidden"].copy()
        neg_tts_last_hidden = voice["neg_tts_last_hidden"].copy()

        lm_pos  = voice["lm_kv_seq_len"]
        tts_pos = voice["tts_kv_seq_len"]

        all_latents: list[np.ndarray] = []
        win_idx  = 0
        finished = False
        torch.manual_seed(0)  # deterministic noise
        np.random.seed(0)

        while not finished:
            # ── Text window ────────────────────────────────────────────────────
            start = win_idx * TEXT_WINDOW
            end   = min(start + TEXT_WINDOW, total_tokens)
            cur_ids = tts_text_ids[:, start:end]  # (1, window_size)
            win_idx += 1
            seq_len = cur_ids.shape[1]

            if seq_len > 0:
                cache_pos = np.arange(lm_pos, lm_pos + seq_len, dtype=np.int64)
                lm_hidden, lm_keys, lm_values = self._run_text_lm(
                    cur_ids, cache_pos, lm_keys, lm_values
                )
                lm_pos += seq_len

                tts_mask = np.ones((1, seq_len), dtype=np.int64)
                tts_cache_pos = np.arange(tts_pos, tts_pos + seq_len, dtype=np.int64)
                tts_last_hidden, _, tts_keys, tts_values = self._run_tts_lm(
                    lm_hidden, tts_mask, tts_cache_pos, tts_keys, tts_values
                )
                tts_pos += seq_len

            # ── Speech window ──────────────────────────────────────────────────
            for _ in range(SPEECH_WINDOW):
                pos_cond = tts_last_hidden[:, -1:, :].reshape(1, HIDDEN_SIZE)
                neg_cond = neg_tts_last_hidden[:, -1:, :].reshape(1, HIDDEN_SIZE)

                latent = self._sample_speech_token(pos_cond, neg_cond, cfg_scale)
                all_latents.append(latent)   # (1, 64)

                # Acoustic connector: latent (1,1,64) → embed (1,1,896)
                embed = self._run_connector(latent.reshape(1, 1, LATENT_DIM))

                # TTS LM — positive path (speech step)
                speech_mask = np.zeros((1, 1), dtype=np.int64)
                sp_cache_pos = np.array([tts_pos], dtype=np.int64)
                tts_last_hidden, eos_logit, tts_keys, tts_values = self._run_tts_lm(
                    embed, speech_mask, sp_cache_pos, tts_keys, tts_values
                )

                # TTS LM — negative path (speech step)
                neg_tts_last_hidden, _, neg_tts_keys, neg_tts_values = self._run_tts_lm(
                    embed, speech_mask, sp_cache_pos, neg_tts_keys, neg_tts_values
                )

                tts_pos += 1

                # EOS check (data-dependent Python control flow)
                eos_prob = float(1.0 / (1.0 + np.exp(-float(eos_logit.flat[0]))))
                all_text_done = (win_idx * TEXT_WINDOW >= total_tokens)
                min_speech = max(total_tokens, 6)
                if eos_prob > 0.5 and len(all_latents) >= min_speech and all_text_done:
                    if verbose:
                        print(f"  EOS at speech token {len(all_latents)} "
                              f"(prob={eos_prob:.3f})")
                    finished = True
                    break

                # Safety stop
                if len(all_latents) >= max_speech_tokens:
                    if verbose:
                        print(f"  Safety stop at {len(all_latents)} speech tokens")
                    finished = True
                    break

        if not all_latents:
            raise RuntimeError("No speech tokens generated")

        if verbose:
            print(f"  Decoding {len(all_latents)} frames via vocoder …")

        waveform = self._run_vocoder(all_latents)
        return waveform


# ── Convenience function (mirrors demo/realtime_inference.py interface) ───────

def generate_speech(
    text:            str,
    voice_path:      str | Path,
    output_path:     str | Path,
    vmfb_dir:        str | Path = EXPORT_DIR,
    backend:         str        = "cpu",
    fp16:            bool       = True,
    cfg_scale:       float      = CFG_SCALE_DEF,
    verbose:         bool       = True,
) -> str:
    """
    Generate speech using IREE .vmfb components and save to a WAV file.

    Args:
        text:        input text
        voice_path:  path to .pt voice preset
        output_path: where to save the output WAV
        vmfb_dir:    directory containing the .vmfb files
        backend:     "cpu" or "vulkan"
        cfg_scale:   classifier-free guidance scale
        verbose:     print progress

    Returns:
        path to the saved WAV file
    """
    try:
        import soundfile as sf
    except ImportError:
        import scipy.io.wavfile as _wav
        sf = None

    engine = IREEVibeVoiceInference(vmfb_dir=vmfb_dir, backend=backend, fp16=fp16)

    start = time.time()
    waveform = engine.generate(
        text=text,
        voice_path=voice_path,
        cfg_scale=cfg_scale,
        verbose=verbose,
    )
    elapsed = time.time() - start

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if sf is not None:
        sf.write(str(out), waveform, SAMPLE_RATE)
    else:
        import scipy.io.wavfile as wavfile
        wavfile.write(str(out), SAMPLE_RATE, waveform)

    duration = len(waveform) / SAMPLE_RATE
    rtf = elapsed / duration if duration > 0 else float("inf")
    if verbose:
        print(f"  Saved {out}  ({duration:.2f}s audio, {elapsed:.2f}s, RTF={rtf:.2f}x)")

    return str(out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VibeVoice IREE inference")
    parser.add_argument("--text",     required=True, help="Input text")
    parser.add_argument(
        "--voice", required=True,
        help="Path to .pt voice preset (e.g. demo/voices/streaming_model/en-Carter_man.pt)",
    )
    parser.add_argument("--output",  default="output_iree.wav", help="Output WAV path")
    parser.add_argument("--vmfb_dir", default=str(EXPORT_DIR), help="Dir with .vmfb files")
    parser.add_argument("--backend", choices=["cpu", "vulkan"], default="cpu")
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=True,
        help="Load *_vulkan_fp16.vmfb files and cast float inputs to float16 (default: True)",
    )
    parser.add_argument("--cfg_scale", type=float, default=CFG_SCALE_DEF)
    args = parser.parse_args()

    generate_speech(
        text=args.text,
        voice_path=args.voice,
        output_path=args.output,
        vmfb_dir=args.vmfb_dir,
        backend=args.backend,
        fp16=args.fp16,
        cfg_scale=args.cfg_scale,
        verbose=True,
    )


if __name__ == "__main__":
    main()
