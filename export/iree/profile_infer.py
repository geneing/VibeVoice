#!/usr/bin/env python3
"""
export/iree/profile_infer.py — per-stage timing profiler for the IREE pipeline.

Usage:
    uv run python profile_infer.py \
        --text "Custom agents enabled." \
        --voice ../../demo/voices/streaming_model/en-Carter_man.pt \
        --backend vulkan

Prints a detailed breakdown of time spent in each stage.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT  = Path(__file__).resolve().parents[2]
EXPORT_DIR = Path(__file__).parent
MODEL_PATH = REPO_ROOT / "model"
sys.path.insert(0, str(REPO_ROOT))

# Import everything from infer.py
from infer import (
    IREEVibeVoiceInference,
    DDPM_TIMESTEPS,
    HIDDEN_SIZE,
    LATENT_DIM,
    N_LM_LAYERS,
    N_TTS_LAYERS,
    TEXT_WINDOW,
    SPEECH_WINDOW,
    CFG_SCALE_DEF,
    SAMPLE_RATE,
    _cast_fp,
    _call,
    dpm_solver_step,
)


class Timer:
    """Simple accumulating timer."""
    def __init__(self):
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int]   = defaultdict(int)

    def __call__(self, name: str):
        return self._Ctx(self, name)

    class _Ctx:
        def __init__(self, timer, name):
            self._t = timer
            self._n = name
        def __enter__(self):
            self._s = time.perf_counter()
            return self
        def __exit__(self, *_):
            self._t.totals[self._n] += time.perf_counter() - self._s
            self._t.counts[self._n] += 1

    def report(self, total_elapsed: float):
        print("\n── Stage Timing Breakdown ─────────────────────────────────────")
        rows = sorted(self.totals.items(), key=lambda x: -x[1])
        for name, t in rows:
            n     = self.counts[name]
            avg   = t / n * 1000
            pct   = t / total_elapsed * 100
            print(f"  {name:<40s}  {t:6.2f}s  {avg:7.1f}ms/call  ×{n:4d}  ({pct:.1f}%)")
        print(f"  {'TOTAL':<40s}  {total_elapsed:6.2f}s")
        print("───────────────────────────────────────────────────────────────\n")


class ProfilingEngine(IREEVibeVoiceInference):
    """Subclass that instruments every IREE call with timing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = Timer()

    def _run_text_lm(self, input_ids, cache_position, lm_keys, lm_values):
        with self.timer("iree_text_lm"):
            return super()._run_text_lm(input_ids, cache_position, lm_keys, lm_values)

    def _run_tts_lm(self, lm_hidden, tts_mask, cache_position, tts_keys, tts_values):
        with self.timer("iree_tts_lm"):
            return super()._run_tts_lm(lm_hidden, tts_mask, cache_position, tts_keys, tts_values)

    def _run_diffusion_step(self, noisy, timestep, condition):
        with self.timer("iree_diffusion"):
            return super()._run_diffusion_step(noisy, timestep, condition)

    def _run_connector(self, latent):
        with self.timer("iree_connector"):
            return super()._run_connector(latent)

    def _run_vocoder(self, all_latents):
        with self.timer("iree_vocoder"):
            return super()._run_vocoder(all_latents)

    def _load_voice(self, voice_path):
        with self.timer("load_voice_pt"):
            return super()._load_voice(voice_path)

    def generate(self, text, voice_path, cfg_scale=CFG_SCALE_DEF,
                 max_speech_tokens=512, verbose=True):
        t0 = time.perf_counter()

        with self.timer("load_voice"):
            voice = self._load_voice(voice_path)

        with self.timer("tokenise"):
            preset = voice["preset"]   # reuse already-loaded preset
            inputs = self._proc.process_input_with_cached_prompt(
                text=text,
                cached_prompt=preset,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
        tts_text_ids = inputs["tts_text_ids"].numpy().astype(np.int64)
        total_tokens = tts_text_ids.shape[1]
        print(f"  {total_tokens} text tokens (max_speech_tokens={max_speech_tokens})")

        lm_keys      = [k.copy() for k in voice["lm_keys"]]
        lm_values    = [v.copy() for v in voice["lm_values"]]
        tts_keys     = [k.copy() for k in voice["tts_keys"]]
        tts_values   = [v.copy() for v in voice["tts_values"]]
        neg_tts_keys   = [k.copy() for k in voice["neg_tts_keys"]]
        neg_tts_values = [v.copy() for v in voice["neg_tts_values"]]

        tts_last_hidden     = voice["tts_last_hidden"].copy()
        neg_tts_last_hidden = voice["neg_tts_last_hidden"].copy()

        lm_pos      = voice["lm_kv_seq_len"]
        tts_pos     = voice["tts_kv_seq_len"]
        neg_tts_pos = voice["neg_tts_kv_seq_len"]

        all_latents: list[np.ndarray] = []
        win_idx  = 0
        finished = False
        torch.manual_seed(0)
        np.random.seed(0)

        while not finished:
            start = win_idx * TEXT_WINDOW
            end   = min(start + TEXT_WINDOW, total_tokens)
            cur_ids = tts_text_ids[:, start:end]
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

                # NOTE: neg path is NOT run through text tokens (matches PyTorch reference).

            for _ in range(SPEECH_WINDOW):
                pos_cond = tts_last_hidden[:, -1:, :].reshape(1, HIDDEN_SIZE)
                neg_cond = neg_tts_last_hidden[:, -1:, :].reshape(1, HIDDEN_SIZE)

                with self.timer("diffusion_total"):
                    # 5 diffusion steps (calls to _run_diffusion_step are timed inside)
                    condition = np.concatenate([pos_cond, neg_cond], axis=0)
                    speech = np.random.randn(2, LATENT_DIM).astype(np.float32)
                    m_list: list = []
                    for step_idx, t_step in enumerate(DDPM_TIMESTEPS):
                        half     = speech[:1]
                        combined = np.concatenate([half, half], axis=0)
                        v_pred_raw = self._run_diffusion_step(combined, t_step, condition)
                        cond_eps, uncond_eps = v_pred_raw[:1], v_pred_raw[1:]
                        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                        v_pred = np.concatenate([guided_eps, guided_eps], axis=0)
                        speech = dpm_solver_step(speech, m_list, step_idx, self._sigmas, v_pred)
                    latent = speech[:1].astype(np.float32)

                all_latents.append(latent)

                embed = self._run_connector(latent.reshape(1, 1, LATENT_DIM))

                speech_mask      = np.zeros((1, 1), dtype=np.int64)
                sp_cache_pos     = np.array([tts_pos],     dtype=np.int64)
                neg_sp_cache_pos = np.array([neg_tts_pos], dtype=np.int64)

                tts_last_hidden, eos_logit, tts_keys, tts_values = self._run_tts_lm(
                    embed, speech_mask, sp_cache_pos, tts_keys, tts_values
                )
                neg_tts_last_hidden, _, neg_tts_keys, neg_tts_values = self._run_tts_lm(
                    embed, speech_mask, neg_sp_cache_pos, neg_tts_keys, neg_tts_values
                )
                tts_pos     += 1
                neg_tts_pos += 1

                eos_prob = float(1.0 / (1.0 + np.exp(-float(eos_logit.flat[0]))))
                all_text_done = (win_idx * TEXT_WINDOW >= total_tokens)
                min_speech = max(total_tokens, 6)
                if eos_prob > 0.5 and len(all_latents) >= min_speech and all_text_done:
                    print(f"  EOS at speech token {len(all_latents)} (prob={eos_prob:.3f})")
                    finished = True
                    break
                if len(all_latents) >= max_speech_tokens:
                    print(f"  Safety stop at {len(all_latents)} speech tokens")
                    finished = True
                    break

        print(f"  Decoding {len(all_latents)} frames via vocoder …")
        waveform = self._run_vocoder(all_latents)

        total = time.perf_counter() - t0
        self.timer.report(total)
        return waveform


def main():
    parser = argparse.ArgumentParser(description="VibeVoice IREE inference profiler")
    parser.add_argument("--text",   required=True)
    parser.add_argument("--voice",  required=True)
    parser.add_argument("--output", default="output_profile.wav")
    parser.add_argument("--backend", choices=["cpu","vulkan"], default="vulkan")
    parser.add_argument("--fp16",   action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_speech_tokens", type=int, default=512)
    args = parser.parse_args()

    engine = ProfilingEngine(
        vmfb_dir=EXPORT_DIR, backend=args.backend, fp16=args.fp16
    )
    t0 = time.perf_counter()
    waveform = engine.generate(
        text=args.text,
        voice_path=args.voice,
        max_speech_tokens=args.max_speech_tokens,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    try:
        import soundfile as sf
        sf.write(args.output, waveform, SAMPLE_RATE)
    except ImportError:
        import scipy.io.wavfile as wavfile
        wavfile.write(args.output, SAMPLE_RATE, waveform)

    duration = len(waveform) / SAMPLE_RATE
    print(f"Saved {args.output}  ({duration:.2f}s audio, {elapsed:.2f}s wall, RTF={elapsed/duration:.2f}x)")


if __name__ == "__main__":
    main()
