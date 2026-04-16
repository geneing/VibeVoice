# AGENTS — VibeVoice Mobile Export

This file documents the conventions, workflows, and decision rules that AI coding
agents should follow when working in this repository. Read this file before
generating or modifying any code.

---

## Repository Purpose

Export `microsoft/VibeVoice-Realtime-0.5B` for accelerated inference on Android
(GPU / NPU) via ONNX Runtime, IREE, or LiteRT. See
[`.github/copilot-instructions.md`](.github/copilot-instructions.md) for full
project context.

---

## Hard Rules

| Rule | Detail |
|------|--------|
| **Never modify `vibevoice/`** | Treat it as a read-only upstream dependency. |
| **Never use `pip install`** | Use `uv add` in the relevant subproject. |
| **Never commit binaries** | `.onnx`, `.vmfb`, `.tflite`, `.pt` model weights go in `.gitignore`; document how to regenerate them. |
| **Never skip listening tests** | Automated metrics alone are insufficient; always save paired `_ref.wav` / `_hyp.wav` files. |
| **Never add backend packages to root** | `onnxruntime`, `iree-*`, `tflite-*` belong in `export/<method>/pyproject.toml` only. |
| Commit locally to git after each major change | preserve project history by commiting code often, include detailed description of the changes. don't push to upstream. |
| **Python 3.12 only** | The root project requires `>=3.12,<3.13`. Match this in every subproject. |

---

## Project Structure

```
export/
  onnx/          # ONNX Runtime export (opset ≥ 17, targeting NNAPI / GPU EP)
  iree/          # IREE export via iree-turbine (torch.export → MLIR → .vmfb)
  litert/        # LiteRT / TFLite export (future)
eval/
  compare.py     # Automated speech metric comparison
  samples/       # WAV output pairs: <method>/<speaker>_<id>_ref.wav / _hyp.wav
demo/
  realtime_inference.py   # Canonical PyTorch baseline — do not break this
  voices/streaming_model/ # Voice embeddings (.pt)
nenad102_onnx/   # Reference ONNX implementation (pure numpy runtime, no PyTorch)
```

---

## Starting a New Export Subproject

```bash
mkdir -p export/<method>
cd export/<method>
uv init --no-workspace
uv add <backend-specific-packages>
```

Create two entry-point scripts:

| Script | Purpose |
|--------|---------|
| `export.py` | One-shot export: load PyTorch model → write artefact(s) |
| `eval.py` | Run inference with exported model, compare to baseline, emit metrics |

Both must be deterministic (fix all RNG seeds, `model.eval()`, disable dropout).

After export, write an `export_manifest.json` in the same directory:

```json
{
  "method": "onnx",
  "timestamp": "2026-04-16T00:00:00Z",
  "opset": 17,
  "components": ["text_lm_kv", "tts_lm_kv", "acoustic_connector", "diffusion_head", "vocoder"],
  "shape_signatures": { "text_lm_kv": {"input_ids": ["batch", "seq"]} },
  "quantisation": null
}
```

---

## Existing Reference: `nenad102_onnx/`

A working 5-component ONNX split already exists. Before writing a new ONNX
export, study `nenad102_onnx/vibevoice_full_onnx.py` to understand:

- How the five components are split and chained.
- The KV-cache I/O conventions (`past_key_values_*`).
- How voice presets are packed into `.npz` arrays.
- The pure-numpy DPM-Solver++ scheduler implementation.

The new `export/onnx/` subproject should improve on this by:
1. Targeting mobile constraints (NNAPI-compatible ops, opset ≥ 17).
2. Labelling all dynamic axes.
3. Passing `onnxruntime.tools.check_onnx_model` with zero warnings.
4. Including a proper eval harness against the PyTorch baseline.

---

## Model Components

| Component | Input(s) | Output(s) | Notes |
|-----------|----------|-----------|-------|
| `text_lm` | `input_ids`, `attention_mask`, `past_key_values` | `logits`, `past_key_values` | Qwen2.5-0.5B, KV-cached |
| `tts_lm` | speech token ids, `past_key_values` | `logits`, `past_key_values` | autoregressive |
| `acoustic_connector` | LM hidden states | projected features | simple MLP |
| `diffusion_head` | noisy latent, timestep, conditioning | denoised latent | 5-step DDPM; needs two passes per step for CFG |
| `vocoder` | mel/latent features | waveform (24 kHz) | |

CFG (`cfg_scale=1.5`): concatenate unconditional and conditional batch on the first
axis, run a single forward pass, then apply:
```
output = uncond + cfg_scale * (cond - uncond)
```

---

## Evaluation Protocol

### Automated metrics (run via `eval/compare.py`)

| Metric | Library | Pass threshold |
|--------|---------|----------------|
| PESQ (WB) | `pesq` | ≥ 3.0 |
| STOI | `pystoi` | ≥ 0.85 |
| MCD | `pymcd` | ≤ 5.0 dB |
| SI-SNR | `torchmetrics.audio` | ≥ 15 dB |
| RTF | wall-clock / audio duration | ≤ 1.0 (CUDA), ≤ 3.0 (CPU) |

### Human listening

- Save `eval/samples/<method>/<speaker>_<id>_ref.wav` and `..._hyp.wav` for every test utterance.
- Test set: ≥ 5 sentences × ≥ 3 speakers (use voices from `demo/voices/streaming_model/`).
- Listen to all pairs before marking an export method as production-ready.

### Numerical parity

- Extract and compare intermediate activations (acoustic tokens, diffusion latents).
- Pass criterion: per-sample L∞ error < 0.02 (FP32 reference vs. exported FP32).
- Failures here indicate an incorrect export, not just quality degradation.

---

## Android Constraints per Backend

### ONNX Runtime + NNAPI

- Opset ≥ 17.
- All dynamic axes must be named (batch, sequence, etc.).
- No custom ops; use only ops in the NNAPI allowlist.
- Run `python -m onnxruntime.tools.check_onnx_model <file.onnx>` — zero warnings required.
- Prefer FP16 where NNAPI supports it; fall back to FP32 for unsupported ops.

### IREE (iree-turbine)

- Export path: `torch.export.export(model, ...)` → `iree.turbine.aot.export(...)` → `.mlir` → compile to `.vmfb`.
- Initial correctness target: `llvm-cpu`.
- Mobile GPU target: `vulkan-spirv`.
- Quantisation: FP16 via `--iree-input-demote-f64-to-f32` + `--iree-opt-const-eval`.

### LiteRT / TFLite (future)

- Convert via TFLite converter with FP16 or INT8 post-training quantisation.
- Calibration dataset: at least 100 representative text/voice pairs.
- Validate all ops against the GPU delegate op allowlist.

---

## Workflow for Adding an Export Method

1. Create `export/<method>/` and initialise a uv subproject.
2. Write `export.py` — loads PyTorch model, exports artefact(s), writes `export_manifest.json`.
3. Write `eval.py` — loads exported model (no PyTorch), runs test sentences, calls `eval/compare.py`, saves WAVs.
4. Write `README.md` covering: prerequisites, export steps, on-device deployment, known limitations.
5. Verify: `uv run python export.py` completes without error.
6. Verify: `uv run python eval.py` passes all automated metric thresholds.
7. Listen to all output WAV pairs before merging.

---

## What Not to Do

- Do not add features or refactor code beyond what is needed for the current task.
- Do not modify `vibevoice/` source files under any circumstances.
- Do not commit `.onnx`, `.vmfb`, `.tflite`, or `.pt` weight files.
- Do not use `pip install`; always `uv add`.
- Do not declare an export method production-ready based on metrics alone — listening
  tests are mandatory.
- Do not add IREE / ONNX / LiteRT packages to the root `pyproject.toml`.
