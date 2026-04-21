# Copilot Instructions — VibeVoice Mobile Export

## Project Goal

Export **VibeVoice-Realtime-0.5B** (`microsoft/VibeVoice-Realtime-0.5B`) for accelerated
inference on Android mobile hardware (GPU / NPU). The exported model must be loadable from
a Kotlin Android app using one or more of:

- **ONNX Runtime** (with NNAPI or GPU execution provider)
- **IREE Runtime** (via `iree-turbine` export)
- **LiteRT** (Google AI Edge / TFLite successor)

Both quality and latency matter. First-token latency ≤ 300 ms on mid-range Android hardware
is the target for real-time streaming use.

---

## Repository Layout

```
vibevoice/            # Core PyTorch model & processor (upstream, do not modify)
demo/
  realtime_inference.py   # PyTorch baseline — the canonical reference
  voices/streaming_model/ # Voice embeddings (.pt files)
nenad102_onnx/        # Existing ONNX export (5-component split, numpy runtime)
export/
  onnx/               # New ONNX export targeting mobile (ORT + NNAPI)
  iree/               # IREE export via iree-turbine  ← working end-to-end on Vulkan
    export.py         # torch.export → MLIR → .vmfb for all 5 components
    infer.py          # IREE inference engine (CPU + Vulkan, fp16 GPU)
    wrappers.py       # nn.Module wrappers + ChunkedConvRMSNorm for Vulkan compat
    export_manifest.json
  litert/             # LiteRT / TFLite export (future)
eval/                 # Shared evaluation harness
```

Each export lives in its own subdirectory under `export/`. Each is an independent
**uv subproject** with its own `pyproject.toml` and `.venv/`. Use `uv init --no-workspace`
or git worktrees to prevent dependency conflicts between export backends.

---

## Model Architecture (VibeVoice-Realtime-0.5B)

The model is a streaming TTS pipeline with five neural components:

| Component | Class / File | Role |
|---|---|---|
| **text_lm** | `modeling_vibevoice_streaming.py` | Qwen2.5-0.5B LLM, text → speech tokens (KV-cached) |
| **tts_lm** | same | Autoregressive TTS token generation |
| **acoustic_connector** | `modular_vibevoice_diffusion_head.py` | Projects LM hidden states |
| **diffusion_head** | same | 5-step DDPM denoiser |
| **vocoder** | same | Neural vocoder → 24 kHz waveform |

Key inference details (see `demo/realtime_inference.py`):
- Python ≥ 3.12 required (`pyproject.toml`)
- Classifier-free guidance (`cfg_scale=1.5`): requires two forward passes per step
- DDPM uses 5 inference steps (`model.set_ddpm_inference_steps(num_steps=5)`)
- Voice conditioning loaded from `.pt` (PyTorch) or `.npz` (ONNX variant) embeddings
- Output: mono 24 kHz WAV
- Processor: `VibeVoiceStreamingProcessor` (tokeniser + audio I/O)
- Model input is variable length (dynamic axes required for export). Intermediate tensors may also be variable length due to some sequence lengths changing based on the input text content influencing prosody.

---

## Baseline

`demo/realtime_inference.py` is the reference implementation. All exports must:
1. Accept the same inputs (text string + voice embedding)
2. Produce audio that passes evaluation against the PyTorch baseline
3. Be benchmarkable in isolation (no PyTorch dependency at inference time)

---

## Dependency Management

- Use **`uv`** exclusively (no pip, no conda, no poetry).
- Each export subproject has its own `pyproject.toml` + `uv.lock`.
- Root project dependencies remain in the top-level `pyproject.toml`.
- Never add export-backend packages (onnxruntime, iree-*, tflite-*) to the root project.
- Use `uv add` to install new packages within each export subproject.


---

## Evaluation

### Metrics (automated)
Run `eval/compare.py --ref <pytorch_wav> --hyp <exported_wav>`:

| Metric | Tool | Threshold |
|---|---|---|
| PESQ (wideband) | `pesq` | ≥ 3.0 |
| STOI | `pystoi` | ≥ 0.85 |
| MCD (Mel Cepstral Distortion) | `pymcd` | ≤ 5.0 dB |
| SI-SNR | `torchmetrics.audio` | ≥ 15 dB |
| RTF (Real-Time Factor) | timed wall-clock | ≤ 1.0 on CUDA, ≤ 3.0 on CPU |

### Human review
- Always save reference and hypothesis WAV files side by side for listening.
- Naming convention: `eval/samples/<method>/<speaker>_<text_id>_ref.wav` and `..._hyp.wav`.
- Use a diverse test set: at least 5 sentences × 3 speakers.
- Never rely solely on automated metrics — listening tests are mandatory before declaring
  an export method production-ready.

### Numerical correctness
- Compare intermediate activations (acoustic tokens, diffusion output) between PyTorch and
  exported model to isolate where degradation occurs.
- Tolerance: per-sample L∞ < 0.02 for lossless numeric parity tests.

---

## Android Integration

The Kotlin Android layer is out of scope for this repository but the exports must satisfy:

- **ONNX export**: opset ≥ 17, dynamic batch/sequence axes labelled, no unsupported ops for
  NNAPI EP. Run `python -m onnxruntime.tools.check_onnx_model` and fix all warnings.
- **IREE export**: all five components compile and run on `vulkan-spirv` with
  `--vulkan-target valhall4`. Use `iree-turbine` (`torch.export` → MLIR → IREE).
  FP16 Vulkan is achieved by casting the wrapper to `.half()` before export — do **not**
  use `--iree-input-demote-f32-to-f16`. See `export/iree/` for the full export pipeline
  including all graph patches required to make torch.export work.
- **LiteRT**: quantise to INT8 / FP16 using representative calibration data; confirm ops
  are supported by the GPU delegate.
- Provide a `README.md` per export method documenting: how to export, how to run on-device,
  known limitations.

---

## Code Style & Conventions

- Python 3.12. Type hints everywhere in new code.
- Do not modify files under `vibevoice/` (upstream source).
- Scripts in `export/<method>/` should be runnable as:
  ```bash
  cd export/<method>
  uv run python export.py        # one-shot export
  uv run python eval.py          # eval against baseline
  ```
- Export scripts must be deterministic (seed RNG, disable dropout).
- Log export metadata (opset, shape signatures, quantisation config) to
  `export/<method>/export_manifest.json`.

---

## Do Not Do

- Do not add ONNX / IREE / LiteRT packages to the root `pyproject.toml`.
- Do not modify `vibevoice/` source files.
- Do not skip human listening evaluation.
- Do not commit large binary model files (`.onnx`, `.vmfb`, `.tflite`) — add them to
  `.gitignore` and document download/export instructions instead.
- Do not use `pip install` — always `uv add`.
