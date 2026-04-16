#!/usr/bin/env python3
"""
export/iree/export.py — export all five VibeVoice components to IREE .vmfb artefacts.

Usage (from repo root):
    uv run python export/iree/export.py
    uv run python export/iree/export.py --backends cpu         # llvm-cpu only
    uv run python export/iree/export.py --backends vulkan      # vulkan-spirv only
    uv run python export/iree/export.py --backends cpu vulkan  # both (default)
    uv run python export/iree/export.py --component text_lm    # single component

Output files:
    export/iree/<component>_cpu.vmfb
    export/iree/<component>_vulkan.vmfb
    export/iree/export_manifest.json

The export uses torch.export.export (strict=False) → iree.turbine.aot.export →
iree.turbine.aot.ExportOutput.compile().

If torch.export fails for a component (commonly due to DynamicCache internals),
the script falls back to fixed-shape export with the shapes from the example inputs
and logs a warning in the manifest.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = Path(__file__).parent
MODEL_PATH = REPO_ROOT / "model"
VOICES_DIR = REPO_ROOT / "demo" / "voices" / "streaming_model"
VOICE_NAME = "en-Carter_man.pt"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPORT_DIR))

# ── Constants (from nenad102_onnx/config.json) ────────────────────────────────
HIDDEN_SIZE   = 896
LATENT_DIM    = 64
N_LM_LAYERS   = 4    # text_lm (lower Qwen2 layers)
N_TTS_LAYERS  = 20   # tts_lm  (upper Qwen2 layers)
N_KV_HEADS    = 2
HEAD_DIM      = 64


# ── Dynamic dimension helpers ─────────────────────────────────────────────────

def _dim(name: str):
    """Create a torch.export.Dim for a named dynamic axis."""
    return torch.export.Dim(name)


def _kv_example(n_layers: int, kv_seq: int = 64) -> tuple[torch.Tensor, ...]:
    """Build a flat (k0,v0,k1,v1,...) KV tuple as example inputs."""
    tensors = []
    for _ in range(n_layers):
        tensors.append(torch.zeros(1, N_KV_HEADS, kv_seq, HEAD_DIM))
        tensors.append(torch.zeros(1, N_KV_HEADS, kv_seq, HEAD_DIM))
    return tuple(tensors)


def _kv_dynamic_shapes(n_layers: int, kv_dim_name: str = "kv_seq"):
    """
    Build dynamic_shapes dict for the interleaved flat KV tuple.
    Each KV tensor has dynamic dim 2 (sequence length).
    Returns a list-of-dicts (one per positional *flat_kv arg).
    """
    kv_d = _dim(kv_dim_name)
    shapes = []
    for _ in range(n_layers):
        shapes.append({2: kv_d})   # key tensor
        shapes.append({2: kv_d})   # value tensor
    return shapes


# ── Per-component export spec ─────────────────────────────────────────────────

def _spec_text_lm(wrappers: dict) -> dict:
    """Return (wrapper, example_inputs, dynamic_shapes) for text_lm."""
    wrapper = wrappers["text_lm"]
    seq = 5  # example: first text window
    kv_seq = 64

    example_inputs = (
        torch.zeros(1, seq, dtype=torch.long),              # input_ids
        torch.arange(kv_seq, kv_seq + seq, dtype=torch.long),  # cache_position
        *_kv_example(N_LM_LAYERS, kv_seq),
    )

    seq_dim = _dim("seq")
    kv_dim  = _dim("kv_seq_lm")

    dynamic_shapes = (
        {1: seq_dim},                             # input_ids
        {0: seq_dim},                             # cache_position
        *_kv_dynamic_shapes(N_LM_LAYERS, "kv_seq_lm"),
    )
    return {"wrapper": wrapper, "example_inputs": example_inputs,
            "dynamic_shapes": dynamic_shapes}


def _spec_tts_lm(wrappers: dict) -> dict:
    wrapper = wrappers["tts_lm"]
    seq = 1
    kv_seq = 128

    example_inputs = (
        torch.zeros(1, seq, HIDDEN_SIZE),                   # lm_hidden_state
        torch.zeros(1, seq, dtype=torch.long),              # tts_text_mask
        torch.arange(kv_seq, kv_seq + seq, dtype=torch.long),  # cache_position
        *_kv_example(N_TTS_LAYERS, kv_seq),
    )

    seq_dim = _dim("seq")
    kv_dim  = _dim("kv_seq_tts")

    dynamic_shapes = (
        {1: seq_dim},                             # lm_hidden_state
        {1: seq_dim},                             # tts_text_mask
        {0: seq_dim},                             # cache_position
        *_kv_dynamic_shapes(N_TTS_LAYERS, "kv_seq_tts"),
    )
    return {"wrapper": wrapper, "example_inputs": example_inputs,
            "dynamic_shapes": dynamic_shapes}


def _spec_acoustic_connector(wrappers: dict) -> dict:
    wrapper = wrappers["acoustic_connector"]
    example_inputs = (torch.zeros(1, 1, LATENT_DIM),)
    frames_dim = _dim("frames")
    dynamic_shapes = ({1: frames_dim},)
    return {"wrapper": wrapper, "example_inputs": example_inputs,
            "dynamic_shapes": dynamic_shapes}


def _spec_diffusion_head(wrappers: dict) -> dict:
    wrapper = wrappers["diffusion_head"]
    example_inputs = (
        torch.zeros(2, LATENT_DIM),              # noisy_latents batch=2 (CFG)
        torch.tensor([999.0, 999.0]),             # timesteps
        torch.zeros(2, HIDDEN_SIZE),              # condition
    )
    # All shapes are static (batch=2 fixed, latent_dim and hidden fixed)
    dynamic_shapes = None
    return {"wrapper": wrapper, "example_inputs": example_inputs,
            "dynamic_shapes": dynamic_shapes}


def _spec_vocoder(wrappers: dict) -> dict:
    wrapper = wrappers["vocoder"]
    example_inputs = (torch.zeros(1, 16, LATENT_DIM),)
    frames_dim = _dim("vocoder_frames")
    dynamic_shapes = ({1: frames_dim},)
    return {"wrapper": wrapper, "example_inputs": example_inputs,
            "dynamic_shapes": dynamic_shapes}


COMPONENT_SPECS = {
    "text_lm":            _spec_text_lm,
    "tts_lm":             _spec_tts_lm,
    "acoustic_connector": _spec_acoustic_connector,
    "diffusion_head":     _spec_diffusion_head,
    "vocoder":            _spec_vocoder,
}

BACKEND_MAP = {
    "cpu":    "llvm-cpu",
    "vulkan": "vulkan-spirv",
}


# ── Core export function ──────────────────────────────────────────────────────

def export_component(
    name: str,
    spec: dict,
    backends: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    """
    Export one component → one .vmfb per backend.

    Returns a manifest entry dict.
    """
    import iree.turbine.aot as aot

    wrapper        = spec["wrapper"]
    example_inputs = spec["example_inputs"]
    dynamic_shapes = spec.get("dynamic_shapes")

    wrapper.eval()
    with torch.no_grad():
        pass  # ensure eval mode

    manifest_entry: dict[str, Any] = {
        "component":     name,
        "dynamic_shapes": dynamic_shapes is not None,
        "backends":      {},
        "warnings":      [],
    }

    # 1. torch.export  ─────────────────────────────────────────────────────────
    exported_prog = None
    dynamic_export_succeeded = False

    if dynamic_shapes is not None:
        try:
            print(f"  [torch.export] {name} (dynamic shapes) …")
            exported_prog = torch.export.export(
                wrapper,
                example_inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
            dynamic_export_succeeded = True
            print(f"  [torch.export] {name} ✓ (dynamic)")
        except Exception as exc:
            warn = (f"torch.export with dynamic_shapes failed for {name}: {exc!r}. "
                    "Falling back to static shapes from example_inputs.")
            print(f"  WARNING: {warn}", file=sys.stderr)
            manifest_entry["warnings"].append(warn)

    if exported_prog is None:
        # Static fallback
        try:
            print(f"  [torch.export] {name} (static shapes, fallback) …")
            exported_prog = torch.export.export(
                wrapper,
                example_inputs,
                strict=False,
            )
            print(f"  [torch.export] {name} ✓ (static)")
        except Exception as exc:
            err = f"torch.export failed entirely for {name}: {exc}"
            print(f"  ERROR: {err}", file=sys.stderr)
            traceback.print_exc()
            manifest_entry["error"] = err
            return manifest_entry

    # 2. iree-turbine AOT  ─────────────────────────────────────────────────────
    try:
        print(f"  [iree.turbine.aot.export] {name} …")
        iree_output = aot.export(exported_prog)
        print(f"  [iree.turbine.aot.export] {name} ✓")
    except Exception as exc:
        err = f"iree.turbine.aot.export failed for {name}: {exc}"
        print(f"  ERROR: {err}", file=sys.stderr)
        traceback.print_exc()
        manifest_entry["error"] = err
        return manifest_entry

    # 3. Compile to each backend  ──────────────────────────────────────────────
    for backend_short, iree_backend in [(b, BACKEND_MAP[b]) for b in backends]:
        vmfb_path = out_dir / f"{name}_{backend_short}.vmfb"
        print(f"  [compile] {name} → {iree_backend} ({vmfb_path.name}) …")
        try:
            iree_output.compile(
                save_to=str(vmfb_path),
                target_backends=[iree_backend],
            )
            size_mb = vmfb_path.stat().st_size / (1024 * 1024)
            print(f"  [compile] {name} ✓  ({size_mb:.1f} MB)")
            manifest_entry["backends"][backend_short] = {
                "file": vmfb_path.name,
                "size_mb": round(size_mb, 2),
                "iree_target": iree_backend,
            }
        except Exception as exc:
            warn = f"Compile to {iree_backend} failed for {name}: {exc!r}"
            print(f"  WARNING: {warn}", file=sys.stderr)
            manifest_entry["warnings"].append(warn)

    return manifest_entry


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export VibeVoice to IREE .vmfb")
    parser.add_argument(
        "--backends", nargs="+", choices=["cpu", "vulkan"], default=["cpu", "vulkan"],
        help="Compile targets (default: both)",
    )
    parser.add_argument(
        "--component", choices=list(COMPONENT_SPECS), default=None,
        help="Export a single component only (default: all)",
    )
    parser.add_argument(
        "--model_path", default=str(MODEL_PATH),
        help="Path to local model directory",
    )
    parser.add_argument(
        "--voice", default=str(VOICES_DIR / VOICE_NAME),
        help="Voice preset .pt file (needed to verify model loads correctly)",
    )
    args = parser.parse_args()

    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from wrappers import build_wrappers

    print(f"Loading model from {args.model_path}")
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        device_map="cpu",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    # Seed for determinism
    torch.manual_seed(0)

    print("Building wrappers …")
    wrappers = build_wrappers(model)

    components_to_export = (
        [args.component] if args.component else list(COMPONENT_SPECS)
    )

    manifest: dict[str, Any] = {
        "method":    "iree",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "backends":  args.backends,
        "components": {},
    }

    for comp_name in components_to_export:
        print(f"\n{'─'*60}")
        print(f"Exporting: {comp_name}")
        spec = COMPONENT_SPECS[comp_name](wrappers)
        entry = export_component(comp_name, spec, args.backends, EXPORT_DIR)
        manifest["components"][comp_name] = entry

    # ── Write manifest ─────────────────────────────────────────────────────────
    manifest_path = EXPORT_DIR / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nManifest written to {manifest_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Export Summary ─────────────────────────────────────────────")
    all_ok = True
    for name, entry in manifest["components"].items():
        if "error" in entry:
            print(f"  {name:<24} ERROR: {entry['error']}")
            all_ok = False
        else:
            backends_ok = ", ".join(
                f"{b}:{info['size_mb']}MB"
                for b, info in entry.get("backends", {}).items()
            ) or "no backends compiled"
            warn_flag = " (warnings)" if entry.get("warnings") else ""
            print(f"  {name:<24} {backends_ok}{warn_flag}")
    print("───────────────────────────────────────────────────────────────")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
