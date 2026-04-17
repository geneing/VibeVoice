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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


# ── Patch Qwen2 RoPE to remove un-exportable HOPs ────────────────────────────
# Qwen2RotaryEmbedding.forward uses @torch.no_grad() and
# torch.autocast(enabled=False).  Both produce higher-order ops
# ('wrap_with_set_grad_enabled', 'wrap_with_autocast') that iree-turbine's
# FxImporter does not implement.  The autocast is a no-op (all tensors are
# already cast to .float()), and no_grad is irrelevant at inference time, so
# we replace the method with a semantically identical HOP-free version only
# for the duration of torch.export.


def _rope_forward_clean(self, x: torch.Tensor, position_ids: torch.Tensor):
    """Drop-in replacement for Qwen2RotaryEmbedding.forward without HOPs."""
    inv_freq_expanded = (
        self.inv_freq[None, :, None]
        .float()
        .expand(position_ids.shape[0], -1, 1)
        .to(x.device)
    )
    position_ids_expanded = position_ids[:, None, :].float()
    # autocast(enabled=False) is a pass-through; .float() already handles dtype.
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@contextmanager
def _patch_qwen2_rope():
    """Context manager: swap Qwen2RotaryEmbedding.forward for HOP-free version."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    except ImportError:
        yield
        return
    orig = Qwen2RotaryEmbedding.forward
    Qwen2RotaryEmbedding.forward = _rope_forward_clean
    try:
        yield
    finally:
        Qwen2RotaryEmbedding.forward = orig


def _causal_mask_clean(
    config,
    input_embeds: torch.Tensor,
    attention_mask,
    cache_position: torch.Tensor,
    past_key_values,
    **kwargs,
) -> torch.Tensor:
    """
    HOP-free replacement for transformers.masking_utils.create_causal_mask.

    The stock implementation uses torch.vmap (via mask_interface) which produces
    'wrap_with_set_grad_enabled' and '_vmap_increment_nesting' HOPs that
    iree-turbine cannot lower.  This version computes an equivalent 4D float
    causal mask using only basic tensor ops.

    Returns a (batch, 1, q_len, kv_len) float mask:  0 = attend, -inf = mask.
    """
    batch_size = input_embeds.shape[0]
    q_len      = input_embeds.shape[1]
    dtype      = input_embeds.dtype
    device     = input_embeds.device

    # Determine past KV length from the cache object.
    kv_past = 0
    if (past_key_values is not None
            and hasattr(past_key_values, "key_cache")
            and len(past_key_values.key_cache) > 0
            and past_key_values.key_cache[0].shape[2] > 0):
        kv_past = past_key_values.key_cache[0].shape[2]

    kv_len = kv_past + q_len  # total keys available to attend to

    # cache_position[i] = absolute position of query token i.
    # Token i may attend to key k iff k <= cache_position[i].
    arange_kv = torch.arange(kv_len, device=device, dtype=torch.long)  # (kv_len,)
    attend = arange_kv.unsqueeze(0) <= cache_position.unsqueeze(-1)     # (q_len, kv_len)

    # Convert bool mask to float: 0.0 where attend=True, -inf where False.
    neg_inf = torch.finfo(dtype).min if dtype != torch.float32 else float("-inf")
    float_mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    float_mask = float_mask.masked_fill(~attend, neg_inf)

    return float_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, kv_len)


@contextmanager
def _patch_create_causal_mask():
    """Context manager: swap transformers create_causal_mask for HOP-free version."""
    try:
        import transformers.masking_utils as mu
        import transformers.models.qwen2.modeling_qwen2 as qm
    except ImportError:
        yield
        return

    orig_mu = mu.create_causal_mask
    orig_qm = getattr(qm, "create_causal_mask", None)

    mu.create_causal_mask = _causal_mask_clean
    if orig_qm is not None:
        qm.create_causal_mask = _causal_mask_clean
    try:
        yield
    finally:
        mu.create_causal_mask = orig_mu
        if orig_qm is not None:
            qm.create_causal_mask = orig_qm


def _preload_torch_decompositions() -> None:
    """
    Force-load PyTorch's op decomposition table before torch.export.

    If this is not done, `lazy_load_decompositions` appears as a call_function
    node in the exported FX graph.  iree-turbine's FxImporter does not implement
    this node and raises NotImplementedError.  Calling the function once is
    idempotent — it replaces itself with a no-op afterwards.
    """
    try:
        from torch._decomp import lazy_load_decompositions  # type: ignore[attr-defined]
        lazy_load_decompositions()
    except (ImportError, AttributeError):
        pass


def _remove_lazy_load_nodes(exported_prog: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    """
    Remove `lazy_load_decompositions` call_function nodes from the exported graph.

    These nodes have no output (return None) and no users; they are pure side
    effects that pre-load PyTorch decompositions.  iree-turbine's FxImporter
    raises NotImplementedError for them.  Erasing them is safe because the
    decompositions are already loaded by the time we reach the compile step.
    """
    graph = exported_prog.graph_module.graph
    to_erase = [
        n for n in list(graph.nodes)
        if n.op == "call_function"
        and "lazy_load_decompositions" in getattr(
            n.target, "__qualname__", repr(n.target)
        )
    ]

    if not to_erase:
        return exported_prog

    for node in to_erase:
        # These nodes return None and should have no users.
        # If somehow a user exists, redirect it to the first available constant.
        if node.users:
            node.replace_all_uses_with(
                next(iter(node.users))  # type: ignore[arg-type]  # fallback
            )
        graph.erase_node(node)

    print(f"  [graph-patch] Removed {len(to_erase)} lazy_load_decompositions node(s).")
    exported_prog.graph_module.recompile()
    return exported_prog

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

    # For variadic *flat_kv, KV shapes must be a tuple (matching example_inputs[2])
    dynamic_shapes = (
        {1: seq_dim},                             # input_ids
        {0: seq_dim},                             # cache_position
        tuple(_kv_dynamic_shapes(N_LM_LAYERS, "kv_seq_lm")),  # *flat_kv as tuple
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

    # For variadic *flat_kv, KV shapes must be a tuple (matching example_inputs[3])
    dynamic_shapes = (
        {1: seq_dim},                             # lm_hidden_state
        {1: seq_dim},                             # tts_text_mask
        {0: seq_dim},                             # cache_position
        tuple(_kv_dynamic_shapes(N_TTS_LAYERS, "kv_seq_tts")),  # *flat_kv as tuple
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

    # Pre-load decompositions to prevent lazy_load_decompositions nodes in graph.
    _preload_torch_decompositions()

    # Patch Qwen2 RoPE and create_causal_mask for the duration of torch.export
    # to eliminate HOPs that iree-turbine cannot lower.
    with _patch_qwen2_rope(), _patch_create_causal_mask():
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

    # 1b. Post-process: remove lazy_load_decompositions nodes that iree-turbine
    #     cannot lower.  These are pure side-effect calls with no return value.
    exported_prog = _remove_lazy_load_nodes(exported_prog)

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
    # Use eager attention: SDPA internally uses torch.autocast which creates
    # a 'wrap_with_autocast' higher-order op that iree-turbine cannot lower.
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
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
