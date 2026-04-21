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


@contextmanager
def _patch_math_ceil():
    """
    Replace math.ceil with int() during torch.export.

    Transposed-conv output-size helpers call math.ceil(float_expr) where
    float_expr is always an integer-valued float (stride-1 divisions of integer
    symbolic dims).  math.ceil produces call_function nodes that iree-turbine's
    FxImporter cannot lower.  Substituting int() — which is semantically
    identical for integer-valued floats — either inlines the conversion or
    produces a more-lowerable node type.
    """
    import math
    orig = math.ceil
    math.ceil = int  # int(x) == ceil(x) for all integer-valued x
    try:
        yield
    finally:
        math.ceil = orig


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


def _replace_ceil_with_input(exported_prog: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    """
    Replace ``math.ceil(x)`` call_function nodes with integer-typed equivalents.

    Context
    -------
    The vocoder's conv_transpose1d output-size helpers emit ``math.ceil(N)``
    nodes when the input sequence length is symbolic.  All such expressions
    have the form::

        ceil(add(truediv(symint_expr, int_stride), float_const))

    or the degenerate form::

        ceil(truediv(symint_expr, int_stride))   (float_const == 0)

    Because the vocoder upsamples by exact integer multiples (HiFiGAN-style,
    strides divide the sequence length evenly), the expression under ceil is
    always integer-valued.  We therefore replace it with the equivalent integer
    arithmetic::

        floordiv(symint_expr, int_stride) + round(float_const)

    This removes the ``truediv``/``add.float`` chain that iree-turbine's
    FxImporter cannot lower, while producing the identical integer result.

    The replacement is also valid for chained ceilings (ceil node feeds into
    the numerator of a later truediv) because we iterate in topological order
    and each replaced node is already an integer-typed FX node before the next
    iteration processes its dependants.
    """
    import math
    import operator

    graph = exported_prog.graph_module.graph
    replaced = 0

    for node in list(graph.nodes):
        if not (node.op == "call_function" and node.target is math.ceil):
            continue

        float_arg = node.args[0]
        if not isinstance(float_arg, torch.fx.Node):
            continue

        # ── Case 1: ceil(add(truediv(X, stride), float_const)) ──────────────
        if float_arg.op == "call_function" and float_arg.target is operator.add:
            truediv_node = None
            float_const: float | None = None
            for a in float_arg.args:
                if (isinstance(a, torch.fx.Node)
                        and a.op == "call_function"
                        and a.target is operator.truediv):
                    truediv_node = a
                elif isinstance(a, (int, float)):
                    float_const = float(a)
            if truediv_node is None:
                continue
            if float_const is None:
                float_const = 0.0
            if len(truediv_node.args) < 2 or not isinstance(truediv_node.args[1], int):
                continue
            stride: int = truediv_node.args[1]
            numerator = truediv_node.args[0]

        # ── Case 2: ceil(truediv(X, stride)) ────────────────────────────────
        elif float_arg.op == "call_function" and float_arg.target is operator.truediv:
            if len(float_arg.args) < 2 or not isinstance(float_arg.args[1], int):
                continue
            truediv_node = float_arg
            stride = float_arg.args[1]
            numerator = float_arg.args[0]
            float_const = 0.0
        else:
            continue

        int_const = round(float_const)  # 1.0 → 1, 0.0 → 0, -1.0 → -1

        # Capture the ceil node's symbolic value BEFORE we erase it.
        # We'll propagate it (and derived values) to the new nodes so that
        # iree-turbine's FxImporter sees is_symbolic(node.meta["val"]) == True
        # and routes the new nodes through _import_symbolic_torch_op.
        ceil_sym_val = node.meta.get("val")   # torch.SymInt or None

        # Build: floordiv(numerator, stride) + int_const
        with graph.inserting_before(node):
            if stride == 1:
                fd = numerator
            else:
                fd = graph.call_function(operator.floordiv, args=(numerator, stride))
                # Propagate SymInt val for the floordiv node
                if ceil_sym_val is not None:
                    try:
                        fd.meta["val"] = ceil_sym_val - int_const  # type: ignore[operator]
                    except Exception:
                        pass

            if int_const != 0:
                int_result: torch.fx.Node = graph.call_function(
                    operator.add, args=(fd, int_const))
                if ceil_sym_val is not None:
                    int_result.meta["val"] = ceil_sym_val
            else:
                int_result = fd  # type: ignore[assignment]
                if ceil_sym_val is not None and int_result is not numerator:
                    int_result.meta["val"] = ceil_sym_val

        node.replace_all_uses_with(int_result)
        graph.erase_node(node)
        replaced += 1

    if replaced:
        # Clean up now-dead float intermediary nodes (truediv / add.float / sub)
        changed = True
        while changed:
            changed = False
            for n in list(graph.nodes):
                if (n.op == "call_function"
                        and not n.users
                        and n.target in (operator.truediv, operator.add, operator.sub)):
                    try:
                        graph.erase_node(n)
                        changed = True
                    except Exception:
                        pass
        print(f"  [graph-patch] Replaced {replaced} math.ceil node(s) with integer floordiv.")
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

def _dim(name: str, min: int = 1, max: int | None = None):
    """Create a torch.export.Dim for a named dynamic axis."""
    return torch.export.Dim(name, min=min, max=max)


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
    kv_d = _dim(kv_dim_name, min=1)
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

    seq_dim = _dim("seq", min=1)
    kv_dim  = _dim("kv_seq_lm", min=1)

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
    # Use example seq=5 (TEXT_WINDOW size) to capture both the text-window and
    # speech-token (seq=1) call sites.  Both seq and kv_seq are marked DYNAMIC
    # so torch.export cannot specialise either, avoiding constraint violations
    # from Qwen2's GQA repeat_kv symbolic guard.
    seq = 5
    kv_seq = 128

    example_inputs = (
        torch.zeros(1, seq, HIDDEN_SIZE),                        # lm_hidden_state
        torch.zeros(1, seq, dtype=torch.long),                   # tts_text_mask
        torch.arange(kv_seq, kv_seq + seq, dtype=torch.long),   # cache_position
        *_kv_example(N_TTS_LAYERS, kv_seq),
    )

    # Both seq and kv_seq are dynamic; Dim.DYNAMIC bypasses torch.export
    # constraint checking so symbolic guards don't cause export failures.
    seq_dyn = torch.export.Dim.DYNAMIC
    kv_dyn  = torch.export.Dim.DYNAMIC
    kv_shapes: list[dict] = []
    for _ in range(N_TTS_LAYERS):
        kv_shapes.append({2: kv_dyn})  # key tensor
        kv_shapes.append({2: kv_dyn})  # value tensor

    dynamic_shapes = (
        {1: seq_dyn},                                         # lm_hidden_state
        {1: seq_dyn},                                         # tts_text_mask
        {0: seq_dyn},                                         # cache_position
        tuple(kv_shapes),                                     # *flat_kv as tuple
    )
    return {"wrapper": wrapper, "example_inputs": example_inputs,
            "dynamic_shapes": dynamic_shapes}


def _spec_acoustic_connector(wrappers: dict) -> dict:
    wrapper = wrappers["acoustic_connector"]
    # frames=1: AcousticConnectorWrapper is called one frame at a time in the decode
    # loop.  torch.export specialises it to 1 anyway, so export it statically.
    example_inputs = (torch.zeros(1, 1, LATENT_DIM),)
    dynamic_shapes = None
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
    # Transposed-conv output-size helpers emit math.ceil(N) nodes when the input
    # length is symbolic.  Dim.DYNAMIC bypasses torch.export constraint checking;
    # _replace_ceil_with_input() then removes the ceil nodes before iree-turbine
    # sees them (ceil(N)==N for all integer N).
    example_inputs = (torch.zeros(1, 32, LATENT_DIM),)
    frames_dim = torch.export.Dim.DYNAMIC
    dynamic_shapes = ({1: frames_dim},)
    return {"wrapper": wrapper, "example_inputs": example_inputs,
            "dynamic_shapes": dynamic_shapes}


def _spec_tts_lm_paired(wrappers: dict) -> dict:
    """
    Export spec for TTSLMPairedWrapper — runs pos+neg CFG paths in one call.

    Both pos and neg KV dims are independently DYNAMIC (they differ: pos starts
    at the voice-prompt length ~316, neg starts at 1).  The pos path is exported
    with kv_seq=128 and the neg path with kv_seq=4 to give torch.export concrete
    but distinct example shapes.  seq=1 for both (speech-token decode step).
    """
    wrapper = wrappers["tts_lm_paired"]
    seq = 1          # speech-token step: seq=1
    pos_kv = 128     # example positive KV length
    neg_kv = 4       # example negative KV length (starts small)

    pos_kv_example = _kv_example(N_TTS_LAYERS, pos_kv)
    neg_kv_example = _kv_example(N_TTS_LAYERS, neg_kv)

    example_inputs = (
        torch.zeros(1, seq, HIDDEN_SIZE),                           # pos_hidden
        torch.zeros(1, seq, dtype=torch.long),                      # pos_mask
        torch.arange(pos_kv, pos_kv + seq, dtype=torch.long),      # pos_cache_pos
        torch.zeros(1, seq, HIDDEN_SIZE),                           # neg_hidden
        torch.zeros(1, seq, dtype=torch.long),                      # neg_mask
        torch.arange(neg_kv, neg_kv + seq, dtype=torch.long),      # neg_cache_pos
        *pos_kv_example,
        *neg_kv_example,
    )

    dyn = torch.export.Dim.DYNAMIC  # both seq and KV lengths fully dynamic
    pos_kv_shapes: list[dict] = []
    neg_kv_shapes: list[dict] = []
    for _ in range(N_TTS_LAYERS):
        pos_kv_shapes.append({2: dyn})
        pos_kv_shapes.append({2: dyn})
        neg_kv_shapes.append({2: dyn})
        neg_kv_shapes.append({2: dyn})

    dynamic_shapes = (
        None,                        # pos_hidden  (1, 1, hidden) — seq=1 static
        None,                        # pos_mask    (1, 1) — static
        None,                        # pos_cache_pos (1,) — static
        None,                        # neg_hidden  — static
        None,                        # neg_mask    — static
        None,                        # neg_cache_pos — static
        # *pos_neg_kv is a single variadic arg — only KV dim 2 is dynamic
        tuple(pos_kv_shapes + neg_kv_shapes),
    )
    return {"wrapper": wrapper, "example_inputs": example_inputs,
            "dynamic_shapes": dynamic_shapes}


COMPONENT_SPECS = {
    "text_lm":            _spec_text_lm,
    "tts_lm":             _spec_tts_lm,
    "tts_lm_paired":      _spec_tts_lm_paired,
    "acoustic_connector": _spec_acoustic_connector,
    "diffusion_head":     _spec_diffusion_head,
    "vocoder":            _spec_vocoder,
}

BACKEND_MAP = {
    "cpu":    "llvm-cpu",
    "vulkan": "vulkan-spirv",
}


# ── Core export function ──────────────────────────────────────────────────────

def _torch_export_to_mlir_bytes(
    name: str,
    wrapper: torch.nn.Module,
    example_inputs: tuple,
    dynamic_shapes,
    warnings_out: list[str],
    label: str = "",
) -> bytes | None:
    """
    Run torch.export → graph patches → iree.turbine.aot.export → MLIR bytes.

    Returns serialised MLIR bytecode, or None if the export failed (error is
    appended to warnings_out and printed to stderr).
    """
    import io as _io
    import iree.turbine.aot as aot

    tag = f" ({label})" if label else ""
    exported_prog = None

    _preload_torch_decompositions()

    with _patch_qwen2_rope(), _patch_create_causal_mask():
        if dynamic_shapes is not None:
            try:
                print(f"  [torch.export] {name}{tag} (dynamic shapes) …")
                exported_prog = torch.export.export(
                    wrapper,
                    example_inputs,
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                )
                print(f"  [torch.export] {name}{tag} ✓ (dynamic)")
            except Exception as exc:
                warn = (f"torch.export with dynamic_shapes failed for {name}{tag}: {exc!r}. "
                        "Falling back to static shapes.")
                print(f"  WARNING: {warn}", file=sys.stderr)
                warnings_out.append(warn)

        if exported_prog is None:
            try:
                print(f"  [torch.export] {name}{tag} (static shapes) …")
                exported_prog = torch.export.export(
                    wrapper,
                    example_inputs,
                    strict=False,
                )
                print(f"  [torch.export] {name}{tag} ✓ (static)")
            except Exception as exc:
                err = f"torch.export failed entirely for {name}{tag}: {exc}"
                print(f"  ERROR: {err}", file=sys.stderr)
                traceback.print_exc()
                warnings_out.append(err)
                return None

    exported_prog = _remove_lazy_load_nodes(exported_prog)
    exported_prog = _replace_ceil_with_input(exported_prog)

    try:
        print(f"  [iree.turbine.aot.export] {name}{tag} …")
        iree_output = aot.export(exported_prog)
        print(f"  [iree.turbine.aot.export] {name}{tag} ✓")
    except Exception as exc:
        err = f"iree.turbine.aot.export failed for {name}{tag}: {exc}"
        print(f"  ERROR: {err}", file=sys.stderr)
        traceback.print_exc()
        warnings_out.append(err)
        return None

    buf = _io.BytesIO()
    iree_output.mlir_module.write_bytecode(buf)
    return buf.getvalue()


def export_component(
    name: str,
    spec: dict,
    backends: list[str],
    out_dir: Path,
    fp16: bool = True,
    vulkan_target: str = "adreno",
) -> dict[str, Any]:
    """
    Export one component → one .vmfb per backend.

    When fp16=True the Vulkan backend is compiled from a float16 model export
    (wrapper cast to .half() before torch.export) so that all constants and ops
    are natively f16 in the MLIR — avoiding the broken --iree-input-demote-f32-to-f16
    pass that produces arith.constant type-verification failures.  The CPU
    backend always uses a float32 export.

    Returns a manifest entry dict.
    """
    import iree.compiler as iree_compiler

    wrapper        = spec["wrapper"]
    example_inputs = spec["example_inputs"]
    dynamic_shapes = spec.get("dynamic_shapes")

    wrapper.eval()

    manifest_entry: dict[str, Any] = {
        "component":      name,
        "dynamic_shapes": dynamic_shapes is not None,
        "backends":       {},
        "warnings":       [],
    }

    # Partition backends: cpu (and vulkan without fp16) use f32; vulkan+fp16 uses f16.
    f32_backends = [b for b in backends if not (b == "vulkan" and fp16)]
    f16_backends = [b for b in backends if b == "vulkan" and fp16]

    # ── Build f32 MLIR (for CPU / non-fp16 Vulkan) ────────────────────────────
    mlir_bytes_f32: bytes | None = None
    if f32_backends:
        mlir_bytes_f32 = _torch_export_to_mlir_bytes(
            name, wrapper, example_inputs, dynamic_shapes,
            manifest_entry["warnings"], label="f32",
        )
        if mlir_bytes_f32 is None:
            manifest_entry["error"] = f"f32 export failed for {name}"
            if not f16_backends:
                return manifest_entry

    # ── Build f16 MLIR (for Vulkan fp16) ──────────────────────────────────────
    mlir_bytes_f16: bytes | None = None
    if f16_backends:
        # Deep-copy the wrapper so we don't mutate the shared model weights.
        wrapper_f16 = copy.deepcopy(wrapper).half()
        wrapper_f16.eval()
        # Cast all floating-point example inputs to f16; leave integer tensors intact.
        example_inputs_f16 = tuple(
            t.half() if isinstance(t, torch.Tensor) and t.is_floating_point() else t
            for t in example_inputs
        )
        mlir_bytes_f16 = _torch_export_to_mlir_bytes(
            name, wrapper_f16, example_inputs_f16, dynamic_shapes,
            manifest_entry["warnings"], label="f16",
        )
        if mlir_bytes_f16 is None:
            manifest_entry["warnings"].append(f"f16 export failed for {name}; skipping Vulkan fp16.")

    # ── Compile each backend ───────────────────────────────────────────────────
    for backend_short, iree_backend in [(b, BACKEND_MAP[b]) for b in backends]:
        is_fp16_vulkan = backend_short == "vulkan" and fp16
        mlir_bytes = mlir_bytes_f16 if is_fp16_vulkan else mlir_bytes_f32
        if mlir_bytes is None:
            continue  # export already failed; warning recorded above

        vmfb_suffix = f"{backend_short}_fp16" if is_fp16_vulkan else backend_short
        vmfb_path   = out_dir / f"{name}_{vmfb_suffix}.vmfb"
        print(f"  [compile] {name} → {iree_backend} ({vmfb_path.name}) …")

        extra_args: list[str] = []
        if backend_short == "cpu":
            # Target the host CPU to enable all available CPU features (AVX, etc.)
            # and silence the generic-CPU performance warning.
            extra_args.append("--iree-llvmcpu-target-cpu=host")
        if backend_short == "vulkan" and vulkan_target and vulkan_target.lower() != "none":
            # Override the default vp_android_baseline_2022 profile which only allows
            # 16 KB shared memory and no fp16 storage — far too conservative for real
            # GPUs. Specifying an explicit target (e.g. "adreno", "rdna3", "valhall4")
            # selects a profile with larger shared memory limits and fp16 support.
            # Pass "none" (or omit) to use the conservative default profile, which
            # is more compatible with non-Adreno Vulkan implementations (e.g. dzn).
            extra_args.append(f"--iree-vulkan-target={vulkan_target}")

        try:
            vmfb_bytes = iree_compiler.compile_str(
                mlir_bytes,
                target_backends=[iree_backend],
                input_type="torch",
                extra_args=extra_args,
            )
            vmfb_path.write_bytes(vmfb_bytes)
            size_mb = vmfb_path.stat().st_size / (1024 * 1024)
            print(f"  [compile] {name} ✓  ({size_mb:.1f} MB)")
            effective_vulkan_target = (
                vulkan_target
                if backend_short == "vulkan" and vulkan_target and vulkan_target.lower() != "none"
                else None
            )
            manifest_entry["backends"][backend_short] = {
                "file":        vmfb_path.name,
                "size_mb":     round(size_mb, 2),
                "iree_target": iree_backend,
                "fp16":        is_fp16_vulkan,
                "vulkan_target": effective_vulkan_target,
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
        help="Compile targets (default: cpu vulkan)",
    )
    parser.add_argument(
        "--component", choices=list(COMPONENT_SPECS), default=None,
        help="Export a single component only (default: all)",
    )
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=True,
        help="Compile Vulkan backend as FP16 (model cast to half() before export; default: True)",
    )
    parser.add_argument(
        "--vulkan-target", default="valhall4", dest="vulkan_target",
        help=(
            "IREE Vulkan GPU target (--iree-vulkan-target). "
            "Use 'adreno' for Qualcomm, 'valhall4' for Mali, 'rdna3' for AMD. "
            "Use 'none' to omit the flag and use IREE's conservative default "
            "vp_android_baseline_2022 profile — compatible with dzn and other "
            "non-Adreno Vulkan implementations. (default: adreno)"
        ),
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
        "method":        "iree",
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "model_path":    args.model_path,
        "backends":      args.backends,
        "fp16_vulkan":   args.fp16,
        "vulkan_target": args.vulkan_target,
        "components":    {},
    }

    for comp_name in components_to_export:
        print(f"\n{'─'*60}")
        print(f"Exporting: {comp_name}")
        spec = COMPONENT_SPECS[comp_name](wrappers)
        entry = export_component(
            comp_name, spec, args.backends, EXPORT_DIR,
            fp16=args.fp16, vulkan_target=args.vulkan_target,
        )
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
