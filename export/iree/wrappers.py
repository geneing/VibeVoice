"""
export/iree/wrappers.py — thin nn.Module wrappers around each VibeVoice component.

Design goals:
  - Accept and return only plain torch.Tensor / tuple-of-Tensor (no dataclasses,
    no DynamicCache).  This makes each wrapper exportable via torch.export.
  - All Python control-flow that depends on tensor values (EOS, windowing, DPM
    scheduling) remains OUTSIDE these wrappers, in the orchestration layer.
  - KV cache is passed as a flat sequence of tensors: (k0, v0, k1, v1, ...) for
    N layers.  The wrapper reconstructs a DynamicCache internally and unpacks the
    updated cache on return.

Layer counts (VibeVoice-Realtime-0.5B, verified from nenad102_onnx/config.json):
  text_lm  : 4 Qwen2 layers  →  8  KV tensors
  tts_lm   : 20 Qwen2 layers → 40  KV tensors
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pack_dynamic_cache(flat_kv: Tuple[torch.Tensor, ...]) -> "DynamicCache":
    """
    Reconstruct a DynamicCache from an interleaved flat tuple (k0,v0,k1,v1,...).
    Uses the transformers DynamicCache API (works with >=4.51, >=4.57 via shims).
    """
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    n = len(flat_kv) // 2
    cache.key_cache   = list(flat_kv[0::2])   # [k0, k1, ...]
    cache.value_cache = list(flat_kv[1::2])   # [v0, v1, ...]
    # Compatibility shim: some transformers versions iterate over cache.layers
    try:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import _ensure_cache_has_layers
        _ensure_cache_has_layers(cache)
    except Exception:
        pass
    return cache


def _unpack_dynamic_cache(cache) -> Tuple[torch.Tensor, ...]:
    """
    Flatten a DynamicCache back to an interleaved tuple (k0,v0,k1,v1,...).
    Supports both the list-based DynamicCache and tuple-of-tuples format.
    """
    out: list[torch.Tensor] = []
    try:
        for k, v in zip(cache.key_cache, cache.value_cache):
            out.extend([k, v])
    except AttributeError:
        # Tuple of (key, value) pairs (old transformers API)
        for k, v in cache:
            out.extend([k, v])
    return tuple(out)


# ── Text-LM wrapper ───────────────────────────────────────────────────────────

class TextLMWrapper(nn.Module):
    """
    One incremental forward pass of the text LM (lower Qwen2 layers).

    Inputs:
        input_ids      : (1, seq)                      int64
        cache_position : (seq,)                         int64
        *flat_kv       : interleaved (k_i, v_i) tensors, each (1, H, kv_seq, D)
                         H=num_kv_heads=2, D=head_dim=64

    Outputs (tuple):
        last_hidden_state : (1, seq, hidden)
        *new_flat_kv      : updated interleaved KV tensors
    """

    def __init__(self, model: nn.Module, embed_tokens: nn.Module) -> None:
        super().__init__()
        self.language_model = model
        self.embed_tokens   = embed_tokens

    def forward(
        self,
        input_ids:      torch.Tensor,   # (1, seq)
        cache_position: torch.Tensor,   # (seq,)
        *flat_kv:       torch.Tensor,   # 2 * N_LM_LAYERS tensors
    ) -> Tuple[torch.Tensor, ...]:

        inputs_embeds = self.embed_tokens(input_ids)

        past_kv = _pack_dynamic_cache(flat_kv) if flat_kv else None

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        new_flat = _unpack_dynamic_cache(outputs.past_key_values)
        return (outputs.last_hidden_state, *new_flat)


# ── TTS-LM wrapper ────────────────────────────────────────────────────────────

class TTSLMWrapper(nn.Module):
    """
    One incremental forward pass of the TTS LM (upper Qwen2 layers).

    Used for BOTH positive and negative (CFG) paths — call twice per step.

    Inputs:
        lm_hidden_state  : (1, seq, hidden)    either text-LM hidden or acoustic embed
        tts_text_mask    : (1, seq)   int64    1=text token position, 0=speech
        cache_position   : (seq,)     int64
        *flat_kv         : interleaved (k_i, v_i) tensors, each (1, H, kv_seq, D)

    Outputs (tuple):
        last_hidden_state : (1, seq, hidden)
        eos_logit         : (1, 1)            raw EOS logit (sigmoid outside)
        *new_flat_kv      : updated interleaved KV tensors
    """

    def __init__(
        self,
        tts_language_model: nn.Module,
        tts_input_types:    nn.Module,  # nn.Embedding(2, hidden)
        eos_classifier:     nn.Module,
    ) -> None:
        super().__init__()
        self.tts_language_model = tts_language_model
        self.tts_input_types    = tts_input_types
        self.eos_classifier     = eos_classifier

    def forward(
        self,
        lm_hidden_state: torch.Tensor,   # (1, seq, hidden)
        tts_text_mask:   torch.Tensor,   # (1, seq) int64
        cache_position:  torch.Tensor,   # (seq,) int64
        *flat_kv:        torch.Tensor,   # 2 * N_TTS_LAYERS tensors
    ) -> Tuple[torch.Tensor, ...]:

        # Add text/speech type embedding
        type_embed    = self.tts_input_types(tts_text_mask)  # (1, seq, hidden)
        inputs_embeds = lm_hidden_state + type_embed

        past_kv = _pack_dynamic_cache(flat_kv) if flat_kv else None

        outputs = self.tts_language_model(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        hidden = outputs.last_hidden_state                  # (1, seq, hidden)
        eos    = self.eos_classifier(hidden[:, -1, :])      # (1, 1)

        new_flat = _unpack_dynamic_cache(outputs.past_key_values)
        return (hidden, eos, *new_flat)


# ── Acoustic-connector wrapper ────────────────────────────────────────────────

class AcousticConnectorWrapper(nn.Module):
    """
    Projects diffusion latent into TTS-LM embedding space.

    Inputs:
        latent : (1, num_frames, latent_dim)   float32   (latent_dim=64)

    Outputs:
        embed  : (1, num_frames, hidden)       float32
    """

    def __init__(self, connector: nn.Module) -> None:
        super().__init__()
        self.connector = connector

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.connector(latent)


# ── Diffusion-head wrapper ────────────────────────────────────────────────────

class DiffusionHeadWrapper(nn.Module):
    """
    One denoising step of the diffusion head.

    CFG is handled externally — this wrapper is always called with batch=2:
        batch dimension 0: positive condition
        batch dimension 1: negative condition

    Inputs:
        noisy_latents : (2, latent_dim)  float32   latent_dim=64
        timesteps     : (2,)             float32   diffusion timestep index
        condition     : (2, hidden)      float32

    Outputs:
        predicted     : (2, latent_dim)  float32
    """

    def __init__(self, prediction_head: nn.Module) -> None:
        super().__init__()
        self.head = prediction_head

    def forward(
        self,
        noisy_latents: torch.Tensor,   # (2, 64)
        timesteps:     torch.Tensor,   # (2,)
        condition:     torch.Tensor,   # (2, hidden)
    ) -> torch.Tensor:
        return self.head(noisy_latents, timesteps, condition)


# ── Vocoder wrapper ───────────────────────────────────────────────────────────

def _unwrap_no_grad(fn):
    """
    Recursively unwrap @torch.no_grad() (and similar) decorators.
    PyTorch's no_grad sets __wrapped__ on the decorated function.
    Returns the innermost original function.
    """
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


class VocoderWrapper(nn.Module):
    """
    Decode a sequence of speech latents to a waveform.

    Scaling / de-biasing is applied internally so that the exported interface
    matches the nenad102_onnx vocoder: input is raw latents from the diffusion
    head (before any scaling).

    Inputs:
        latents : (1, num_frames, latent_dim)  float32   latent_dim=64

    Outputs:
        waveform : (1, num_samples)            float32   24 kHz mono
    """

    def __init__(
        self,
        acoustic_tokenizer: nn.Module,
        scaling_factor:     torch.Tensor,
        bias_factor:        torch.Tensor,
    ) -> None:
        super().__init__()
        self.acoustic_tokenizer = acoustic_tokenizer
        self.register_buffer("scaling_factor", scaling_factor.clone().detach())
        self.register_buffer("bias_factor",    bias_factor.clone().detach())
        # Unwrap @torch.no_grad() on the decode method to prevent
        # 'wrap_with_set_grad_enabled' HOPs that iree-turbine cannot lower.
        cls_decode = type(acoustic_tokenizer).decode
        self._raw_decode = _unwrap_no_grad(cls_decode)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        scaled = latents / self.scaling_factor - self.bias_factor
        # Call the unwrapped decode directly (bypasses @torch.no_grad decorator).
        try:
            return self._raw_decode(self.acoustic_tokenizer, scaled, use_cache=False)
        except TypeError:
            return self._raw_decode(self.acoustic_tokenizer, scaled)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_wrappers(model: nn.Module) -> dict[str, nn.Module]:
    """
    Build all five exportable wrappers from a loaded
    VibeVoiceStreamingForConditionalGenerationInference instance.

    Returns a dict keyed by component name:
        "text_lm"           → TextLMWrapper
        "tts_lm"            → TTSLMWrapper
        "acoustic_connector"→ AcousticConnectorWrapper
        "diffusion_head"    → DiffusionHeadWrapper
        "vocoder"           → VocoderWrapper
    """
    m = model.model  # VibeVoiceStreamingModel

    return {
        "text_lm": TextLMWrapper(
            model=m.language_model,
            embed_tokens=m.get_input_embeddings(),
        ),
        "tts_lm": TTSLMWrapper(
            tts_language_model=m.tts_language_model,
            tts_input_types=m.tts_input_types,
            eos_classifier=model.tts_eos_classifier,
        ),
        "acoustic_connector": AcousticConnectorWrapper(m.acoustic_connector),
        "diffusion_head":     DiffusionHeadWrapper(m.prediction_head),
        "vocoder": VocoderWrapper(
            acoustic_tokenizer=m.acoustic_tokenizer,
            scaling_factor=m.speech_scaling_factor,
            bias_factor=m.speech_bias_factor,
        ),
    }
