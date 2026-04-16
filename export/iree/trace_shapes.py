#!/usr/bin/env python3
"""
export/iree/trace_shapes.py — run PyTorch inference and log all tensor shapes.

This must be run BEFORE export.py to produce shapes_report.json.
The report drives dynamic_shapes specifications in export.py.

Usage:
    cd /path/to/VibeVoice
    uv run python export/iree/trace_shapes.py

Output:
    export/iree/shapes_report.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

# Resolve paths relative to repo root regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "model"
VOICES_DIR = REPO_ROOT / "demo" / "voices" / "streaming_model"
OUT_PATH    = Path(__file__).parent / "shapes_report.json"

sys.path.insert(0, str(REPO_ROOT))

TEST_TEXTS = [
    "Hello, this is a test of the speech synthesis system.",
    "The model uses a transformer-based architecture with five diffusion steps.",
]
VOICE_NAME = "en-Carter_man.pt"


def _tensor_info(t: Any) -> dict | None:
    if not isinstance(t, torch.Tensor):
        return None
    return {"shape": list(t.shape), "dtype": str(t.dtype)}


def _kv_info(past_kv) -> list[dict]:
    """Extract shape info from DynamicCache or tuple-of-tuples."""
    if past_kv is None:
        return []
    info = []
    try:
        # DynamicCache: .key_cache is a list of tensors
        for i, (k, v) in enumerate(zip(past_kv.key_cache, past_kv.value_cache)):
            info.append({
                "layer": i,
                "key":   _tensor_info(k),
                "value": _tensor_info(v),
            })
    except AttributeError:
        # Tuple of (key, value) tuples
        for i, (k, v) in enumerate(past_kv):
            info.append({
                "layer": i,
                "key":   _tensor_info(k),
                "value": _tensor_info(v),
            })
    return info


class ShapeTracer:
    """Hooks into model sub-modules to record input/output tensor shapes."""

    def __init__(self) -> None:
        self.records: dict[str, list[dict]] = {}
        self._hooks: list = []

    def attach(self, module: torch.nn.Module, name: str) -> None:
        self.records.setdefault(name, [])

        def hook(mod, inputs, output):
            record: dict = {"call": len(self.records[name])}
            # Inputs: flatten
            inp_shapes = []
            for x in inputs:
                info = _tensor_info(x)
                if info:
                    inp_shapes.append(info)
            record["inputs"] = inp_shapes
            # Output
            if isinstance(output, torch.Tensor):
                record["output"] = _tensor_info(output)
            elif hasattr(output, "last_hidden_state"):
                record["last_hidden_state"] = _tensor_info(output.last_hidden_state)
                record["kv_cache"] = _kv_info(output.past_key_values)
                if hasattr(output, "logits") and output.logits is not None:
                    record["logits"] = _tensor_info(output.logits)
            self.records[name].append(record)

        h = module.register_forward_hook(hook)
        self._hooks.append(h)

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def to_dict(self) -> dict:
        return self.records


def main() -> None:
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    device = "cpu"
    model_path = str(MODEL_PATH)
    voice_path = str(VOICES_DIR / VOICE_NAME)

    print(f"Loading processor from {model_path}")
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

    print(f"Loading model from {model_path}")
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        device_map="cpu",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    print(f"Loading voice from {voice_path}")
    all_prefilled = torch.load(voice_path, map_location="cpu", weights_only=False)

    # ── Attach hooks ────────────────────────────────────────────────────────
    tracer = ShapeTracer()
    tracer.attach(model.model.language_model,     "text_lm_forward")
    tracer.attach(model.model.tts_language_model, "tts_lm_forward")
    tracer.attach(model.model.acoustic_connector, "acoustic_connector")
    tracer.attach(model.model.prediction_head,    "diffusion_head")

    # Vocoder: the acoustic_tokenizer uses .decode(), not .forward().
    # Hook the decode method directly using a wrapper class approach.
    _orig_decode = model.model.acoustic_tokenizer.__class__.decode
    vocoder_shapes: list[dict] = []

    def _decode_hook(self_tok, *args, **kwargs):
        result = _orig_decode(self_tok, *args, **kwargs)
        entry: dict = {"call": len(vocoder_shapes)}
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                entry[f"input_{i}"] = _tensor_info(a)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                entry[f"kwarg_{k}"] = _tensor_info(v)
        if isinstance(result, torch.Tensor):
            entry["output"] = _tensor_info(result)
        vocoder_shapes.append(entry)
        return result

    model.model.acoustic_tokenizer.__class__.decode = _decode_hook
    tracer.records["vocoder"] = vocoder_shapes

    # ── Run inference for each test text ────────────────────────────────────
    report: dict[str, Any] = {"texts": []}

    for text_idx, text in enumerate(TEST_TEXTS):
        print(f"\n[{text_idx+1}/{len(TEST_TEXTS)}] {text}")
        # Clear records between runs
        for key in list(tracer.records.keys()):
            tracer.records[key] = []

        inputs = processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=all_prefilled,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to("cpu")

        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.5,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(all_prefilled),
            )

        report["texts"].append({
            "text":    text,
            "records": {k: v for k, v in tracer.records.items()},
        })
        print(f"  Hooks fired: { {k: len(v) for k, v in tracer.records.items()} }")

    tracer.detach()
    model.model.acoustic_tokenizer.__class__.decode = _orig_decode

    # ── Summarise unique shapes per component ────────────────────────────────
    summary: dict[str, Any] = {}
    for component in ["text_lm_forward", "tts_lm_forward", "acoustic_connector",
                       "diffusion_head", "vocoder"]:
        shapes_seen: list[dict] = []
        for text_record in report["texts"]:
            for call_record in text_record["records"].get(component, []):
                entry: dict = {}
                for field in ["inputs", "output", "last_hidden_state",
                              "logits", "kv_cache"]:
                    if field in call_record:
                        entry[field] = call_record[field]
                # De-duplicate
                if entry not in shapes_seen:
                    shapes_seen.append(entry)
        summary[component] = shapes_seen

    # Also capture scaling buffers
    summary["speech_scaling_factor"] = float(model.model.speech_scaling_factor)
    summary["speech_bias_factor"]    = float(model.model.speech_bias_factor)

    # LM / TTS-LM layer counts
    lm  = model.model.language_model
    tts = model.model.tts_language_model
    summary["text_lm_num_layers"] = len(lm.layers) if hasattr(lm, "layers") else "unknown"
    summary["tts_lm_num_layers"]  = len(tts.layers) if hasattr(tts, "layers") else "unknown"

    # Dump full report
    full: dict = {"summary": summary, "raw": report}
    OUT_PATH.write_text(json.dumps(full, indent=2, default=str))
    print(f"\nShapes report written to {OUT_PATH}")

    # Pretty-print summary
    print("\n── Shape Summary ──────────────────────────────────────────────")
    for comp, entries in summary.items():
        if isinstance(entries, list):
            print(f"\n{comp}: {len(entries)} unique call shape(s)")
            for e in entries[:3]:  # show first 3
                print(f"  {json.dumps(e, default=str)}")
        else:
            print(f"{comp}: {entries}")


if __name__ == "__main__":
    main()
