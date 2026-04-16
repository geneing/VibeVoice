import os
import copy
import time
import torch
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor


def generate_speech(
    text: str,
    model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
    voice_sample_path: str = None,
    output_path: str = "./output.wav",
    device: str = "cuda",
    cfg_scale: float = 1.5,
) -> str:
    """
    Generate speech from text using VibeVoice streaming model.
    
    Args:
        text: Input text to synthesize
        model_path: Path to the HuggingFace model
        voice_sample_path: Path to voice embedding (.pt file)
        output_path: Where to save the generated audio
        device: Device to use (cuda, mps, cpu). Defaults to cpu
        cfg_scale: CFG scale for generation
    
    Returns:
        Path to the generated audio file
    """
    
    # Normalize device name
    if device.lower() == "mpx":
        device = "mps"
    
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load processor
    print(f"Loading processor from {model_path}")
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
    
    # Determine dtype and attention implementation based on device
    if device == "mps":
        load_dtype = torch.float32
        attn_impl = "sdpa"
    elif device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl = "flash_attention_2"
    else:  # cpu
        load_dtype = torch.float32
        attn_impl = "sdpa"
    
    print(f"Using dtype: {load_dtype}, attn_implementation: {attn_impl}")
    
    # Load model
    print(f"Loading model from {model_path}")
    try:
        if device == "mps":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                attn_implementation=attn_impl,
                device_map=None,
            )
            model.to("mps")
        elif device == "cuda":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map="cuda",
                attn_implementation=attn_impl,
            )
        else:  # cpu
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map="cpu",
                attn_implementation=attn_impl,
            )
    except Exception as e:
        if attn_impl == 'flash_attention_2':
            print(f"Warning: Failed to load with flash_attention_2, falling back to SDPA: {e}")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=(device if device in ("cuda", "cpu") else None),
                attn_implementation='sdpa'
            )
            if device == "mps":
                model.to("mps")
        else:
            raise
    
    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)
    
    # Load voice embedding
    target_device = device if device != "cpu" else "cpu"
    if voice_sample_path and os.path.exists(voice_sample_path):
        print(f"Loading voice embedding from {voice_sample_path}")
        all_prefilled_outputs = torch.load(voice_sample_path, map_location=target_device, weights_only=False)
    else:
        all_prefilled_outputs = None
        if voice_sample_path:
            print(f"Warning: Voice sample not found at {voice_sample_path}, continuing without voice conditioning")
    
    # Prepare inputs
    print(f"Processing text input")
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # Move inputs to target device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)
    
    # Generate speech
    print(f"Generating speech with cfg_scale={cfg_scale}")
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={'do_sample': False},
        verbose=False,
        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs) if all_prefilled_outputs is not None else None,
    )
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Save audio
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        processor.save_audio(
            outputs.speech_outputs[0],
            output_path=output_path,
        )
        print(f"Audio saved to {output_path}")
        
        # Calculate metrics
        sample_rate = 24000
        audio_samples = outputs.speech_outputs[0].shape[-1]
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        print(f"Audio duration: {audio_duration:.2f}s, RTF: {rtf:.2f}x")
        
        return output_path
    else:
        print("Error: No audio output generated")
        return None


if __name__ == "__main__":
    # Example usage
    text = "Hello, this is a test of the speech synthesis system."
    voice_path = "./demo/voices/streaming_model/en-Carter_man.pt"  # Update with actual voice path
    
    output = generate_speech(
        text=text,
        voice_sample_path=voice_path,
        output_path="./test_output.wav",
        cfg_scale=1.5
    )
