import whisper
import subprocess
import torch
import os
from pathlib import Path

def transcribe_with_turbo(video_path: str):
    """
    Runs Whisper large-v3-turbo entirely locally on RTX 3050
    """
    
    # Verify GPU is being used
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Running on: {device.upper()}")
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # Load model - first time downloads ~1.6GB, then cached locally
    print("\n📦 Loading Whisper large-v3-turbo model...")
    print("   (First time will download ~1.6GB, subsequent runs are instant)")
    model = whisper.load_model("large-v3-turbo", device=device)
    
    # Extract audio from video
    print("\n🎵 Extracting audio from video...")
    temp_audio = "temp_audio_16khz.wav"
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-ac', '1',           # Mono channel
        '-ar', '16000',       # 16kHz sample rate
        '-vn', '-y',          # No video, overwrite output
        temp_audio
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    print("\n📝 Transcribing (running entirely on your RTX 3050)...")
    
    # Transcribe with local GPU
    result = model.transcribe(
        temp_audio,
        language="en",           # Change as needed
        task="transcribe",
        verbose=True,            # Shows progress
        fp16=True,               # Keeps VRAM under 6GB
        temperature=0.0
    )
    
    # Cleanup temporary audio file
    os.remove(temp_audio)
    
    # Save transcript
    output_path = Path(video_path).stem + "_transcript.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TRANSCRIPTION - Whisper large-v3-turbo (Local)\n")
        f.write("="*80 + "\n\n")
        
        for segment in result["segments"]:
            start = format_time(segment["start"])
            end = format_time(segment["end"])
            f.write(f"[{start} -> {end}] {segment['text'].strip()}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FULL TEXT:\n")
        f.write("="*80 + "\n\n")
        f.write(result["text"])
    
    print(f"\n✅ Transcript saved to: {output_path}")
    print(f"\n📊 Stats:")
    print(f"   • Model: large-v3-turbo (809M parameters)")
    print(f"   • Local execution on RTX 3050")
    print(f"   • No API calls - everything offline")
    
    return result

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Run it - your video path
video_path = r"C:\Users\BALAJI\Downloads\Telegram Desktop\lesson17.mp4"
result = transcribe_with_turbo(video_path)