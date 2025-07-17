import time
import os
import torch
from TTS.api import TTS



# Read input text from file
with open("input_text.txt", "r", encoding="utf-8") as f:
    input_text = f.read().strip()

print("Loading XTTS V2 voice cloning model...")
start_time = time.time()

# Use XTTS V2 model - multilingual voice cloning
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

print(f"Model loaded in {time.time() - start_time:.1f} seconds")

# Voice cloning with XTTS V2
start_time = time.time()
tts.tts_to_file(
    text=input_text,
    speaker=tts.speakers[37],     #"voice_samples/marty_voice.wav",
    language="en",
    file_path="output_audio/voice_chandra_mcfarlane.wav"
)
print(f"Voice cloning completed in {time.time() - start_time:.1f} seconds")
print("Output: output_audio/voice_marty.wav")


print("Built-in XTTS-V2 speakers: ")
for speaker in tts.speakers:
    print(speaker)
print("=====================")
'''
#all speakers generating files
print("Generating voice files for all available speakers...")
for i, speaker in enumerate(tts.speakers[3:]):
    output_file = f"output_audio/{speaker}.wav"
    print(f"Generating {speaker}.wav ({speaker})...")
    start_time = time.time()
    tts.tts_to_file(
        text=input_text,
        speaker=speaker,
        language="en",
        file_path=output_file
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Completed in {time.time() - start_time:.1f} seconds")
print(f"Finished Generating {len(tts.speakers)} additional voice files in output_audio/ directory")
'''