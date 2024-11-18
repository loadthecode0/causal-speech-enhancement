# -*- coding: utf-8 -*-
import torch

def truncate_audio_to_4_seconds(directory, duration_seconds=4):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if the file is an audio file (based on extension)
        if filename.lower().endswith(('.wav', '.mp3', '.flac', '.aac', '.ogg')):
            try:
                # Load the audio file
                waveform, sample_rate = torchaudio.load(file_path)
                print(f"Processing: {filename} | Original Duration: {waveform.size(1) / sample_rate:.2f} seconds")

                # Calculate the number of samples for the target duration
                target_num_samples = int(duration_seconds * sample_rate)

                # Check if truncation is needed
                if waveform.size(1) > target_num_samples:
                    # Randomly select a start point for truncation
                    max_start = waveform.size(1) - target_num_samples
                    start_sample = torch.randint(0, max_start, (1,)).item()

                    # Truncate the waveform
                    truncated_waveform = waveform[:, start_sample:start_sample + target_num_samples]

                    # Overwrite the original file with the truncated audio
                    torchaudio.save(file_path, truncated_waveform, sample_rate)
                    print(f"Truncated and overwritten: {file_path}")
                else:
                    print(f"No truncation needed for {filename}; shorter than {duration_seconds} seconds.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Skipping non-audio file: {filename}")

# Define the directory containing the audio files
directory = "/dtu/blackhole/01/212577/datasets_final"

# Truncate all audio files to 4 seconds
truncate_audio_to_4_seconds(directory, duration_seconds=4)
