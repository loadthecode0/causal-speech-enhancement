pip install torchaudio

import os
import torchaudio
import torchaudio.transforms as T

def resample_audio_files_in_place(directory, target_sample_rate=8000):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if the file is an audio file (based on extension)
        if filename.lower().endswith(('.wav', '.mp3', '.flac', '.aac', '.ogg')):
            try:
                # Load the audio file
                waveform, original_sample_rate = torchaudio.load(file_path)

                # Print file information
                print(f"Processing: {filename} | Original Sample Rate: {original_sample_rate} Hz")

                # Check if resampling is needed
                if original_sample_rate != target_sample_rate:
                    # Define the resampler
                    resampler = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)

                    # Resample the waveform
                    resampled_waveform = resampler(waveform)

                    # Overwrite the original file with the resampled audio
                    torchaudio.save(file_path, resampled_waveform, target_sample_rate)
                    print(f"Resampled and overwritten: {file_path}")
                else:
                    print(f"No resampling needed for {filename}.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Skipping non-audio file: {filename}")

# Define the directory containing the audio files
directory = "/dtu/blackhole/01/212577/datasets_final"

# Resample all audio files to 8 kHz in place
resample_audio_files_in_place(directory, target_sample_rate=8000)
