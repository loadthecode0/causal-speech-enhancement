import os
import torchaudio
import torchaudio.transforms as T

def resample_audio_files(input_directory, output_directory, target_sample_rate=16000):
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(input_directory):
        for filename in files:
            # Check if the file is an audio file (based on extension)
            if filename.lower().endswith('.wav'):
                input_file_path = os.path.join(root, filename)

                # Construct the output file path by replicating the directory structure
                relative_path = os.path.relpath(root, input_directory)
                output_dir = os.path.join(output_directory, relative_path)
                output_file_path = os.path.join(output_dir, filename)

                # Ensure the output directory exists
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                try:
                    # Load the audio file
                    waveform, original_sample_rate = torchaudio.load(input_file_path)

                    # Print file information
                    print(f"Processing: {input_file_path} | Original Sample Rate: {original_sample_rate} Hz")

                    # Check if resampling is needed
                    if original_sample_rate != target_sample_rate:
                        # Define the resampler
                        resampler = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)

                        # Resample the waveform
                        resampled_waveform = resampler(waveform)

                        # Save the resampled audio to the output directory
                        torchaudio.save(output_file_path, resampled_waveform, target_sample_rate)
                        print(f"Resampled and saved to: {output_file_path}")
                    else:
                        # Copy the original file to the output directory
                        torchaudio.save(output_file_path, waveform, original_sample_rate)
                        print(f"Copied without resampling: {output_file_path}")

                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")
            else:
                # Optional: Handle non-audio files or other file types
                pass  # Or you can log skipped files if needed

                
# Define the directory containing the audio files
input_directory = "../datasets_final/EARS-WHAM"
output_directory = "../datasets_final/EARS-WHAM16kHz"

# Resample all audio files to 16 kHz in place
resample_audio_files(input_directory, output_directory, target_sample_rate=16000)
