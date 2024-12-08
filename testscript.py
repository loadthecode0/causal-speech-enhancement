import sys
import os
import torch
import torchaudio
from torchaudio.pipelines import SQUIM_SUBJECTIVE
from tqdm import tqdm
import json
from data.dataloader import EARSWHAMDataLoader
from torchaudio.functional import downmix_to_mono
from models.conv_tasnet import build_conv_tasnet  # Conv-TasNet model

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model directory
saved_model_dir = '/dtu/blackhole/18/212376/causal-speech-enhancement/models/saved-models/'

# Function to load a pretrained model
def load_model(model_name, causal):
    model = build_conv_tasnet(causal=causal, num_sources=2).to(device)
    model_path = os.path.join(saved_model_dir, f"{model_name}_best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.eval()  # Set model to evaluation mode
    return model

# Function to calculate MOS
def calculate_mos(clean_waveform, enhanced_waveform, subjective_model):
    # Ensure both clean and enhanced waveforms are mono
    if clean_waveform.size(1) > 1:  # Check for multiple channels
        clean_waveform = downmix_to_mono(clean_waveform)
    if enhanced_waveform.size(1) > 1:  # Check for multiple channels
        enhanced_waveform = downmix_to_mono(enhanced_waveform)
    
    # Pass the waveforms to the subjective model
    mos = subjective_model(enhanced_waveform[0:1, :], clean_waveform[0:1, :])
    return mos[0].item()

def main():
    # Load models
    model_names = ["conv_tasnet_noncausal", "conv_tasnet_causal", "conv_tasnet_causal_transfer_kld"]
    model_causal = [False, True, True]
    models = [load_model(name, causal) for name, causal in zip(model_names, model_causal)]

    # Initialize data loaders
    data_loader = EARSWHAMDataLoader(
        base_dir="/dtu/blackhole/01/212577/datasets_final/EARS-WHAM16kHz",
        seg_length=16000,
        batch_size=1,
        num_workers=4
    )
    test_loader = data_loader.get_loader(split="test")
    print('Test dataloader loaded')

    # Initialize SQUIM-Subjective model
    subjective_model = SQUIM_SUBJECTIVE.get_model()
    print('SQUIM model loaded')

    # Iterate through test data
    results = []
    for batch_idx, (noisy_waveform, clean_waveform) in tqdm(enumerate(test_loader), desc="Processing Batches"):
        # Ensure non-overlapping noisy and clean samples
        noisy_waveform, clean_waveform = noisy_waveform.to(device), clean_waveform.to(device)

        mos_scores = []
        with torch.no_grad():
            for model in models:
                denoised_waveform = model(noisy_waveform)
                mos_score = calculate_mos(clean_waveform, denoised_waveform, subjective_model)
                mos_scores.append(mos_score)

        # Append results
        results.append({
            "batch_idx": batch_idx,
            "mos_scores": mos_scores
        })
        print(f"Batch {batch_idx}: MOS Scores - {mos_scores}")

    # Save results
    with open("mos_results.json", "w") as f:
        json.dump(results, f)
    print("MOS results saved to 'mos_results.json'")

if __name__ == "__main__":
    main()
