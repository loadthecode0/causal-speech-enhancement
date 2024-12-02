import os
import glob
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PairedAudioDataset(Dataset):
    def __init__(self, base_dir = "EARS-WHAM16kHz", dataset="train", transform=None):
        """
        Args:
            base_dir (str): Path to the base directory containing train, valid, and test subdirectories.
            dataset (str): Dataset split to use ("train", "valid", "test").
            transform (callable, optional): Optional transform to apply to the audio data.
        """
        assert dataset in ["train", "valid", "test"], "Invalid dataset split. Choose from 'train', 'valid', or 'test'."
        self.transform = transform

        # Set directories for clean and noisy files based on the dataset split
        self.clean_dir = os.path.join(base_dir, dataset, "clean")
        self.noisy_dir = os.path.join(base_dir, dataset, "noisy")

        # Gather and sort file names to ensure pairing
        self.clean_files = sorted(glob.glob(os.path.join(self.clean_dir, "*.wav")))
        self.noisy_files = sorted(glob.glob(os.path.join(self.noisy_dir, "*.wav")))

        # Ensure clean and noisy file lists match
        assert len(self.clean_files) == len(self.noisy_files), \
            f"Mismatch in the number of clean and noisy files for {dataset} dataset."
        assert all(os.path.basename(c) == os.path.basename(n) for c, n in zip(self.clean_files, self.noisy_files)), \
            f"File names in clean and noisy directories do not match for {dataset} dataset."

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_file_path = self.clean_files[idx]
        noisy_file_path = self.noisy_files[idx]

        # Load clean and noisy audio
        clean_waveform, clean_sample_rate = torchaudio.load(clean_file_path)
        noisy_waveform, noisy_sample_rate = torchaudio.load(noisy_file_path)

        # Apply optional transformation
        if self.transform:
            clean_waveform = self.transform(clean_waveform)
            noisy_waveform = self.transform(noisy_waveform)

        return clean_waveform, noisy_waveform  # Tuple of clean and noisy waveforms




# Create datasets for each split
#train_dataset = PairedAudioDataset(base_dir, dataset="train")
#valid_dataset = PairedAudioDataset(base_dir, dataset="valid")
#test_dataset = PairedAudioDataset(base_dir, dataset="test")


# Create DataLoaders
#train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
#valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)
#test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)