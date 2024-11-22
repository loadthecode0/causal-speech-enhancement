import os
import glob
import torch 
import torchaudio
from torch.utils.data import Dataset

class EARSWHAMAudioDataset(Dataset):
    def __init__(self, base_dir = "data/resampled/EARS-WHAM-16.0kHz", dataset="train", transform=None, seg_length = 16000):
        """
        Args:
            base_dir (str): Path to the base directory containing train, valid, and test subdirectories.
            dataset (str): Dataset split to use ("train", "valid", "test").
            transform (callable, optional): Optional transform to apply to the audio data.
        """
        assert dataset in ["train", "valid", "test"], "Invalid dataset split. Choose from 'train', 'valid', or 'test'."
        self.transform = transform
        self.seg_length = seg_length

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
    
        self.data = []
        for clean_path, noisy_path in zip(self.clean_files, self.noisy_files):
            clean_waveform, _ = torchaudio.load(clean_path)
            noisy_waveform, _ = torchaudio.load(noisy_path)
            self.data.append((clean_waveform, noisy_waveform))


    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_waveform, noisy_waveform = self.data[idx]

        audio_length = clean_waveform.size(1)


        #We find a random starting point for the segment
        if audio_length > self.seg_length:
            start = torch.randint(0, audio_length - self.seg_length, (1, )).item()
            clean_waveform = clean_waveform[:, start: start + self.segment_length]
            noisy_waveform = noisy_waveform[:, start:start + self.segment_length]
        else:
            pad_length = self.seg_length - audio_length
            clean_waveform = torch.nn.functional.pad(clean_waveform, (0, pad_length))
            noisy_waveform = torch.nn.functional.pad(noisy_waveform, (0, pad_length))


        # Apply optional transformation
        if self.transform:
            clean_waveform = self.transform(clean_waveform)
            noisy_waveform = self.transform(noisy_waveform)

        return clean_waveform, noisy_waveform  # Tuple of clean and noisy waveforms

'''
DataLoader Usage:
# Create dataset instances for each split
train_dataset = AudioDataset(base_dir="EARS-WHAM-16.0kHz", dataset="train", seg_length=16000)
valid_dataset = AudioDataset(base_dir="EARS-WHAM-16.0kHz", dataset="valid", seg_length=16000)
test_dataset = AudioDataset(base_dir="EARS-WHAM-16.0kHz", dataset="test", seg_length=16000)

# Define DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# Example usage: Iterate through the training DataLoader
for batch_idx, (clean_batch, noisy_batch) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}")
    print(f"Clean Shape: {clean_batch.shape}, Noisy Shape: {noisy_batch.shape}")

For Training:
for epoch in range(num_epochs):
    for clean_batch, noisy_batch in train_loader:
        # Move to GPU if available
        clean_batch = clean_batch.to(device)
        noisy_batch = noisy_batch.to(device)

        # Forward pass
        output = model(noisy_batch)

        # Loss calculation and optimization
        loss = loss_function(output, clean_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed.")
'''