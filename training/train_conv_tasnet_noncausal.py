import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.conv_tasnet import build_conv_tasnet  # Conv-TasNet model
from training.losses.si_snr import SISNRLoss     # SI-SNR loss function
from data.dataloader import EARSWHAMDataLoader  # Your custom dataloader class
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize data loaders
data_loader = EARSWHAMDataLoader(
    base_dir="../datasets_final/EARS-WHAM16kHz",  # Path to the resampled dataset
    seg_length=16000,                            # Segment length
    batch_size=8,                                # Batch size
    num_workers=4                                # Number of workers for DataLoader
)
logger.info('Data loader initialized')

train_loader = data_loader.get_loader(split="train")
valid_loader = data_loader.get_loader(split="valid")

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_conv_tasnet(causal=False, num_sources=2).to(device)
criterion = SISNRLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
logger.info('Model, loss and optimizer initialized')

# Training loop
num_epochs = 1
logger.info(f'Starting training loop with {num_epochs} epochs')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(len(train_loader))
    for clean_waveform, noisy_waveform in train_loader:
        # Move data to the GPU if available
        clean_waveform = clean_waveform.to(device)
        noisy_waveform = noisy_waveform.to(device)

        # Forward pass
        estimated_waveform = model(noisy_waveform)

        # Compute loss
        loss = criterion(clean_waveform, estimated_waveform)
        print(f"Loss: {loss}")
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for clean_waveform, noisy_waveform in valid_loader:
            clean_waveform = clean_waveform.to(device)
            noisy_waveform = noisy_waveform.to(device)

            # Forward pass
            estimated_waveform = model(noisy_waveform)

            # Compute validation loss
            loss = criterion(clean_waveform, estimated_waveform)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_loader)

    logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
