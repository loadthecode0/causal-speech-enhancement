import sys
import os
import time
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt  # For plotting training curves

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.conv_tasnet import build_conv_tasnet  # Conv-TasNet model
from training.losses.si_snr import SISNRLoss     # SI-SNR loss function
from data.dataloader import EARSWHAMDataLoader  
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define directories for saving models and training stats
model_dir = "/dtu/blackhole/18/212376/causal-speech-enhancement/models/saved-models/"
stats_dir = "/dtu/blackhole/18/212376/causal-speech-enhancement/experiments/"

# # Create directories if they don't exist
# os.makedirs(model_dir, exist_ok=True)
# os.makedirs(stats_dir, exist_ok=True)

# Initialize data loaders
data_loader = EARSWHAMDataLoader(
    base_dir="/dtu/blackhole/01/212577/datasets_final/EARS-WHAM16kHz",  # Path to the resampled dataset
    seg_length=16000,                            # Segment length
    batch_size=1,                                # Batch size
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
num_epochs = 20
checkpoint_interval = 5  # Save model every N epochs
best_val_loss = float('inf')  # Initialize best validation loss
train_losses = []  # Store training losses
val_losses = []    # Store validation losses

logger.info(f'Starting training loop with {num_epochs} epochs')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()  # Track time for ETA

    # Training with progress bar
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", unit="batch")
    for batch_idx, (clean_waveform, noisy_waveform) in enumerate(train_loader_tqdm):
        clean_waveform = clean_waveform.to(device)
        noisy_waveform = noisy_waveform.to(device)

        # Forward pass
        estimated_waveform = model(noisy_waveform)

        # Compute loss
        loss = criterion(clean_waveform, estimated_waveform)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=loss.item(), avg_loss=running_loss / (batch_idx + 1))

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation with progress bar
    model.eval()
    val_loss = 0.0
    valid_loader_tqdm = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]", unit="batch")
    with torch.no_grad():
        for clean_waveform, noisy_waveform in valid_loader_tqdm:
            clean_waveform = clean_waveform.to(device)
            noisy_waveform = noisy_waveform.to(device)

            # Forward pass
            estimated_waveform = model(noisy_waveform)

            # Compute validation loss
            loss = criterion(clean_waveform, estimated_waveform)
            val_loss += loss.item()
            valid_loader_tqdm.set_postfix(loss=loss.item(), avg_loss=val_loss / (len(valid_loader_tqdm) + 1))

    avg_val_loss = val_loss / len(valid_loader)
    val_losses.append(avg_val_loss)

    # Logging and ETA
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Time: {elapsed_time:.2f}s")
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Time: {elapsed_time:.2f}s")

    # Save model every N epochs
    checkpoint_path = os.path.join(model_dir, f"conv_tasnet_noncausal_epoch_{epoch + 1}.pth")
    if (epoch + 1) % checkpoint_interval == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")

    # Save model if validation loss improves
    best_model_path = os.path.join(model_dir, "conv_tasnet_noncausal_best_model.pth")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }, best_model_path)
        logger.info(f"Best model saved at {best_model_path} with validation loss {best_val_loss:.4f}")

# Save final training curve
training_curve_path = os.path.join(stats_dir, "conv_tasnet_noncausal_training_curve.png")
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig(training_curve_path)
logger.info(f"Training curve saved as '{training_curve_path}'.")
