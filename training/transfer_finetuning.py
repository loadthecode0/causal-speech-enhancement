import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt  # For plotting training curves
import torch
from models.conv_tasnet import build_conv_tasnet  # Conv-TasNet model
from training.losses.si_snr import SISNRLoss
from data.dataloader import EARSWHAMDataLoader
import logging 
import time
'''
This is based on https://github.com/patrickloeber/pytorchTutorial/blob/master/15_transfer_learning.py
'''

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("transfer.log"),  logging.StreamHandler()]
)
logger = logging.getLogger()

# Directories for saving models and stats
model_dir = "/dtu/blackhole/18/212376/causal-speech-enhancement/models/saved-models/"
stats_dir = "/dtu/blackhole/18/212376/causal-speech-enhancement/experiments/"

# Initialize data loaders
data_loader = EARSWHAMDataLoader(
    base_dir="/dtu/blackhole/01/212577/datasets_final/EARS-WHAM16kHz",  # Path to the dataset
    seg_length=16000,  # Segment length
    batch_size=1,  # Batch size
    num_workers=4  # Number of workers for DataLoader
)
logger.info('Data loader initialized')

train_loader = data_loader.get_loader(split="train")
valid_loader = data_loader.get_loader(split="valid")

# Load pre-trained student model (causal)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student = build_conv_tasnet(causal=True, num_sources=2).to(device)
student_checkpoint = model_dir + "conv_tasnet_noncausal_best_model.pth"
student.load_state_dict(torch.load(student_checkpoint, map_location=device)["model_state_dict"])
logger.info("Student model loaded with non-causal weights")

# Compile the student model for optimization
student = torch.compile(student, backend="inductor")
logger.info("Student model compiled with torch.compile")

# Fine-Tune the Causal Model
criterion = SISNRLoss()
optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)  # Lower learning rate for fine-tuning

# Training parameters
num_epochs = 25
checkpoint_interval = 5
best_val_loss = float('inf')
train_losses = []
val_losses = []

logger.info(f"Starting transfer learning with {num_epochs} epochs")

# Define a compiled train step
def train_step(student, noisy_waveform, clean_waveform, criterion_task, optimizer):
    # Student's predictions
    student_output = student(noisy_waveform)

    # Loss computation
    loss = criterion_task(clean_waveform, student_output)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

# Compile the train step
train_step_compiled = torch.compile(train_step)

# Training loop
for epoch in range(num_epochs):
    student.train()
    running_loss = 0.0
    start_time = time.time()

    # Training with progress bar
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", unit="batch")
    for batch_idx, (clean_waveform, noisy_waveform) in enumerate(train_loader_tqdm):
        if batch_idx == 20: # initial test
            break
        clean_waveform = clean_waveform.to(device)
        noisy_waveform = noisy_waveform.to(device)

        # Perform a single training step
        loss = train_step_compiled(student, noisy_waveform, clean_waveform, optimizer).item()

        # Accumulate loss
        running_loss += loss
        train_loader_tqdm.set_postfix(loss=loss, avg_loss=running_loss / (batch_idx + 1))

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    student.eval()
    val_loss = 0.0
    valid_loader_tqdm = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]", unit="batch")
    with torch.no_grad():
        for clean_waveform, noisy_waveform in valid_loader_tqdm:
            clean_waveform = clean_waveform.to(device)
            noisy_waveform = noisy_waveform.to(device)

            # Student's predictions
            student_output = student(noisy_waveform)

            # Validation loss
            loss = criterion(clean_waveform, student_output)
            val_loss += loss.item()
            valid_loader_tqdm.set_postfix(loss=loss, avg_loss=val_loss / (len(valid_loader_tqdm) + 1))

    avg_val_loss = val_loss / len(valid_loader)
    val_losses.append(avg_val_loss)

    # Logging and ETA
    elapsed_time = time.time() - start_time
    logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Time: {elapsed_time:.2f}s")

    # Save model every N epochs
    checkpoint_path = os.path.join(model_dir, f"conv_tasnet_causal_transfer_finetuned_{epoch + 1}.pth")
    if (epoch + 1) % checkpoint_interval == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")

    # Save best model
    best_model_path = os.path.join(model_dir, f"conv_tasnet_causal_transfer_finetuned_best_model.pth")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }, best_model_path)
        logger.info(f"Best model saved at {best_model_path} with validation loss {best_val_loss:.4f}")

# Save final training curve
training_curve_path = os.path.join(stats_dir, "conv_tasnet_causal_transfer_finetuned_training_curve.png")
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve (Transfer Learning by finetuning noncausal weights Compiled)")
plt.legend()
plt.grid(True)
plt.savefig(training_curve_path)
logger.info(f"Training curve saved as '{training_curve_path}'.")
