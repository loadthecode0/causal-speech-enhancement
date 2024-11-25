import torch
from models.conv_tasnet import build_conv_tasnet  # Conv-TasNet model
from training.losses.si_snr import SISNRLoss
from data.dataloader import EARSWHAMDataLoader
import logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("transfer.log"),  logging.StreamHandler()]
)
logger = logging.getLogger()

'''
This is based on https://github.com/patrickloeber/pytorchTutorial/blob/master/15_transfer_learning.py
'''

# Paths for saving and loading weights
non_causal_weights_path = "non_causal_model.pth"
causal_weights_path = "causal_model.pth"

# Save the Non-Causal Model Weights
# After training the non-causal model
logger.info("Saving non-causal model weights...")
torch.save(model.state_dict(), non_causal_weights_path)
logger.info(f"Non-causal model weights saved at {non_causal_weights_path}")

# Initialize Causal Model and Load Weights
logger.info("Initializing causal model and loading weights from non-causal model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
causal_model = build_conv_tasnet(causal=True, num_sources=2).to(device)

# Load weights
non_causal_weights = torch.load(non_causal_weights_path)
causal_model.load_state_dict(non_causal_weights, strict=False)  # Allow partial weight loading
logger.info("Weights loaded successfully into causal model")

# Fine-Tune the Causal Model
criterion = SISNRLoss()
optimizer = torch.optim.Adam(causal_model.parameters(), lr=1e-4)  # Lower learning rate for fine-tuning

num_epochs = 2  # Fewer epochs for fine-tuning
for epoch in range(num_epochs):
    causal_model.train()
    running_loss = 0.0
    logger.info(f"Starting epoch {epoch + 1}/{num_epochs} for fine-tuning...")

    for clean_waveform, noisy_waveform in train_loader:
        clean_waveform = clean_waveform.to(device)
        noisy_waveform = noisy_waveform.to(device)

        # Forward pass
        estimated_waveform = causal_model(noisy_waveform)

        # Compute loss
        loss = criterion(clean_waveform, estimated_waveform)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        if batch_idx % 10 == 0:  # Log every 10 batches
            logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")


    avg_train_loss = running_loss / len(train_loader)
    logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. Average Train Loss: {avg_train_loss:.4f}")

    # Validation
    causal_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for clean_waveform, noisy_waveform in valid_loader:
            clean_waveform = clean_waveform.to(device)
            noisy_waveform = noisy_waveform.to(device)

            # Forward pass
            estimated_waveform = causal_model(noisy_waveform)

            # Compute validation loss
            loss = criterion(clean_waveform, estimated_waveform)
            val_loss += loss.item()

            if batch_idx % 10 == 0:  # Log every 10 validation batches
                logger.info(f"Validation Batch {batch_idx}/{len(valid_loader)}, Loss: {loss.item():.4f}")

    avg_val_loss = val_loss / len(valid_loader)
    logger.info(f"Validation completed. Average Validation Loss: {avg_val_loss:.4f}")

# Save the fine-tuned causal model
logger.info("Saving the fine-tuned causal model weights...")
torch.save(causal_model.state_dict(), causal_weights_path)
logger.info(f"Fine-tuned causal model weights saved at {causal_weights_path}")
