import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCES_PATH = "results/temporal_sequences.npy"

MODEL_PATH = "results/temporal_gru.pth"

FEATURES_PATH = "results/temporal_embeddings.npy"

SEQUENCE_LENGTH = 12

INPUT_SIZE = 512

HIDDEN_SIZE = 128

BATCH_SIZE = 16

EPOCHS = 20

LEARNING_RATE = 0.001


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("=" * 60)
print("VISIONSEEK — TEMPORAL GRU")
print("=" * 60)

print("\nDevice:", device)


# ============================================================
# LOAD TEMPORAL SEQUENCES
# ============================================================

print("\nLoading temporal sequences...")

sequences = np.load(SEQUENCES_PATH)

print(
    "Loaded sequences:",
    sequences.shape
)


# Convert to PyTorch tensor

data = torch.tensor(
    sequences,
    dtype=torch.float32
)

# ============================================================
# DATA LOADERS
# ============================================================

dataset = TensorDataset(data)

# Loader used for training
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Loader used for generating embeddings
# Keep the original temporal order
embedding_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# TEMPORAL GRU AUTOENCODER
# ============================================================

class TemporalGRU(nn.Module):

    def __init__(
        self,
        input_size=512,
        hidden_size=128
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.decoder = nn.Linear(
            hidden_size,
            input_size
        )


    def forward(self, x):

        outputs, hidden = self.gru(x)

        # Last temporal hidden state

        temporal_feature = hidden[-1]

        reconstruction = self.decoder(
            temporal_feature
        )

        return reconstruction, temporal_feature


# ============================================================
# MODEL
# ============================================================

model = TemporalGRU(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE
).to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...")

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for batch in train_loader:

        x = batch[0].to(device)

        optimizer.zero_grad()

        reconstruction, temporal_feature = model(x)

        # Target = average representation
        # of the sequence

        target = x.mean(
            dim=1
        )

        loss = criterion(
            reconstruction,
            target
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch "
        f"{epoch + 1:02d}/{EPOCHS} "
        f"- Loss: "
        f"{average_loss:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# ============================================================
# GENERATE TEMPORAL EMBEDDINGS
# ============================================================

print("\nGenerating temporal embeddings...")

model.eval()

temporal_embeddings = []


with torch.no_grad():

    for batch in embedding_loader:

        x = batch[0].to(device)

        _, temporal_feature = model(x)

        temporal_embeddings.append(
            temporal_feature.cpu().numpy()
        )


temporal_embeddings = np.vstack(
    temporal_embeddings
)


# ============================================================
# SAVE TEMPORAL EMBEDDINGS
# ============================================================

np.save(
    FEATURES_PATH,
    temporal_embeddings
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL GRU TRAINING COMPLETE")
print("=" * 60)

print(
    "\nTemporal embedding shape:",
    temporal_embeddings.shape
)

print(
    "\nModel saved to:",
    MODEL_PATH
)

print(
    "Temporal embeddings saved to:",
    FEATURES_PATH
)
