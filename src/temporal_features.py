import os
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_PATH = "results/frame_embeddings.npy"

SEQUENCE_LENGTH = 12

OUTPUT_PATH = "results/temporal_sequences.npy"


# ============================================================
# LOAD CLIP EMBEDDINGS
# ============================================================

print("=" * 60)
print("VISIONSEEK — TEMPORAL FEATURE EXTRACTION")
print("=" * 60)

print("\nLoading CLIP embeddings...")

embeddings = np.load(EMBEDDINGS_PATH)

print(
    f"Loaded embeddings: {embeddings.shape}"
)


# ============================================================
# CREATE TEMPORAL SEQUENCES
# ============================================================

sequences = []

num_frames = embeddings.shape[0]

print("\nCreating temporal sequences...")

for start in range(
    0,
    num_frames - SEQUENCE_LENGTH + 1
):

    end = start + SEQUENCE_LENGTH

    sequence = embeddings[start:end]

    sequences.append(sequence)


# Convert to NumPy array

sequences = np.array(sequences)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

np.save(
    OUTPUT_PATH,
    sequences
)


# ============================================================
# INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL SEQUENCE EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"\nNumber of frames       : {num_frames}"
)

print(
    f"Sequence length        : {SEQUENCE_LENGTH}"
)

print(
    f"Number of sequences    : {len(sequences)}"
)

print(
    f"Embedding dimension    : {embeddings.shape[1]}"
)

print(
    f"Output shape           : {sequences.shape}"
)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)
