import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"

FRAMES_DIR = "frames"
METADATA_PATH = "results/frame_metadata.json"
EMBEDDINGS_PATH = "results/frame_embeddings.npy"

BATCH_SIZE = 8


# --------------------------------------------------
# Load CLIP
# --------------------------------------------------

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

print("CLIP model loaded successfully!\n")


# --------------------------------------------------
# Load frame metadata
# --------------------------------------------------

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

print(f"Frames found: {len(metadata)}")


# --------------------------------------------------
# Generate image embeddings
# --------------------------------------------------

all_embeddings = []

total_frames = len(metadata)

for start in range(0, total_frames, BATCH_SIZE):

    batch_metadata = metadata[start:start + BATCH_SIZE]

    images = []

    valid_metadata = []

    for item in batch_metadata:

        image_path = item["frame_path"]

        if not os.path.exists(image_path):
            print(f"Warning: Missing frame: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        images.append(image)
        valid_metadata.append(item)

    if not images:
        continue

    # Prepare images for CLIP
    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    # Generate embeddings
    with torch.no_grad():

        image_output = model.get_image_features(
            **inputs
        )

        # Extract tensor from Transformers model output
        if isinstance(image_output, torch.Tensor):

            image_features = image_output

        elif hasattr(image_output, "pooler_output"):

            image_features = image_output.pooler_output

        elif hasattr(image_output, "last_hidden_state"):

            image_features = image_output.last_hidden_state[:, 0, :]

        else:

            raise TypeError(
                f"Unexpected CLIP output type: {type(image_output)}"
            )


    # Normalize embeddings
    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    )

    embeddings = image_features.cpu().numpy()
    all_embeddings.append(embeddings)

    processed = min(
        start + BATCH_SIZE,
        total_frames
    )

    print(
        f"Processed {processed}/{total_frames} frames"
    )


# --------------------------------------------------
# Combine embeddings
# --------------------------------------------------

if not all_embeddings:
    raise RuntimeError("No embeddings were generated.")


final_embeddings = np.vstack(all_embeddings)


# --------------------------------------------------
# Save embeddings
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

np.save(
    EMBEDDINGS_PATH,
    final_embeddings
)


# --------------------------------------------------
# Display information
# --------------------------------------------------

print("\nEmbedding generation complete!")
print("----------------------------------------")

print(
    f"Embedding shape: {final_embeddings.shape}"
)

print(
    f"Saved to: {EMBEDDINGS_PATH}"
)
