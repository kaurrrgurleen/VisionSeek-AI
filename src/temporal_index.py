import numpy as np
import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

TEMPORAL_EMBEDDINGS_PATH = (
    "results/temporal_embeddings.npy"
)

METADATA_PATH = (
    "results/frame_metadata.json"
)

OUTPUT_PATH = (
    "results/temporal_index.json"
)

SEQUENCE_LENGTH = 12


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("VISIONSEEK — TEMPORAL INDEX")
print("=" * 60)

temporal_embeddings = np.load(
    TEMPORAL_EMBEDDINGS_PATH
)

print(
    "\nTemporal embeddings:",
    temporal_embeddings.shape
)


with open(
    METADATA_PATH,
    "r"
) as file:

    metadata = json.load(file)


print(
    "Frame metadata:",
    len(metadata)
)


# ============================================================
# BUILD INDEX
# ============================================================

temporal_index = []


for i in range(
    len(temporal_embeddings)
):

    start_frame = i

    end_frame = (
        i + SEQUENCE_LENGTH - 1
    )

    center_frame = (
        start_frame
        + SEQUENCE_LENGTH // 2
    )

    # Make sure the center frame
    # exists in metadata

    if center_frame >= len(metadata):
        continue

    frame_info = metadata[
        center_frame
    ]

    temporal_index.append({

        "sequence_id": i,

        "start_frame": start_frame,

        "end_frame": end_frame,

        "center_frame": center_frame,

        "timestamp": frame_info[
            "timestamp"
        ],

        "frame_path": frame_info[
            "frame_path"
        ]

    })


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w"
) as file:

    json.dump(
        temporal_index,
        file,
        indent=4
    )


# ============================================================
# RESULTS
# ============================================================

print(
    "\nTemporal index created:"
)

print(
    "Number of temporal windows:",
    len(temporal_index)
)

print(
    "\nExample entries:"
)

for item in temporal_index[:3]:

    print(item)


print(
    "\nSaved to:",
    OUTPUT_PATH
)
