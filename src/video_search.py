import os
import json
import numpy as np
import torch

from transformers import CLIPProcessor, CLIPModel
from query_analyser import analyze_query


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

EMBEDDINGS_PATH = "results/frame_embeddings.npy"
METADATA_PATH = "results/frame_metadata.json"
TEMPORAL_ACTIVITY_PATH = "results/temporal_activity.json"

TOP_K = 5
MIN_TIME_GAP = 5

# Temporal ranking weights
CLIP_WEIGHT = 0.8
TEMPORAL_WEIGHT = 0.2


# ==================================================
# LOAD CLIP MODEL
# ==================================================

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

print("CLIP model loaded successfully!\n")


# ==================================================
# LOAD FRAME EMBEDDINGS
# ==================================================

print("Loading frame embeddings...")

frame_embeddings = np.load(
    EMBEDDINGS_PATH
)

print(
    f"Loaded embeddings: {frame_embeddings.shape}"
)


# ==================================================
# LOAD FRAME METADATA
# ==================================================

with open(
    METADATA_PATH,
    "r"
) as file:

    metadata = json.load(file)

print(
    f"Loaded metadata: {len(metadata)} frames"
)


# ==================================================
# LOAD TEMPORAL ACTIVITY
# ==================================================

print("\nLoading temporal activity...")

with open(
    TEMPORAL_ACTIVITY_PATH,
    "r"
) as file:

    temporal_activity = json.load(file)


# Map center frame → motion score

activity_lookup = {
    item["center_frame"]: item["motion_score"]
    for item in temporal_activity
}

print(
    f"Loaded temporal activity: "
    f"{len(activity_lookup)} windows\n"
)


# ==================================================
# CREATE TEXT EMBEDDING
# ==================================================

def create_text_embedding(query):

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        text_output = model.get_text_features(
            **inputs
        )

    # Handle different Transformers outputs
    if isinstance(
        text_output,
        torch.Tensor
    ):

        text_features = text_output

    elif hasattr(
        text_output,
        "pooler_output"
    ):

        text_features = (
            text_output.pooler_output
        )

    elif hasattr(
        text_output,
        "last_hidden_state"
    ):

        text_features = (
            text_output.last_hidden_state[
                :, 0, :
            ]
        )

    else:

        raise TypeError(
            "Unexpected CLIP output type: "
            f"{type(text_output)}"
        )

    # Normalize text embedding

    text_features = (
        text_features /
        text_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    return text_features.cpu().numpy()[0]


# ==================================================
# SEARCH VIDEO
# ==================================================

def search_video(
    query,
    top_k=TOP_K
):

    print("\n" + "=" * 60)
    print("VISIONSEEK SEARCH")
    print("=" * 60)

    print(
        "\nAnalyzing query with LLM..."
    )

    # --------------------------------------------------
    # LLM QUERY ANALYSIS
    # --------------------------------------------------

    intent = analyze_query(query)

    print(
        "\nStructured Search Intent"
    )

    print("-" * 60)

    print(
        json.dumps(
            intent,
            indent=4
        )
    )

    visual_query = intent[
        "visual_query"
    ]

    print(
        f"\nQuery: {query}"
    )

    # --------------------------------------------------
    # CREATE QUERY EMBEDDING
    # --------------------------------------------------

    query_embedding = (
        create_text_embedding(
            visual_query
        )
    )

    # --------------------------------------------------
    # CLIP SIMILARITY
    # --------------------------------------------------

    similarities = np.dot(
        frame_embeddings,
        query_embedding
    )

    # --------------------------------------------------
    # DETERMINE WHETHER TEMPORAL
    # --------------------------------------------------

    is_temporal = intent.get(
        "temporal",
        False
    )

    print(
        "\nTemporal reasoning:",
        "ENABLED" if is_temporal
        else "DISABLED"
    )

    # --------------------------------------------------
    # CALCULATE FINAL SCORES
    # --------------------------------------------------

    final_scores = similarities.copy()

    if is_temporal:

        print(
            "Ranking mode: "
            "CLIP + Temporal Activity"
        )

        for index in range(
            len(similarities)
        ):

            # Activity is associated with
            # the center frame of a temporal window

            temporal_score = activity_lookup.get(
                index,
                0.0
            )

            final_scores[index] = (
                CLIP_WEIGHT *
                similarities[index]
                +
                TEMPORAL_WEIGHT *
                temporal_score
            )

    else:

        print(
            "Ranking mode: CLIP only"
        )

    # --------------------------------------------------
    # RANK RESULTS
    # --------------------------------------------------

    ranked_indices = np.argsort(
        final_scores
    )[::-1]

    # --------------------------------------------------
    # SELECT TEMPORALLY DIVERSE RESULTS
    # --------------------------------------------------

    selected_indices = []

    for index in ranked_indices:

        timestamp = metadata[
            index
        ]["timestamp"]

        # Check whether this result
        # is too close to an existing result

        too_close = any(
            abs(
                timestamp -
                metadata[selected][
                    "timestamp"
                ]
            ) < MIN_TIME_GAP

            for selected in selected_indices
        )

        if not too_close:

            selected_indices.append(
                index
            )

        if len(
            selected_indices
        ) >= top_k:

            break

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print(
        "\nTop Matching Frames"
    )

    print("-" * 60)

    results = []

    for rank, index in enumerate(
        selected_indices,
        start=1
    ):

        frame_info = metadata[
            index
        ]

        clip_score = float(
            similarities[index]
        )

        final_score = float(
            final_scores[index]
        )

        temporal_score = float(
            activity_lookup.get(
                index,
                0.0
            )
        )

        timestamp = frame_info[
            "timestamp"
        ]

        frame_path = frame_info[
            "frame_path"
        ]

        minutes = int(
            timestamp // 60
        )

        seconds = int(
            timestamp % 60
        )

        formatted_time = (
            f"{minutes:02d}:{seconds:02d}"
        )

        # --------------------------------------------------
        # RESULT OBJECT
        # --------------------------------------------------

        result = {

            "rank": rank,

            "timestamp": timestamp,

            "formatted_time":
                formatted_time,

            "clip_similarity":
                clip_score,

            "temporal_score":
                temporal_score,

            "final_score":
                final_score,

            "frame_path":
                frame_path

        }

        results.append(
            result
        )

        # --------------------------------------------------
        # PRINT RESULT
        # --------------------------------------------------

        print(
            f"\n#{rank}"
        )

        print(
            f"Timestamp       : "
            f"{formatted_time}"
        )

        print(
            f"CLIP Score      : "
            f"{clip_score:.4f}"
        )

        if is_temporal:

            print(
                f"Temporal Score  : "
                f"{temporal_score:.4f}"
            )

            print(
                f"Final Score     : "
                f"{final_score:.4f}"
            )

        print(
            f"Frame           : "
            f"{frame_path}"
        )

    return results


# ==================================================
# MAIN PROGRAM
# ==================================================

if __name__ == "__main__":

    while True:

        query = input(
            "\nEnter your search query "
            "(or type 'exit'): "
        )

        if query.lower() == "exit":

            print(
                "\nExiting VisionSeek..."
            )

            break

        if not query.strip():

            print(
                "Please enter a valid query."
            )

            continue

        search_video(
            query
        )
