import numpy as np
import json
import os


# ============================================================
# VISIONSEEK — TEMPORAL ACTIVITY ANALYSIS
# ============================================================

FRAME_EMBEDDINGS_PATH = "results/frame_embeddings.npy"

TEMPORAL_INDEX_PATH = "results/temporal_index.json"

OUTPUT_PATH = "results/temporal_activity.json"

SEQUENCE_LENGTH = 12


# ============================================================
# START
# ============================================================

print("=" * 60)
print("VISIONSEEK — TEMPORAL ACTIVITY")
print("=" * 60)


# ============================================================
# LOAD FRAME EMBEDDINGS
# ============================================================

print("\nLoading frame embeddings...")

embeddings = np.load(
    FRAME_EMBEDDINGS_PATH
)

print(
    "Frame embeddings:",
    embeddings.shape
)


# ============================================================
# LOAD TEMPORAL INDEX
# ============================================================

print("\nLoading temporal index...")

with open(
    TEMPORAL_INDEX_PATH,
    "r"
) as file:

    temporal_index = json.load(file)


print(
    "Temporal windows:",
    len(temporal_index)
)


# ============================================================
# COMPUTE TEMPORAL ACTIVITY
# ============================================================

print("\nComputing temporal activity...")

activity = []


for item in temporal_index:

    start_frame = item["start_frame"]

    end_frame = item["end_frame"]

    # Get embeddings belonging to
    # this temporal window

    sequence = embeddings[
        start_frame:end_frame + 1
    ]

    # --------------------------------------------------------
    # Calculate frame-to-frame changes
    # --------------------------------------------------------

    differences = np.linalg.norm(
        sequence[1:] - sequence[:-1],
        axis=1
    )

    # Average change within the window

    motion_score = float(
        np.mean(differences)
    )

    activity.append({

        "sequence_id":
            item["sequence_id"],

        "start_frame":
            start_frame,

        "end_frame":
            end_frame,

        "center_frame":
            item["center_frame"],

        "timestamp":
            item["timestamp"],

        "frame_path":
            item["frame_path"],

        "motion_score":
            motion_score

    })


# ============================================================
# NORMALIZE MOTION SCORES
# ============================================================

print("\nNormalizing activity scores...")

raw_scores = np.array([
    item["motion_score"]
    for item in activity
])


minimum = raw_scores.min()

maximum = raw_scores.max()


# Avoid division by zero

if maximum > minimum:

    normalized_scores = (
        (raw_scores - minimum)
        /
        (maximum - minimum)
    )

else:

    normalized_scores = np.zeros(
        len(raw_scores)
    )


# Store normalized scores

for i, score in enumerate(
    normalized_scores
):

    activity[i]["motion_score"] = float(
        score
    )


# ============================================================
# SAVE RESULTS
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
        activity,
        file,
        indent=4
    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL ACTIVITY ANALYSIS COMPLETE")
print("=" * 60)

print(
    "\nTotal temporal windows:",
    len(activity)
)


# ============================================================
# SHOW MOST ACTIVE WINDOWS
# ============================================================

sorted_activity = sorted(
    activity,
    key=lambda x: x["motion_score"],
    reverse=True
)


print("\nTop 5 Most Active Moments")
print("-" * 60)


for rank, item in enumerate(
    sorted_activity[:5],
    start=1
):

    print(
        f"\n#{rank}"
    )

    print(
        "Timestamp    :",
        item["timestamp"]
    )

    print(
        "Motion Score :",
        f"{item['motion_score']:.4f}"
    )

    print(
        "Frame        :",
        item["frame_path"]
    )


# ============================================================
# SAVE CONFIRMATION
# ============================================================

print(
    "\nSaved to:",
    OUTPUT_PATH
)

print("=" * 60)
