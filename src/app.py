import streamlit as st
import os
import sys
import json
import numpy as np
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# PROJECT PATHS
# ============================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FRAMES_DIR = os.path.join(SRC_DIR, "frames")

sys.path.append(SRC_DIR)

from query_analyser import analyze_query


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

EMBEDDINGS_PATH = os.path.join(
    RESULTS_DIR,
    "frame_embeddings.npy"
)

METADATA_PATH = os.path.join(
    RESULTS_DIR,
    "frame_metadata.json"
)

TEMPORAL_ACTIVITY_PATH = os.path.join(
    RESULTS_DIR,
    "temporal_activity.json"
)

TOP_K = 5
MIN_TIME_GAP = 5

CLIP_WEIGHT = 0.8
TEMPORAL_WEIGHT = 0.2


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VisionSeek | AI Video Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HEADER
# ============================================================

st.title("🔎 VisionSeek")

st.subheader(
    "AI-Powered Semantic & Temporal Video Search"
)

st.write(
    "Search inside videos using natural language. "
    "VisionSeek combines LLM query understanding, "
    "CLIP visual matching, temporal activity analysis, "
    "and hybrid ranking."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Search Settings")

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()

    st.header("🧠 AI Pipeline")

    st.markdown(
        """
        **1️⃣ LLM Query Analyzer**

        Understands natural-language queries and extracts:
        objects, actions, scenes and temporal intent.

        **2️⃣ CLIP**

        Converts text into a visual-semantic representation
        and matches it with video frames.

        **3️⃣ Temporal Analysis**

        Detects important activity and motion patterns
        across video sequences.

        **4️⃣ Hybrid Ranking**

        Combines CLIP similarity with temporal activity
        for better ranking of events.
        """
    )

    st.divider()

    st.caption(
        "VisionSeek — Multimodal Video Search Engine"
    )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    EMBEDDINGS_PATH,
    METADATA_PATH,
    TEMPORAL_ACTIVITY_PATH
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    st.error(
        "⚠️ Required VisionSeek files are missing."
    )

    for file in missing_files:
        st.write(file)

    st.info(
        "Please make sure these files exist inside "
        "the project's results folder."
    )

    st.stop()


# ============================================================
# LOAD CLIP MODEL
# ============================================================

@st.cache_resource
def load_clip():

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return model, processor


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    frame_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    with open(
        METADATA_PATH,
        "r"
    ) as file:

        metadata = json.load(file)

    with open(
        TEMPORAL_ACTIVITY_PATH,
        "r"
    ) as file:

        temporal_activity = json.load(file)

    activity_lookup = {
        int(item["center_frame"]):
        float(item["motion_score"])
        for item in temporal_activity
    }

    return (
        frame_embeddings,
        metadata,
        activity_lookup
    )


# ============================================================
# CREATE TEXT EMBEDDING
# ============================================================

def create_text_embedding(
    query,
    model,
    processor
):

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        text_output = model.get_text_features(
            **inputs
        )

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
            text_output.last_hidden_state[:, 0, :]
        )

    else:

        raise TypeError(
            f"Unexpected CLIP output type: "
            f"{type(text_output)}"
        )

    text_features = (
        text_features /
        text_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    return text_features.cpu().numpy()[0]


# ============================================================
# SEARCH ENGINE
# ============================================================

def perform_search(
    query,
    top_k,
    model,
    processor,
    frame_embeddings,
    metadata,
    activity_lookup
):

    # --------------------------------------------------------
    # ANALYZE QUERY
    # --------------------------------------------------------

    intent = analyze_query(query)

    visual_query = intent.get(
        "visual_query",
        query
    )

    # --------------------------------------------------------
    # CREATE TEXT EMBEDDING
    # --------------------------------------------------------

    query_embedding = create_text_embedding(
        visual_query,
        model,
        processor
    )

    # --------------------------------------------------------
    # CLIP SIMILARITY
    # --------------------------------------------------------

    similarities = np.dot(
        frame_embeddings,
        query_embedding
    )

    # --------------------------------------------------------
    # TEMPORAL REASONING
    # --------------------------------------------------------

    is_temporal = intent.get(
        "temporal",
        False
    )

    final_scores = similarities.copy()

    if is_temporal:

        for index in range(
            len(similarities)
        ):

            temporal_score = activity_lookup.get(
                index,
                0.0
            )

            final_scores[index] = (
                CLIP_WEIGHT * similarities[index]
                +
                TEMPORAL_WEIGHT * temporal_score
            )

    # --------------------------------------------------------
    # RANK RESULTS
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        final_scores
    )[::-1]

    selected_indices = []

    for index in ranked_indices:

        timestamp = metadata[index]["timestamp"]

        too_close = any(
            abs(
                timestamp -
                metadata[selected]["timestamp"]
            ) < MIN_TIME_GAP

            for selected in selected_indices
        )

        if not too_close:

            selected_indices.append(index)

        if len(selected_indices) >= top_k:

            break

    return (
        intent,
        selected_indices,
        similarities,
        final_scores,
        is_temporal
    )


# ============================================================
# SEARCH INTERFACE
# ============================================================

st.header("🔍 Search inside your video")

st.write(
    "Describe the event you want to find using natural language."
)

query = st.text_input(
    "Search query",
    placeholder=(
        "Example: Find the moment where a car turns left on the road"
    )
)

search_button = st.button(
    "🔎 Search Video",
    use_container_width=True
)


# ============================================================
# SEARCH EXECUTION
# ============================================================

if search_button:

    if not query.strip():

        st.warning(
            "Please enter a search query."
        )

        st.stop()

    # --------------------------------------------------------
    # LOAD MODELS AND DATA
    # --------------------------------------------------------

    with st.spinner(
        "Loading VisionSeek AI models..."
    ):

        try:

            model, processor = load_clip()

            (
                frame_embeddings,
                metadata,
                activity_lookup
            ) = load_data()

        except Exception as e:

            st.error(
                f"Unable to load VisionSeek resources: {e}"
            )

            st.stop()

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    with st.spinner(
        "🧠 Understanding query and searching video..."
    ):

        try:

            (
                intent,
                selected_indices,
                similarities,
                final_scores,
                is_temporal
            ) = perform_search(
                query,
                top_k,
                model,
                processor,
                frame_embeddings,
                metadata,
                activity_lookup
            )

        except Exception as e:

            st.error(
                f"Search failed: {e}"
            )

            st.stop()

    # ========================================================
    # QUERY UNDERSTANDING
    # ========================================================

    st.divider()

    st.header("🧠 Query Understanding")


    # --------------------------------------------------------
    # HELPER FUNCTION
    # --------------------------------------------------------

    def format_items(items):

        if not items:
            return "None"

        formatted = []

        for item in items:

            if isinstance(item, dict):

                # Try common keys returned by the LLM
                value = (
                    item.get("name")
                    or item.get("label")
                    or item.get("object")
                    or item.get("action")
                    or item.get("value")
                )

                if value is not None:
                    formatted.append(str(value))
                else:
                    formatted.append(
                        ", ".join(
                            str(v)
                            for v in item.values()
                        )
                    )

            else:

                formatted.append(
                    str(item)
                )

        return ", ".join(formatted)


    # --------------------------------------------------------
    # DISPLAY QUERY INFORMATION
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # VISUAL QUERY
    # --------------------------------------------------------

    with col1:

        st.markdown("### Visual Query")

        visual_query = intent.get(
            "visual_query",
            "N/A"
        )

        st.info(
            str(visual_query)
        )


    # --------------------------------------------------------
    # OBJECTS
    # --------------------------------------------------------

    with col2:

        st.markdown("### Objects")

        objects = intent.get(
            "objects",
            []
        )

        st.info(
            format_items(objects)
        )


    # --------------------------------------------------------
    # ACTIONS
    # --------------------------------------------------------

    with col3:

        st.markdown("### Actions")

        actions = intent.get(
            "actions",
            []
        )

        st.info(
            format_items(actions)
        )
    # ========================================================
    # SEARCH MODE
    # ========================================================

    if is_temporal:

        st.success(
            "⏱️ Temporal reasoning enabled — "
            "CLIP + Temporal Activity ranking"
        )

    else:

        st.info(
            "🖼️ Visual semantic search mode — "
            "CLIP ranking"
        )

    # ========================================================
    # SEARCH RESULTS
    # ========================================================

    st.divider()

    st.header("🎯 Search Results")

    st.write(
        f"Showing the top {len(selected_indices)} "
        "matching moments."
    )

    # --------------------------------------------------------
    # RESULT LOOP
    # --------------------------------------------------------

    for rank, index in enumerate(
        selected_indices,
        start=1
    ):

        frame_info = metadata[index]

        timestamp = frame_info["timestamp"]

        frame_path = frame_info["frame_path"]

        clip_score = similarities[index]

        final_score = final_scores[index]

        temporal_score = activity_lookup.get(
            index,
            0.0
        )

        # ----------------------------------------------------
        # FORMAT TIME
        # ----------------------------------------------------

        minutes = int(
            timestamp // 60
        )

        seconds = int(
            timestamp % 60
        )

        formatted_time = (
            f"{minutes:02d}:{seconds:02d}"
        )

        # ----------------------------------------------------
        # FIND FRAME
        # ----------------------------------------------------

        frame_filename = os.path.basename(
            frame_path
        )

        actual_frame_path = os.path.join(
            FRAMES_DIR,
            frame_filename
        )

        # ----------------------------------------------------
        # RESULT CONTAINER
        # ----------------------------------------------------

        with st.container(border=True):

            result_col1, result_col2 = st.columns(
                [1.1, 1.9]
            )

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            with result_col1:

                if os.path.exists(
                    actual_frame_path
                ):

                    image = Image.open(
                        actual_frame_path
                    )

                    st.image(
                        image,
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "Frame image not found."
                    )

                    st.caption(
                        actual_frame_path
                    )

            # ------------------------------------------------
            # RESULT INFORMATION
            # ------------------------------------------------

            with result_col2:

                st.subheader(
                    f"#{rank} — ⏱️ {formatted_time}"
                )

                st.write(
                    f"**Frame:** "
                    f"{os.path.basename(frame_path)}"
                )

                st.markdown(
                    "### 📊 Ranking Scores"
                )

                score_col1, score_col2 = st.columns(2)

                with score_col1:

                    st.metric(
                        "CLIP Similarity",
                        f"{clip_score:.4f}"
                    )

                with score_col2:

                    st.metric(
                        "Final Score",
                        f"{final_score:.4f}"
                    )

                if is_temporal:

                    st.metric(
                        "Temporal Activity",
                        f"{temporal_score:.4f}"
                    )

                    st.progress(
                        min(
                            max(
                                float(temporal_score),
                                0.0
                            ),
                            1.0
                        )
                    )

                st.caption(
                    "Higher score = stronger semantic match"
                )
