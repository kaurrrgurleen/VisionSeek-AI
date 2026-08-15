# 🔎 VisionSeek-AI

### AI-Powered Natural Language Video Search with Temporal Understanding

VisionSeek-AI is an AI-powered video search system that allows users to find relevant moments in a video using **natural-language queries** instead of manually searching through the entire video.

The system combines **Large Language Models (LLMs), CLIP visual embeddings, temporal feature extraction, GRU-based sequence modeling, and relevance scoring** to understand both *what* appears in a video and *how events occur over time*.

---

## 📸 Project Demo
<img width="1366" height="697" alt="Screenshot (260)" src="https://github.com/user-attachments/assets/478e2adb-6e9e-492a-8503-057af1a691da" />

## 📌 Problem Statement

Searching for a specific event inside a long video can be time-consuming when users have to manually watch or scan through the footage.

Traditional video search methods often rely on:

* Manual browsing
* Keyword-based metadata
* Fixed timestamps
* Simple image or object matching

These approaches have difficulty understanding natural-language descriptions of events.

For example, a user may ask:

> **"Find the moment when a car turns left."**

VisionSeek-AI aims to understand this query and identify the most relevant portion of the video automatically.

---

## 🎯 Objectives

The main objectives of VisionSeek-AI are:

* Enable **natural-language video search**
* Understand the semantic meaning of user queries
* Extract meaningful visual information from video frames
* Capture **temporal relationships between consecutive frames**
* Identify relevant video moments rather than only individual frames
* Rank candidate moments according to their relevance to the query
* Provide an efficient foundation for intelligent video retrieval

---

## ✨ Key Features

### 🧠 Natural Language Query Understanding

The system analyzes the user's query using an LLM to understand the search intent and determine whether temporal reasoning is required.

### 👁️ Visual Understanding with CLIP

CLIP embeddings are used to represent video frames in a semantic feature space, allowing text queries and visual content to be compared.

### ⏱️ Temporal Understanding

Instead of evaluating frames independently, VisionSeek-AI groups consecutive frames into temporal windows to understand events occurring over time.

### 🔄 GRU-Based Temporal Modeling

A Gated Recurrent Unit (GRU) processes temporal sequences and generates representations that capture relationships between frames.

### 📊 Relevance Ranking

Candidate moments are ranked using a combined relevance score based on:

* CLIP semantic similarity
* Motion information

### 💬 Natural-Language Search

Users can search using queries such as:

* `car turning left`
* `person walking outdoors`
* `vehicle near building`

---

# 🏗️ System Workflow

The overall VisionSeek-AI pipeline is:

```text
                 User Query
                     │
                     ▼
             ┌───────────────┐
             │ LLM Query     │
             │ Analysis      │
             └───────┬───────┘
                     │
                     ▼
             Query Representation
                     │
                     │
Video ──► Frame Extraction
                     │
                     ▼
             ┌───────────────┐
             │ CLIP Encoder  │
             └───────┬───────┘
                     │
                     ▼
             Frame Embeddings
                     │
                     ▼
          Temporal Window Creation
                     │
                     ▼
             ┌───────────────┐
             │ GRU Temporal   │
             │ Encoder        │
             └───────┬───────┘
                     │
                     ▼
          Temporal Representations
                     │
                     ▼
             Relevance Scoring
                     │
                     ▼
             Ranked Video Moments
                     │
                     ▼
              Search Results
```

---

# 🧩 Architecture

VisionSeek-AI consists of several major components.

### 1. Query Analysis

The user's natural-language query is passed to the query analysis module.

The LLM extracts the important semantic information and determines whether the query requires temporal reasoning.

<img width="1366" height="692" alt="Screenshot (261)" src="https://github.com/user-attachments/assets/ddee56aa-a4af-4b36-a8ec-68fc70fdadc3" />

For example:

```text
Query:
"Find the moment when a car turns left."

        ↓

Query Analysis

        ↓

Semantic concept:
car + turning + left

Temporal requirement:
Yes
```

---

### 2. Video Frame Processing

The input video is processed frame-by-frame.

Each frame becomes an individual visual representation that can later be compared with the user's query.

---

### 3. CLIP Feature Extraction

CLIP is used to generate visual embeddings for the extracted frames.

In the implemented pipeline, the processed video produced:

```text
307 frames
307 × 512 embedding matrix
```

Each frame is therefore represented by a **512-dimensional feature vector**.

These embeddings allow semantic comparison between the user's text query and visual content.

---

### 4. Temporal Window Creation

Individual frames do not always provide enough information to understand an event.

For example:

Frame 1 → Car approaching
<img width="1366" height="686" alt="Screenshot (263)" src="https://github.com/user-attachments/assets/01710584-2117-4a3d-a263-93de275bea12" />

Frame 2 → Car approaching
<img width="1366" height="677" alt="Screenshot (264)" src="https://github.com/user-attachments/assets/2b733b42-e169-4477-a2d3-69114dd06599" />

Frame 3 → Car beginning to turn
<img width="1366" height="679" alt="Screenshot (265)" src="https://github.com/user-attachments/assets/42c36bbf-f70a-4fba-8f26-9fdb63dab7bf" />

Frame 4 → Car turning
<img width="1366" height="684" alt="Screenshot (266)" src="https://github.com/user-attachments/assets/066e0fdc-2cb5-4b9b-ac36-3b44954b092e" />

Frame 5 → Car completing turn
<img width="1366" height="684" alt="Screenshot (267)" src="https://github.com/user-attachments/assets/2ba02610-75fc-4758-8165-1b6afef53e47" />


Together, these frames describe the event much better than any individual frame.

VisionSeek-AI therefore creates **12-frame sliding windows**.

For the processed sequence:

```text
307 frames
        ↓
12-frame sliding windows
        ↓
296 temporal sequences
```

The resulting tensor has the shape:

```text
296 × 12 × 512
```

---

### 5. GRU Temporal Encoder

The temporal sequences are passed through a **Gated Recurrent Unit (GRU)**.

The GRU learns relationships between consecutive frames and produces a compact temporal representation.

The resulting representation is approximately:

```text
296 × 128
```

where each temporal window is represented using a **128-dimensional temporal feature vector**.

---

### 6. Relevance Scoring

The system combines semantic similarity and motion information to determine how relevant each temporal window is to the user's query.

The implemented scoring approach is:

```text
Relevance Score =
0.8 × CLIP Similarity
+
0.2 × Motion Score
```

This gives greater importance to semantic similarity while still considering movement within the video.

---

# 🛠️ Technology Stack

| Technology     | Purpose                             |
| -------------- | ----------------------------------- |
| Python         | Core programming language           |
| PyTorch        | GRU and deep learning components    |
| CLIP           | Visual and semantic embeddings      |
| LLM            | Natural-language query analysis     |
| OpenRouter API | LLM access                          |
| OpenCV         | Video/frame processing              |
| NumPy          | Numerical computation               |
| Git & GitHub   | Version control and project hosting |

---

# 📂 Project Structure

```text
VisionSeek-AI/
│
├── query_analyser.py
├── video_search.py
├── temporal_features.py
├── temporal_gru.py
├── temporal_index.py
│
├── README.md
├── requirements.txt
│
└── ...
```

The project is divided into separate modules so that query processing, visual feature extraction, temporal modeling, and retrieval can be maintained independently.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/kaurrrgurleen/VisionSeek-AI.git
```

Navigate into the project directory:

```bash
cd VisionSeek-AI
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure the API Key

VisionSeek-AI uses an LLM for query analysis.

Create an API key through your configured LLM provider and add it to the project environment.

For example:

```text
OPENROUTER_API_KEY=your_api_key_here
```

⚠️ **Never commit your API key to GitHub.**

Use an environment variable or `.env` file instead.

---

# ▶️ Usage

After installing the dependencies and configuring the required API key, run the appropriate project entry point.

The user can then provide a natural-language query describing the desired video event.

Example:

```text
Enter your query:
car turning left
```

The system processes the query and searches the indexed video representations for the most relevant temporal moments.

---

# 🔍 Example Queries

VisionSeek-AI can handle natural-language queries such as:

### Query 1

```text
car turning left
```

The system searches for visual and temporal patterns corresponding to a vehicle performing a left turn.

### Query 2

```text
person walking outdoors
```

The system searches for moments containing a person walking in an outdoor environment.

### Query 3

```text
vehicle near building
```

The system searches for video moments where a vehicle appears in proximity to a building.

---

# 🧪 Technical Implementation

## Query Analysis

The query analysis module uses an LLM to transform an unstructured natural-language query into information useful for video retrieval.

This helps the system distinguish between simple visual queries and queries that require understanding an action or event over time.

---

## Visual Representation

CLIP is used to create semantic representations of video frames.

The implemented pipeline generated:

```text
307 frame embeddings
Embedding dimension = 512
```

Therefore:

```text
307 × 512
```

represents the frame-level feature matrix.

---

## Temporal Representation

The frame embeddings are converted into overlapping temporal windows.

```text
Window size = 12 frames
```

For 307 frames:

```text
307 - 12 + 1 = 296 windows
```

Therefore, the temporal input is:

```text
296 × 12 × 512
```

---

## GRU Processing

The temporal sequences are passed to the GRU:

```text
Input:
296 × 12 × 512

        ↓

GRU

        ↓

Temporal Representation:
296 × 128
```

The GRU allows the system to capture information across consecutive frames rather than treating every frame independently.

---

# 📊 Relevance Scoring

VisionSeek-AI combines semantic similarity and motion information.

The final relevance score is calculated as:

```text
Score = 0.8 × CLIP Similarity + 0.2 × Motion Score
```

### Why combine both?

**CLIP similarity** helps determine whether the visual content matches the user's query.

**Motion score** helps identify temporal activity and movement.

Combining both provides a more useful ranking mechanism for event-based video retrieval.

---

# 📈 Results

The system successfully demonstrates the complete video-search pipeline:

```text
Natural Language Query
        ↓
LLM Query Analysis
        ↓
Visual Feature Extraction
        ↓
Temporal Sequence Construction
        ↓
GRU Temporal Encoding
        ↓
Relevance Scoring
        ↓
Relevant Video Moments
```

The project demonstrates that combining **semantic visual understanding with temporal modeling** can provide a stronger approach to natural-language video search than relying only on individual-frame similarity.

---

# ⚠️ Limitations

Although VisionSeek-AI demonstrates the complete retrieval pipeline, there are several areas that can be improved:

* Retrieval quality depends on the quality of the CLIP representations.
* The current system works with a predefined video/indexing pipeline.
* Complex multi-event queries may require more advanced temporal reasoning.
* The GRU component could benefit from additional training data.
* Large-scale video collections would require more optimized indexing and retrieval.
* The system does not yet provide a complete production-grade web interface.

---

# 🚀 Future Scope

VisionSeek-AI can be extended in several ways:

### 🌐 Web-Based Interface

Develop an interactive web application where users can upload videos and search them using natural language.

### ⚡ Faster Retrieval

Use vector databases such as FAISS or other scalable vector-search systems for large video collections.

### 🤖 Advanced Temporal Models

Replace or complement the GRU with Transformer-based temporal architectures for more complex event understanding.

### 🎯 Better Ranking

Develop a learned ranking model that combines:

* Text-video similarity
* Temporal features
* Motion
* Object detection
* Scene information

### 📹 Multi-Video Search

Extend the system to search across an entire collection of videos rather than a single indexed video.

### 🧠 Improved Query Understanding

Support more complex queries such as:

```text
Find the moment when a person enters the building
after a car stops outside.
```

This would require deeper temporal and relational reasoning.

---

# 🎓 Project Learning Outcomes

Through VisionSeek-AI, the project explores the integration of several AI concepts:

* Natural Language Processing
* Large Language Models
* Computer Vision
* Multimodal AI
* CLIP embeddings
* Representation learning
* Temporal sequence modeling
* GRU networks
* Similarity search
* AI-based information retrieval

The project demonstrates how these individual technologies can be combined into a single end-to-end AI system.

---

# 👩‍💻 Author

**Gurleen Kaur**

AI/ML(Stark AI) Cohort — Final Capstone Project

GitHub:
https://github.com/kaurrrgurleen

---

# 📜 License

This project was developed as an educational AI/ML capstone project.

You may use the code for learning and experimentation with appropriate attribution.
