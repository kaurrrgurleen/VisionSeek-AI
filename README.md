# 🔎 VisionSeek AI

### AI-Powered Semantic & Temporal Video Search Engine

VisionSeek AI is an intelligent video search system that allows users to find specific moments inside a video using natural-language queries.

Instead of manually watching an entire video, users can search for events such as:

- "Find the moment where a car turns left on the road"
- "Show me a person walking outdoors"
- "Find a vehicle near a building"

VisionSeek understands the query, searches the visual content of the video, and returns the most relevant moments with timestamps and frames.

---

## 🚀 Key Features

- 🔍 Natural-language video search
- 🧠 LLM-based query understanding
- 🖼️ CLIP-based semantic image-text matching
- ⏱️ Temporal activity analysis
- 🤖 GRU-based temporal representation
- 📊 Hybrid CLIP + temporal ranking
- 🎯 Top matching video moments with timestamps
- 🌐 Interactive Streamlit web interface

---

## 🧠 How VisionSeek Works

```text
User Query
    ↓
LLM Query Analysis
    ↓
Structured Search Intent
    ↓
CLIP Text Embedding
    ↓
Frame Embedding Similarity
    ↓
Temporal Activity Analysis
    ↓
Hybrid Ranking
    ↓
Top Matching Frames

For temporal queries, VisionSeek combines:

Final Score =
0.8 × CLIP Similarity
+
0.2 × Temporal Activity

For non-temporal queries, the system uses CLIP semantic similarity directly.

🛠️ Technologies Used
Python
Streamlit
PyTorch
Hugging Face Transformers
CLIP
NumPy
PIL
LLM-based Query Analysis
📁 Project Structure
VisionSeek-AI/
│
├── Data/
│
├── results/
│   ├── frame_embeddings.npy
│   ├── frame_metadata.json
│   ├── temporal_activity.json
│   ├── temporal_embeddings.npy
│   ├── temporal_gru.pth
│   ├── temporal_index.json
│   └── temporal_sequences.npy
│
├── src/
│   ├── app.py
│   ├── clip_embeddings.py
│   ├── clip_encoder.py
│   ├── frame_extractor.py
│   ├── query_analyser.py
│   ├── temporal_activity.py
│   ├── temporal_features.py
│   ├── temporal_gru.py
│   ├── temporal_index.py
│   └── video_search.py
│
├── .gitignore
├── app.py
├── requirements.txt
└── README.md
⚙️ Installation

Clone the repository:

git clone https://github.com/kaurrrgurleen/VisionSeek-AI.git
cd VisionSeek-AI

Install dependencies:

pip install -r requirements.txt
▶️ Run the Application

Run the Streamlit application using:

python -m streamlit run src/app.py

The application will open in your browser at:

http://localhost:8501
🔎 Example Queries

Try queries such as:

Find the moment where a car turns left on the road
Show me a person walking outdoors
Find a vehicle near a building

The system returns relevant frames along with:

Timestamp
Frame
CLIP similarity score
Temporal activity score
Final ranking score
🎯 Project Outcome

VisionSeek demonstrates how multimodal AI, semantic search, and temporal video understanding can be combined to create an intelligent video retrieval system.

It reduces the need for manual video browsing by allowing users to search video content using natural language.

👩‍💻 Author

Gurleen Kaur

AI/ML Student | Building projects in Artificial Intelligence, Machine Learning and Computer Vision