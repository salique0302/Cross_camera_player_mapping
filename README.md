# 📌 Cross-Camera Player Re-Identification


## 🧾 Project Overview

This project solves the problem of player re-identification across two distinct camera views (broadcast and tacticam) of the same soccer play. The objective is to ensure that the same player retains the same identity across both feeds, even with changes in viewpoint, occlusion, or motion blur. The pipeline involves object detection, deep embedding extraction, similarity-based matching, and visual validation.
---

## 🔍 Project Workflow

1. **Detection:** Detect players from each frame using the provided YOLOv8 model.
2. **Feature Extraction:** Crop player bounding boxes and generate visual embeddings using ResNet50.
3. **Re-Identification:** Match players across camera feeds using cosine similarity.
4. **Visualization:** Render annotated videos and a side-by-side demo of matched player pairs.


## 📦 Requirements

- Python 3.11+
- OpenCV
- PyTorch
- torchvision
- tqdm
- numpy


## 🛠️ Installation & Setup

bash
# Clone the repo
git clone https://github.com/salique0302/Cross_camera_player_mapping.git
cd Cross_camera_player_mapping

# (Optional) Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # or use venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

##📥 Required Files
Videos:

data/broadcast.mp4

data/tacticam.mp4

Model:

yolo_model.pt (provided in assignment)
(Not included in repo due to size limits. Download it here)

##🚀 How to Run
bash
Copy
Edit
# Step 1: Run detection
python detect_yolo8.py

# Step 2: Generate ResNet embeddings
python extract_features.py

# Step 3: Match players across videos
python match_players.py

# Step 4: Create visual demo
python generate_mapping_demo.py

