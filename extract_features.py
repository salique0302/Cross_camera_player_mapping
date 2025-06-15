import json
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from tqdm import tqdm
from pathlib import Path

def load_detections(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def extract_features(video_path, detections, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(pretrained=True).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(str(video_path))
    frame_cache = {}
    features = []

    for det in tqdm(detections, desc=f"Extracting features from {video_path.name}"):
        frame_id = det["frame"]
        bbox = list(map(int, det["bbox"]))

        if frame_id not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_cache[frame_id] = frame
        else:
            frame = frame_cache[frame_id]

        H, W, _ = frame.shape
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        try:
            input_tensor = transform(crop).unsqueeze(0).to(device)
        except:
            continue

        with torch.no_grad():
            embedding = model(input_tensor).cpu().squeeze().tolist()

        features.append({
            "frame": frame_id,
            "bbox": [x1, y1, x2, y2],
            "embedding": embedding
        })

    cap.release()

    with open(output_path, "w") as f:
        json.dump(features, f, indent=2)
    print(f"✅ Saved {len(features)} embeddings to {output_path}")

if __name__ == "__main__":
    base = Path("data")
    out = Path("output")
    extract_features(
        base / "broadcast.mp4",
        load_detections(out / "detections_broadcast.json"),
        out / "embeddings_broadcast.json"
    )
    extract_features(
        base / "tacticam.mp4",
        load_detections(out / "detections_tacticam.json"),
        out / "embeddings_tacticam.json"
    )
