from ultralytics import YOLO
import cv2
import json
from pathlib import Path

def run_yolo8_inference(model_path, video_path, output_json, class_id=2):
    model = YOLO(model_path)
    results = model.predict(video_path, stream=True, conf=0.25, classes=class_id)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = -1
    detections = []

    for result in results:
        frame_id += 1
        if result.boxes is None: continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append({
                "frame": frame_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })

    with open(output_json, "w") as f:
        json.dump(detections, f, indent=2)
    print(f"✅ Saved {len(detections)} detections to {output_json}")

if __name__ == "__main__":
    base = Path("data")
    out = Path("output")
    run_yolo8_inference(base / "yolo_model.pt", base / "broadcast.mp4", out / "detections_broadcast.json")
    run_yolo8_inference(base / "yolo_model.pt", base / "tacticam.mp4", out / "detections_tacticam.json")
