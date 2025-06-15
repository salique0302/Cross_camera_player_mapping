import cv2
import json
import os

def save_sample_detections(video_path, detections_json, out_img_path, frame_num=10):
    # Load detection results
    with open(detections_json, "r") as f:
        detections = json.load(f)

    # Filter detections for the target frame
    frame_dets = [d for d in detections if d["frame"] == frame_num]

    # Load video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Could not read frame {frame_num} from {video_path}")
        return

    # Draw bounding boxes
    for det in frame_dets:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save output image
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    cv2.imwrite(out_img_path, frame)
    print(f"✅ Saved visual to {out_img_path}")

if __name__ == "__main__":
    base = "/Users/mdsalique/Desktop/track_player"
    
    save_sample_detections(
        video_path=f"{base}/data/broadcast.mp4",
        detections_json=f"{base}/output/detections_broadcast.json",
        out_img_path=f"{base}/visuals/detections_broadcast_frame10.jpg",
        frame_num=10
    )

    save_sample_detections(
        video_path=f"{base}/data/tacticam.mp4",
        detections_json=f"{base}/output/detections_tacticam.json",
        out_img_path=f"{base}/visuals/detections_tacticam_frame10.jpg",
        frame_num=10
    )
