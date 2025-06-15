import cv2
import json
import numpy as np

def generate_mapping_demo(
    broadcast_path, tacticam_path, matches_path, output_path, max_pairs=10
):
    # Load matches
    with open(matches_path, "r") as f:
        matches = json.load(f)

    # Use only top-N matches by similarity
    matches = sorted(matches, key=lambda x: -x["similarity"])[:max_pairs]

    # Open both videos
    cap_b = cv2.VideoCapture(broadcast_path)
    cap_t = cv2.VideoCapture(tacticam_path)

    width = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_b.get(cv2.CAP_PROP_FPS)
    output_size = (width * 2, height)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)

    for match in matches:
        b_frame_idx = match["broadcast"]["frame"]
        t_frame_idx = match["tacticam"]["frame"]
        b_bbox = list(map(int, match["broadcast"]["bbox"]))
        t_bbox = list(map(int, match["tacticam"]["bbox"]))
        pid = match["player_id"]
        sim = match["similarity"]

        # Read broadcast frame
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, b_frame_idx)
        ret_b, frame_b = cap_b.read()

        # Read tacticam frame
        cap_t.set(cv2.CAP_PROP_POS_FRAMES, t_frame_idx)
        ret_t, frame_t = cap_t.read()

        if not (ret_b and ret_t):
            continue

        # Annotate broadcast
        cv2.rectangle(frame_b, (b_bbox[0], b_bbox[1]), (b_bbox[2], b_bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame_b, f"ID {pid}", (b_bbox[0], b_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Annotate tacticam
        cv2.rectangle(frame_t, (t_bbox[0], t_bbox[1]), (t_bbox[2], t_bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame_t, f"ID {pid}", (t_bbox[0], t_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add similarity text on broadcast side
        sim_text = f"Similarity: {sim:.2f}"
        cv2.putText(frame_b, sim_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Combine both frames side-by-side
        combined = np.hstack((frame_b, frame_t))

        # Repeat the combined frame for ~1.5 seconds to let viewer see clearly
        for _ in range(int(fps * 1.5)):
            out.write(combined)

    cap_b.release()
    cap_t.release()
    out.release()
    print(f"✅ Mapping demo saved to {output_path}")


if __name__ == "__main__":
    generate_mapping_demo(
        broadcast_path="data/broadcast.mp4",
        tacticam_path="data/tacticam.mp4",
        matches_path="output/matched_players.json",
        output_path="visuals/mapping_demo.mp4",
        max_pairs=10
    )
