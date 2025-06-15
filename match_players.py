import json
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

def load_embeddings(path):
    with open(path, "r") as f:
        return json.load(f)

def match_players(broadcast_embeddings, tacticam_embeddings, threshold=0.85):
    broadcast_vectors = [e["embedding"] for e in broadcast_embeddings]
    tacticam_vectors = [e["embedding"] for e in tacticam_embeddings]

    sim_matrix = cosine_similarity(tacticam_vectors, broadcast_vectors)
    matches = []
    used_broadcast_ids = set()

    for i, row in enumerate(sim_matrix):
        best_idx = row.argmax()
        best_score = row[best_idx]

        # Avoid duplicate broadcast matches
        if best_score >= threshold and best_idx not in used_broadcast_ids:
            used_broadcast_ids.add(best_idx)
            matches.append({
                "player_id": len(matches) + 1,
                "broadcast": {
                    "frame": broadcast_embeddings[best_idx]["frame"],
                    "bbox": broadcast_embeddings[best_idx]["bbox"]
                },
                "tacticam": {
                    "frame": tacticam_embeddings[i]["frame"],
                    "bbox": tacticam_embeddings[i]["bbox"]
                },
                "similarity": float(best_score)
            })

    return matches

if __name__ == "__main__":
    out = Path("output")
    broadcast = load_embeddings(out / "embeddings_broadcast.json")
    tacticam = load_embeddings(out / "embeddings_tacticam.json")

    matches = match_players(broadcast, tacticam, threshold=0.85)

    with open(out / "matched_players.json", "w") as f:
        json.dump(matches, f, indent=2)

    print(f"✅ Saved {len(matches)} matched player ID mappings to matched_players.json")
