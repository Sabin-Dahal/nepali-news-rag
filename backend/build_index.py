
import json
import os
import numpy as np
import faiss

EMB_FILE    = "data/embeddings.npy"
MAP_FILE    = "data/chunk_map.json"
INDEX_FILE  = "data/faiss.index"
TOP_K       = 3  


def build_index(embeddings: np.ndarray):
    dim   = embeddings.shape[1]          # 384
    index = faiss.IndexFlatIP(dim)      
    index.add(embeddings)
    return index


def search(index, chunk_map, query_vec, k=TOP_K):
    scores, ids = index.search(query_vec, k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0])):
        chunk = chunk_map[idx]
        results.append({
            "rank":       rank + 1,
            "score":      round(float(score), 4),
            "title":      chunk["title"],
            "url":        chunk["url"],
            "chunk_text": chunk["chunk_text"][:200] + "...",
        })
    return results


def main():
    # ── Load embeddings ──
    if not os.path.exists(EMB_FILE):
        print(f"[ERROR] {EMB_FILE} not found.")
        print("  Run embeddings.py first (Day 5).")
        return

    embeddings = np.load(EMB_FILE)          # shape: (N, 384)
    print(f"\n Loaded embeddings: {embeddings.shape}")

    with open(MAP_FILE, "r", encoding="utf-8") as f:
        chunk_map = json.load(f)

    print(f" Building FAISS index (IndexFlatIP, dim={embeddings.shape[1]})...")
    index = build_index(embeddings)
    print(f"  Index contains {index.ntotal} vectors")

    faiss.write_index(index, INDEX_FILE)
    print(f" Index saved → {INDEX_FILE}")


    #test
    query_vec = embeddings[5:6]            
    results   = search(index, chunk_map, query_vec, k=TOP_K)

    for r in results:
        print(f"\n  #{r['rank']}  score={r['score']}")
        print(f"      title : {r['title'][:60]}")
        print(f"      text  : {r['chunk_text'][:100]}...")

    # ── Summary ──
    print(f"\n{'─'*45}")
    print(f"  Vectors in index  : {index.ntotal}")
    print(f"  Dimensions        : {embeddings.shape[1]}")
    index_size = os.path.getsize(INDEX_FILE) // 1024
    print(f"  Index size        : {index_size} KB")
    print(f"{'─'*45}\n")


if __name__ == "__main__":
    main()