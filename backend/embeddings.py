import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME   = "paraphrase-multilingual-mpnet-base-v2"  
INPUT_FILE   = "data/chunked_data.json"
EMB_FILE     = "data/embeddings.npy"
MAP_FILE     = "data/chunk_map.json"
BATCH_SIZE   = 32


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] {INPUT_FILE} not found.")
        print("  Make sure you ran chunker.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\n Loaded {len(chunks)} chunks from {INPUT_FILE}")

    print(f"\n Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("  Model loaded.\n")

    texts = [chunk["chunk_text"] for chunk in chunks]
    print(f" Embedding {len(texts)} chunks (batch_size={BATCH_SIZE})...")
    print("  This may take a few minutes\n")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype("float32")

    os.makedirs("data", exist_ok=True)
    np.save(EMB_FILE, embeddings)
    print(f"\n Embeddings saved → {EMB_FILE}")
    print(f"  Shape: {embeddings.shape}  (rows=chunks, cols=dimensions)")


    chunk_map = [
        {
            "chunk_id":   chunk["chunk_id"],
            "title":      chunk["title"],
            "url":        chunk["url"],
            "author":     chunk.get("author", ""),
            "chunk_text": chunk["chunk_text"],
        }
        for chunk in chunks
    ]
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)
    print(f" Chunk map saved  → {MAP_FILE}")

    print(f"\n{'─'*45}")
    print(f"  Chunks embedded   : {len(chunks)}")
    print(f"  Embedding shape   : {embeddings.shape}")
    print(f"  Dimensions        : {embeddings.shape[1]}")
    print(f"  Size on disk      : ~{embeddings.nbytes // 1024} KB")
    print(f"{'─'*45}\n")



if __name__ == "__main__":
    main()