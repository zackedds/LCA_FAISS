
# search_faiss.py
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

PKL_PATH = Path("faiss_index.pkl")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

def load_index(pkl_path: Path = PKL_PATH):
    """Load FAISS index + metadata dict list from pickle."""
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    index: faiss.Index = payload["index"]
    metadata: list[dict] = payload["metadata"]
    return index, metadata

# initialise once (re‑use across queries)
print("Loading embedding model…")
model = SentenceTransformer(EMBED_MODEL)
print("Loading FAISS index…")
index, metadata = load_index()

def search(query: str, k: int = TOP_K):
    """Embed query, run FAISS search, and return list[dict] results."""
    q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_vec, k)
    hits = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue  # no more hits
        doc = metadata[idx].copy()
        doc["distance"] = float(dist)
        hits.append(doc)
    return hits

# -------------- demo / CLI --------------
def demo_search():
    query = "What are the current treatment guidelines for X?"
    print(f"\nDEMO QUERY: {query!r}\n")
    for rank, doc in enumerate(search(query), 1):
        print(f"#{rank}  (L2={doc['distance']:.4f})  {doc['title']}")
        print(f"   → {doc['text'][:120]}…\n")

def main():
    # --- interactive version ---
    # while True:
    #     q = input("\nAsk anything (or 'exit'): ").strip()
    #     if q.lower() in {"exit", "quit"}:
    #         break
    #     for rank, doc in enumerate(search(q), 1):
    #         print(f"\n#{rank}  (L2={doc['distance']:.4f})  {doc['title']}")
    #         print(f"   → {doc['text'][:160]}…")

    # --- non‑interactive quick test ---
    demo_search()

if __name__ == "__main__":
    main()