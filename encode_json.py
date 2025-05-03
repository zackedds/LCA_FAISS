import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# Load data
with open("pdf_doc.json", "r") as f:
    docs = json.load(f)

# Extract texts
texts = [doc['text'] for doc in docs]

# Encode
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

# FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.float32(embeddings))

# Save
with open("faiss_index.pkl", "wb") as f:
    pickle.dump({
        "index": index,
        "metadata": docs
    }, f)