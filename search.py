import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = None
def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

index_path = "vector_db/index.faiss"
meta_path = "vector_db/meta.pkl"


def search(query, k=3):
    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    query_vector = get_model().encode([query])

    distances, indices = index.search(np.array(query_vector), k)

    results = []
    seen = set()

    for i in indices[0]:
        if i < len(meta):
            text = meta[i]
            if text not in seen:
                results.append(text)
                seen.add(text)

    return results