import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_FILE = "vector_db/index.faiss"
META_FILE = "vector_db/meta.pkl"

def search(query, k=3):
    index = faiss.read_index(INDEX_FILE)
    meta = pickle.load(open(META_FILE, "rb"))

    q_emb = model.encode([query])
    distances, indices = index.search(q_emb, k)

    results = []
    for i in indices[0]:
        if i < len(meta):
            results.append(meta[i])

    return results