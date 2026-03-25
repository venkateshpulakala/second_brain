import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

DB_PATH = "vector_db"
INDEX_FILE = f"{DB_PATH}/index.faiss"
META_FILE = f"{DB_PATH}/meta.pkl"

os.makedirs(DB_PATH, exist_ok=True)

def load_index():
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    return faiss.IndexFlatL2(384)

def ingest_text(text):
    index = load_index()

    embedding = model.encode([text])
    index.add(embedding)

    faiss.write_index(index, INDEX_FILE)

    meta = []
    if os.path.exists(META_FILE):
        meta = pickle.load(open(META_FILE, "rb"))

    meta.append(text)
    pickle.dump(meta, open(META_FILE, "wb"))