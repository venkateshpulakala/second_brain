from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import ingest_text, ingest_file, ingest_folder

app = FastAPI()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

index_path = "vector_db/index.faiss"
meta_path = "vector_db/meta.pkl"


# 🔹 Request Models
class QueryRequest(BaseModel):
    query: str


class FileRequest(BaseModel):
    file_path: str


class FolderRequest(BaseModel):
    folder_path: str


# 🔹 Search function
def search(query, top_k=3):

    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    query_vector = model.encode(query)
    D, I = index.search(np.array([query_vector]), top_k)

    results = [meta[i] for i in I[0] if i < len(meta)]

    return results


# 🔹 Ask API (RAG style)
@app.post("/ask")
def ask_question(data: QueryRequest):
    try:
        results = search(data.query)

        if not results:
            return {"answer": "No relevant information found."}

        context = " ".join(results)

        # Simple answer generation (no LLM dependency fallback)
        answer = f"Based on stored knowledge: {context}"

        return {
            "answer": answer,
            "sources": results
        }

    except Exception as e:
        return {"error": str(e)}


# 🔹 Upload single file
@app.post("/upload")
def upload_file(data: FileRequest):
    try:
        result = ingest_file(data.file_path)
        return {"status": result}
    except Exception as e:
        return {"error": str(e)}


# 🔹 Upload folder
@app.post("/upload-folder")
def upload_folder(data: FolderRequest):
    try:
        result = ingest_folder(data.folder_path)
        return {"status": result}
    except Exception as e:
        return {"error": str(e)}


# 🔹 Direct text input (optional)
@app.post("/add-text")
def add_text(data: QueryRequest):
    try:
        result = ingest_text(data.query)
        return {"status": result}
    except Exception as e:
        return {"error": str(e)}