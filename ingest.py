import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

# Load embedding model lazily
model = None
def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

# Create vector DB folder
if not os.path.exists("vector_db"):
    os.makedirs("vector_db")

index_path = "vector_db/index.faiss"
meta_path = "vector_db/meta.pkl"

# Initialize FAISS index
if not os.path.exists(index_path):
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump([], f)


# 🔹 FINAL FIXED ingest_text
def ingest_text(text):

    if not text or not isinstance(text, str):
        return "Invalid text input"

    # 🔥 Clean text
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = text.strip()

    if text == "":
        return "Empty text"

    # 🔥 Chunking (safe size)
    chunk_size = 300
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    for chunk in chunks:
        chunk = chunk.strip()

        if len(chunk) < 5:
            continue

        try:
            # 🔥 KEY FIX
            embedding = get_model().encode(chunk)
            index.add(np.array([embedding]))
            meta.append(chunk)
        except:
            continue  # skip bad chunks

    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    return "Text ingested successfully"


# 🔹 FILE ingestion
def ingest_file(file_path):

    if not os.path.exists(file_path):
        return "File not found"

    text = ""

    # PDF
    if file_path.endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text and isinstance(page_text, str):
                    text += page_text + " "
        except Exception as e:
            return f"PDF read error: {str(e)}"

    # TXT
    elif file_path.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            return f"TXT read error: {str(e)}"

    # DOCX
    elif file_path.endswith(".docx"):
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                if para.text:
                    text += para.text + " "
        except Exception as e:
            return f"DOCX read error: {str(e)}"

    else:
        return "Unsupported file type"

    # Final check
    if not text or text.strip() == "":
        return "File has no readable content"

    return ingest_text(text)


def ingest_folder(folder_path):

    if not os.path.exists(folder_path):
        return "Folder not found"

    results = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # skip folders
        if os.path.isdir(file_path):
            continue

        result = ingest_file(file_path)
        results.append({file_name: result})

    return results