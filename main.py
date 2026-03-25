from fastapi import FastAPI
from ingest import ingest_text
from search import search
from rag import generate_answer

app = FastAPI(title="SecondBrain API")


# 🟢 Home route
@app.get("/")
def home():
    return {"message": "SecondBrain Running"}


# 🟢 Ingest data into knowledge base
@app.post("/ingest")
def ingest_api(data: dict):
    text = data["text"]
    ingest_text(text)
    return {"status": "stored"}


# 🟢 Semantic search (raw retrieval)
@app.post("/search")
def search_api(data: dict):
    query = data["query"]
    results = search(query)
    return {"results": results}


# 🟢 RAG (final intelligent answer)
@app.post("/ask")
def ask_api(data: dict):
    query = data["query"]
    result = generate_answer(query)
    return result