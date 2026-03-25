from search import search
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_answer(query):
    try:
        results = search(query)
        context = "\n".join(results)

        if not context:
            return {"answer": "No relevant information found.", "sources": []}

        prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": results
        }

    except Exception as e:
        return {"error": str(e)}