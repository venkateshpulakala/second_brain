import streamlit as st
import requests

st.set_page_config(page_title="SecondBrain", layout="centered")

st.title("🧠 SecondBrain")
st.write("Ask questions from your personal knowledge base")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"query": query}
        )

        data = response.json()

        st.subheader("Answer")
        st.write(data.get("answer"))

        st.subheader("Sources")
        for src in data.get("sources", []):
            st.write("•", src)
    else:
        st.warning("Please enter a question")