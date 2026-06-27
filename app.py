#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:53:04 2026

@author: kristenrose
"""

import streamlit as st
from pypdf import PdfReader
import chromadb
import anthropic
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]

# --- setup: database + Claude client ---
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("documents")
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def index_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)        # pypdf can read the uploaded file directly
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    chunks = chunk_text(text)

    existing_ids = collection.get()["ids"]   # clear the previously indexed document
    if existing_ids:
        collection.delete(ids=existing_ids)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.upsert(documents=chunks, ids=ids)


def answer_question(question):
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n".join(results["documents"][0])
    prompt = f"""Answer the question using only the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}"""
    message = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# --- the web page ---
st.title("📄 Ask Your PDF")
st.write("Upload a PDF, then ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # only re-index when a NEW file is uploaded, not on every interaction
    if st.session_state.get("indexed_file") != uploaded_file.name:
        with st.spinner("Reading and indexing your document..."):
            index_pdf(uploaded_file)
        st.session_state["indexed_file"] = uploaded_file.name
        st.success("Document ready — ask away!")

    question = st.text_input("Your question:")
    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(question)
        st.markdown(answer)