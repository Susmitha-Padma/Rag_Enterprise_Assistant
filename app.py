import os
import streamlit as st
from rank_bm25 import BM25Okapi

from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
DATA_PATH = "data"
VECTOR_DB_PATH = "vectorstore"

st.set_page_config(page_title="RAG CHATBOT", layout="wide")
st.title("🚀 RAG-ENTERPRISE-ASSISTANT")

# ---------------- LOAD DOCUMENTS ----------------
def load_docs():
    docs = []
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())
    return docs


def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)

    # BM25 corpus
    corpus = [c.page_content for c in chunks]
    tokenized = [text.split() for text in corpus]
    bm25 = BM25Okapi(tokenized)

    return db, chunks, bm25


def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

# ---------------- MODEL (LOW RAM SAFE) ----------------
# IMPORTANT: Use only tinyllama or phi3 on 4 GB RAM
llm = OllamaLLM(
    model="tinyllama",
    temperature=0,
    num_ctx=1024
)

# ---------------- HYBRID RETRIEVER (NO RERANKER, NO AGENTS) ----------------
def hybrid_retrieve(query):
    # Vector search
    vectorstore = load_vectorstore()
    vector_docs = vectorstore.similarity_search(query, k=2)

    # BM25 search
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    corpus = [c.page_content for c in chunks]
    bm25 = BM25Okapi([c.split() for c in corpus])
    bm25_docs = [chunks[i] for i in bm25.get_top_n(query.split(), range(len(chunks)), n=2)]

    # Combine and deduplicate
    combined = list({id(d): d for d in vector_docs + bm25_docs}.values())
    return combined


# ---------------- ANSWER GENERATION ----------------
def answer_question(query, docs):
    context_text = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question:
{query}
"""

    answer = llm.invoke(prompt)
    return answer

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Document Indexing")

if st.sidebar.button("📥 Index Documents"):
    with st.spinner("Building hybrid index..."):
        docs = load_docs()
        if not docs:
            st.sidebar.error("No PDFs found in data folder")
        else:
            build_vectorstore(docs)
            st.sidebar.success("Hybrid index created successfully!")

# ---------------- MAIN ----------------
if os.path.exists(VECTOR_DB_PATH):
    query = st.text_input("💬 Ask a question")

    if query:
        with st.spinner("Searching and answering..."):
            docs = hybrid_retrieve(query)
            answer = answer_question(query, docs)

            st.subheader("✅ Answer")
            st.write(answer)

            st.subheader("📌 Sources")
            for d in docs:
                src = d.metadata.get("source", "Unknown")
                st.markdown(f"- {src}")
else:
    st.warning("Please index documents first using the sidebar.")
