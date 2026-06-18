import os
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNKS_PATH, INDEX_PATH, RERANK_MIN_SCORE, TOP_N_RELEVANT, embeddings

PDF_PATHS = [
    "./data/250153.pdf",
    "./data/d2l-en.pdf",
    "./data/Building Reliable AI Systems (MEAP, all 11 chapters) (Rush Shahani) (Z-Library).pdf",
    "./data/Hands-On Machine Learning with Scikit-Learn and PyTorch Concepts, Tools, and Techniques to Build Intelligent Systems (Aurélien Géron) (Z-Library).pdf",
]


def load_and_chunk_docs() -> List[Document]:
    docs: List[Document] = []
    for pdf_path in PDF_PATHS:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Missing required PDF: {pdf_path}")
        docs.extend(PyPDFLoader(pdf_path).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    return splitter.split_documents(docs)


def load_or_build_vector_store() -> Tuple[List[Document], FAISS]:
    index_path = Path(INDEX_PATH)
    chunks_path = Path(CHUNKS_PATH)

    if index_path.exists():
        print(f"Loading cached FAISS index from {INDEX_PATH} ...")
        vector_store = FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )

        if chunks_path.exists():
            print(f"Loading cached chunks from {CHUNKS_PATH} ...")
            with chunks_path.open("rb") as f:
                chunks = pickle.load(f)
        else:
            print("Chunks cache missing - re-chunking documents for BM25 (no re-embedding needed)...")
            chunks = load_and_chunk_docs()
            with chunks_path.open("wb") as f:
                pickle.dump(chunks, f)
    else:
        print("No cached index found - loading, chunking, and embedding documents (first run, several minutes)...")
        chunks = load_and_chunk_docs()
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(INDEX_PATH)
        with chunks_path.open("wb") as f:
            pickle.dump(chunks, f)
        print(f"Cached {len(chunks)} chunks + FAISS index for fast startup next time.")

    print(f"Ready: {len(chunks)} chunks loaded.")
    return chunks, vector_store


chunks, vector_store = load_or_build_vector_store()

dense_retriever = vector_store.as_retriever(search_kwargs={"k": 8})

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 8

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6],
)

try:
    from sentence_transformers import CrossEncoder

    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_AVAILABLE = True
    print("Cross-encoder reranker loaded (cross-encoder/ms-marco-MiniLM-L-6-v2)")
except Exception as e:
    reranker_model = None
    RERANKER_AVAILABLE = False
    print(f"Cross-encoder reranker unavailable ({e}) - falling back to unranked top-N docs.")


def retrieve_documents(search_texts: Sequence[str]) -> List[Document]:
    seen = set()
    merged_docs: List[Document] = []

    for text in search_texts:
        for doc in retriever.invoke(text):
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                merged_docs.append(doc)

    return merged_docs


def rerank_documents(question: str, docs: Sequence[Document]) -> List[Document]:
    if not docs:
        return []

    if not RERANKER_AVAILABLE:
        return list(docs)[:TOP_N_RELEVANT]

    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker_model.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    print("  rerank top scores:", [round(float(score), 2) for _, score in scored_docs[:5]])

    relevant_docs = [doc for doc, score in scored_docs if score >= RERANK_MIN_SCORE][:TOP_N_RELEVANT]
    return relevant_docs
