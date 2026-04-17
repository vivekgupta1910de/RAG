"""Basic Retrieval-Augmented Generation (RAG) pipeline.

Run with:
    python src/app.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


DATA_DIR = Path("data")
TOP_K = 3
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


@dataclass
class Document:
    """Simple in-memory document representation."""

    page_content: str
    metadata: dict


class RecursiveCharacterTextSplitter:
    """A lightweight recursive character splitter.

    This mirrors the behavior commonly used in LangChain:
    - recursively split text by separators
    - enforce chunk size and overlap
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Sequence[str] = ("\n\n", "\n", " ", ""),
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split all documents and preserve metadata for each resulting chunk."""
        chunks: List[Document] = []
        for doc in documents:
            for chunk_text in self._split_text(doc.page_content):
                chunks.append(Document(page_content=chunk_text, metadata=dict(doc.metadata)))
        return chunks

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        pieces = self._split_recursive(text, self.separators)

        # Merge pieces into chunk_size windows with overlap.
        merged: List[str] = []
        current = ""

        for piece in pieces:
            candidate = f"{current}{piece}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current.strip():
                merged.append(current.strip())

            # Start the next chunk with overlap from previous chunk.
            overlap = current[-self.chunk_overlap :] if current else ""
            current = f"{overlap}{piece}"

            # Handle very large single piece.
            while len(current) > self.chunk_size:
                merged.append(current[: self.chunk_size].strip())
                overlap = current[self.chunk_size - self.chunk_overlap : self.chunk_size]
                current = f"{overlap}{current[self.chunk_size:]}"

        if current.strip():
            merged.append(current.strip())

        return merged

    def _split_recursive(self, text: str, separators: Sequence[str]) -> List[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text]

        separator = separators[0]

        if separator == "":
            # Character-level fallback split.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        if separator not in text:
            return self._split_recursive(text, separators[1:])

        parts = text.split(separator)
        output: List[str] = []

        for idx, part in enumerate(parts):
            fragment = part if idx == len(parts) - 1 else f"{part}{separator}"
            if len(fragment) > self.chunk_size:
                output.extend(self._split_recursive(fragment, separators[1:]))
            else:
                output.append(fragment)

        return output


class OpenAIEmbeddings:
    """Minimal OpenAI embeddings wrapper."""

    def __init__(self, model: str = EMBEDDING_MODEL) -> None:
        self.model = model
        self.client = OpenAI()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


class FAISSVectorStore:
    """FAISS-backed vector store for document chunks."""

    def __init__(self, index: faiss.IndexFlatL2, documents: List[Document], vectors: np.ndarray) -> None:
        self.index = index
        self.documents = documents
        self.vectors = vectors

    @classmethod
    def from_documents(cls, chunks: List[Document], embeddings: OpenAIEmbeddings) -> "FAISSVectorStore":
        texts = [doc.page_content for doc in chunks]
        vectors = np.array(embeddings.embed_documents(texts), dtype="float32")

        if vectors.size == 0:
            raise ValueError("No chunk embeddings generated.")

        # L2 index over normalized vectors for cosine-like retrieval behavior.
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        return cls(index=index, documents=chunks, vectors=vectors)

    def similarity_search(self, query: str, embeddings: OpenAIEmbeddings, k: int = TOP_K) -> List[Document]:
        query_vector = np.array([embeddings.embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_vector)

        _, indices = self.index.search(query_vector, k)

        results: List[Document] = []
        for idx in indices[0]:
            if idx == -1:
                continue
            results.append(self.documents[idx])

        return results


def load_pdf_documents(data_dir: Path) -> List[Document]:
    """Load all PDF documents from the given directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in directory: {data_dir.resolve()}")

    documents: List[Document] = []

    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_num,
                    },
                )
            )

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: List[Document], embeddings: OpenAIEmbeddings) -> FAISSVectorStore:
    """Generate embeddings and store them in a FAISS index."""
    return FAISSVectorStore.from_documents(chunks, embeddings)


def get_relevant_chunks(
    query: str,
    vector_store: FAISSVectorStore,
    embeddings: OpenAIEmbeddings,
    top_k: int = TOP_K,
) -> List[Document]:
    """Retrieve top-k chunks most relevant to the query."""
    return vector_store.similarity_search(query, embeddings, k=top_k)


def generate_answer(query: str, context_docs: List[Document], client: OpenAI) -> str:
    """Generate an answer from retrieved context using an OpenAI chat model."""
    context = "\n\n".join(doc.page_content for doc in context_docs)

    system_prompt = (
        "You are a helpful assistant that answers questions using provided context. "
        "If the answer is not in the context, say you do not know."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer concisely and cite relevant details from the context."
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content or "No answer generated."


def query_rag_pipeline(
    query: str,
    vector_store: FAISSVectorStore,
    embeddings: OpenAIEmbeddings,
    client: OpenAI,
) -> tuple[str, List[Document]]:
    """Run retrieval + generation for one query."""
    top_chunks = get_relevant_chunks(query, vector_store, embeddings, top_k=TOP_K)
    answer = generate_answer(query, top_chunks, client)
    return answer, top_chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Basic RAG pipeline over local PDFs")
    parser.add_argument("--query", type=str, help="Single query to answer")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load environment variables from .env (e.g., OPENAI_API_KEY).
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")

    print("Loading PDF documents...")
    documents = load_pdf_documents(DATA_DIR)

    print("Splitting documents into chunks...")
    chunks = split_documents(documents)

    print("Building FAISS vector store (embedding chunks)...")
    embeddings = OpenAIEmbeddings()
    vector_store = build_vector_store(chunks, embeddings)

    openai_client = OpenAI()

    if args.query:
        answer, _ = query_rag_pipeline(args.query, vector_store, embeddings, openai_client)
        print("\nAnswer:\n")
        print(answer)
        return

    print("\nRAG pipeline ready. Type a question (or 'exit' to quit).\n")
    while True:
        user_query = input("> ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_query:
            continue

        answer, retrieved_docs = query_rag_pipeline(user_query, vector_store, embeddings, openai_client)

        print("\nAnswer:\n")
        print(answer)
        print("\nTop retrieved sources:")
        for idx, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "n/a")
            print(f"{idx}. {source} (page {page})")
        print()


if __name__ == "__main__":
    main()
