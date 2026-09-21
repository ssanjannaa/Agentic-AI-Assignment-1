"""
Task 4: RAG-Based Question Answering (5 Marks)
================================================
A basic Retrieval-Augmented Generation app:
    1. Load a PDF or TXT document.
    2. Split it into chunks.
    3. Index the chunks with TF-IDF (no external embedding API required).
    4. For a user query, retrieve the most relevant chunk(s) via cosine
       similarity.
    5. Ask the LLM to answer the query using ONLY the retrieved chunks.

Usage
-----
    python rag_qa.py --doc sample_docs/company_policy.txt --query "What is the refund policy?"

    # interactive mode
    python rag_qa.py --doc sample_docs/company_policy.txt --interactive
"""

import os
import sys
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.llm_client import LLMClient  # noqa: E402


# ---------------------------------------------------------------------
def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .pdf or .txt")


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100):
    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------
class RAGQAApp:
    def __init__(self, doc_path: str, chunk_size: int = 700, overlap: int = 100, top_k: int = 3):
        self.llm = LLMClient()
        self.top_k = top_k

        text = load_document(doc_path)
        if not text.strip():
            raise ValueError(f"No text could be extracted from '{doc_path}'.")

        self.chunks = chunk_text(text, chunk_size, overlap)
        print(f"[RAGQAApp] Loaded '{os.path.basename(doc_path)}' -> {len(self.chunks)} chunk(s).")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, top_k: int = None):
        top_k = top_k or self.top_k
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.chunk_matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [
            {"text": self.chunks[i], "score": float(sims[i])}
            for i in ranked_idx if sims[i] > 0
        ]

    def answer(self, query: str) -> dict:
        retrieved = self.retrieve(query)
        if not retrieved:
            return {"answer": "No relevant information found in the document.", "context_used": []}

        context = "\n\n---\n\n".join(r["text"] for r in retrieved)
        system_prompt = (
            "You are a document Q&A assistant. Answer the question using ONLY "
            "the provided context. If the answer isn't in the context, say so "
            "clearly instead of guessing."
        )
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
        answer_text = self.llm.generate(system_prompt, user_prompt)
        return {"answer": answer_text, "context_used": retrieved}


# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Task 4: RAG-Based Question Answering")
    parser.add_argument("--doc", required=True, help="Path to a .pdf or .txt document")
    parser.add_argument("--query", help="A single question to ask about the document")
    parser.add_argument("--interactive", action="store_true", help="Start interactive Q&A loop")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    app = RAGQAApp(args.doc, top_k=args.top_k)

    if args.interactive:
        print("\nInteractive RAG Q&A. Type 'exit' to quit.\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue
            result = app.answer(q)
            print(f"\nAgent: {result['answer']}\n")
    elif args.query:
        result = app.answer(args.query)
        print("\n--- ANSWER ---")
        print(result["answer"])
        print(f"\n(Retrieved {len(result['context_used'])} relevant chunk(s) from the document)")
    else:
        parser.error("Provide --query \"...\" or use --interactive")


if __name__ == "__main__":
    main()
