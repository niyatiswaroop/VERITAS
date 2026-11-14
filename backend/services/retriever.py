import os
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from models import EvidenceSpan

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Persistent FEVER knowledge base built by scripts/build_fever_kb.py (Phase 2, item 2.1).
FEVER_KB_DIR = os.path.join(BACKEND_DIR, "data", "fever_kb", "chroma")
FEVER_COLLECTION = "fever_kb"
# Fallback knowledge base: the prototype's text files, used only when no FEVER KB has been built.
FALLBACK_DATA_DIR = os.path.join(BACKEND_DIR, "data")

# Sentences retrieved per claim. FEVER's baseline also keeps the top 5 sentences (l = 5, §5.6).
DEFAULT_TOP_K = 5

# Max distance (Chroma's default squared L2 on MiniLM's unit-length embeddings) for a
# sentence to count as relevant evidence. Farther sentences are treated as "not about the
# claim", so the claim is reported Not Enough Information rather than being force-matched
# to whatever is closest. Tuned on the FEVER dev split together with the Stage 4 thresholds
# (scripts/eval_aggregation.py); was 1.0, hand-set on the prototype's text files.
MAX_EVIDENCE_DISTANCE = 1.2


def split_into_chunks(text, chunk_size=500, overlap=50):
    """Fixed-size character chunks, only used by the fallback text-file knowledge base."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 30:
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


class EvidenceRetriever:
    def __init__(self, kb_dir: str = FEVER_KB_DIR):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        if os.path.isdir(kb_dir):
            self.client = chromadb.PersistentClient(path=kb_dir)
            self.collection = self.client.get_collection(FEVER_COLLECTION)
            self.source = "fever"
            print(f"Using FEVER knowledge base: {self.collection.count()} sentences.")
        else:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection("veritas_fallback_kb")
            self.source = "fallback"
            self._index_fallback_files()

    def _index_fallback_files(self):
        """Index the prototype's text files in memory (no FEVER KB built yet)."""
        ids, chunks, metadatas = [], [], []
        for filename in sorted(os.listdir(FALLBACK_DATA_DIR)):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(FALLBACK_DATA_DIR, filename), "r", encoding="utf-8") as f:
                text = f.read()
            page = filename[:-len(".txt")]
            for i, chunk in enumerate(split_into_chunks(text, chunk_size=200, overlap=20)):
                ids.append(f"{page}::chunk_{i}")
                chunks.append(chunk)
                metadatas.append({"page": page})
        if chunks:
            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=self.embedder.encode(chunks).tolist(),
                metadatas=metadatas,
            )
        print(f"No FEVER knowledge base found; using fallback text files ({len(chunks)} chunks).")

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K,
                 max_distance: float = MAX_EVIDENCE_DISTANCE) -> List[EvidenceSpan]:
        """Return up to top_k evidence spans for a query, closest first.

        Spans farther than max_distance are dropped; an empty list means the
        knowledge base has nothing relevant, so the claim is reported Not Enough
        Information instead of being matched to an unrelated sentence.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[self.embedder.encode(query).tolist()],
            n_results=top_k,
        )
        spans = []
        for span_id, text, metadata, distance in zip(
                results["ids"][0], results["documents"][0],
                results["metadatas"][0], results["distances"][0]):
            if distance <= max_distance:
                spans.append(EvidenceSpan(
                    span_id=span_id,
                    page=metadata.get("title") or metadata.get("page", ""),
                    text=text,
                    distance=round(distance, 4),
                ))
        return spans


retriever = EvidenceRetriever()
