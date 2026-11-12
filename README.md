# VERITAS

**A Change-Aware Framework for Verifying Evidence and Hallucination in Retrieval-Augmented Generation Systems**

Major Project, B.Tech CSE, Jaypee Institute of Information Technology, Noida (2026–2027).
Team: Om Jadoun, Ojas Tayal, Niyati. Supervisor: Ms. Ambalika Sarkar.

## What VERITAS does

VERITAS verifies the factual claims in an LLM's answer against a knowledge base, along three aspects:

1. **Evidence Trust:** check whether the retrieved evidence itself is trustworthy instead of assuming it is correct.
2. **Change-Aware Re-Verification:** when the knowledge base is updated, selectively re-verify only the stored claims whose evidence changed.
3. **Interpretable Trust Assessment:** give a structured, human-readable explanation for every verdict.

Verdicts use FEVER's labels: **Supported**, **Refuted**, **Not Enough Information**.

## Six-stage pipeline and current status

| Stage | Status | Notes |
|---|---|---|
| 1. Claim Decomposition | Not implemented | Regex sentence split; LLM-based decomposition (FActScore) planned for Phase 2 |
| 2. Evidence Retrieval | Partially implemented | MiniLM (`all-MiniLM-L6-v2`) embeddings + in-memory ChromaDB over `backend/data/`, single granularity |
| 3. Evidence Trust Check *(novel)* | Not implemented | Trust score for each retrieved passage (agreement, recency, LLM-judge expertise weight) |
| 4. Claim Verification | Partially implemented | `cross-encoder/nli-deberta-v3-small`; returns entailment/contradiction/neutral probabilities; verdict is currently the most probable class |
| 5. Change-Aware Re-Verification *(novel)* | Not implemented | Claim store, change detector, dependency mapper, selective re-verification |
| 6. Explainable Verdict Output | Not implemented | Structured verdict with evidence used, error type and what changed |

Datasets: FEVER (primary), RAGTruth (secondary), FENICE (evidence-alignment methodology). The current prototype still uses a small set of text files in `backend/data/`.

### Demo examples

The three example answers in the UI (Eiffel Tower, Great Wall, Statue of Liberty) return **pinned verdicts without calling the NLI model** (`backend/services/demo_examples.py`). The API marks every claim with a `verdict_source` (`nli_model`, `pinned_demo`, `no_evidence`, `model_unavailable`), and the UI tags pinned verdicts so they are not mistaken for model output. Any other input goes through the real pipeline.

## API

`POST /verify` with `{"generated_answer": "..."}` returns per-claim verdicts, retrieved evidence, NLI probabilities and verdict source, plus a faithfulness score, hallucination score and overall classification.

## Setup

### Backend (Python)

```bash
cd backend
python -m venv venv
# Windows: .\venv\Scripts\activate    Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The first start downloads the embedding and NLI models from Hugging Face.

### Frontend (Node.js)

```bash
cd frontend
npm install
npm run dev
```
