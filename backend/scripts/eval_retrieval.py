"""Measure Stage 2 retrieval on the sampled FEVER claims (Phase 2, item 2.1 check).

For every SUPPORTS / REFUTES claim in data/fever_kb/claims.jsonl (NEI claims have no
gold evidence), retrieve the top-k sentences and report:
  - hit@k        at least one gold evidence sentence is in the top k
  - full@k       some complete gold evidence set is in the top k (FEVER's "fully supported", §5.3)
  - distances    of gold hits vs. the rest, to recalibrate MAX_EVIDENCE_DISTANCE

Run from backend/:  python scripts/eval_retrieval.py
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.retriever import retriever  # noqa: E402

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIMS_PATH = os.path.join(BACKEND_DIR, "data", "fever_kb", "claims.jsonl")
KS = [1, 5, 10, 20]


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")


def main():
    if retriever.source != "fever":
        sys.exit("No FEVER knowledge base found; run scripts/build_fever_kb.py first.")

    with open(CLAIMS_PATH, encoding="utf-8") as f:
        claims = [json.loads(line) for line in f]
    labelled = [c for c in claims if c["evidence_sets"]]

    max_k = max(KS)
    hits = {k: 0 for k in KS}
    full = {k: 0 for k in KS}
    gold_dist, other_dist, nei_top1 = [], [], []

    for c in labelled:
        spans = retriever.retrieve(c["claim"], top_k=max_k, max_distance=float("inf"))
        ids = [s.span_id for s in spans]
        gold = {s for ev in c["evidence_sets"] for s in ev}
        for k in KS:
            top = set(ids[:k])
            hits[k] += bool(top & gold)
            full[k] += any(set(ev) <= top for ev in c["evidence_sets"])
        for s in spans[:5]:
            (gold_dist if s.span_id in gold else other_dist).append(s.distance)

    # Closest distance for NEI claims: a reference for "nothing relevant in the KB".
    for c in claims:
        if not c["evidence_sets"]:
            spans = retriever.retrieve(c["claim"], top_k=1, max_distance=float("inf"))
            if spans:
                nei_top1.append(spans[0].distance)

    n = len(labelled)
    report = {
        "claims_with_evidence": n,
        "hit_at_k": {k: round(hits[k] / n, 4) for k in KS},
        "full_evidence_at_k": {k: round(full[k] / n, 4) for k in KS},
        "distance_gold_hits": {"n": len(gold_dist), "median": round(statistics.median(gold_dist), 4),
                               "p90": round(percentile(gold_dist, 0.9), 4)} if gold_dist else None,
        "distance_non_gold_top5": {"n": len(other_dist), "median": round(statistics.median(other_dist), 4),
                                   "p10": round(percentile(other_dist, 0.1), 4)} if other_dist else None,
        "distance_nei_top1": {"n": len(nei_top1), "median": round(statistics.median(nei_top1), 4),
                              "p10": round(percentile(nei_top1, 0.1), 4)} if nei_top1 else None,
    }
    print(json.dumps(report, indent=2))
    with open(os.path.join(BACKEND_DIR, "data", "fever_kb", "retrieval_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
