"""Retrieve evidence and NLI-score every sampled claim once, caching the results.

For each claim in data/fever_kb/claims.jsonl: retrieve the top-k sentences, then score
each sentence against the claim with the NLI model in two premise formats:
  plain   the sentence as stored
  titled  "<page title>: <sentence>" -- FEVER annotators used the page title to resolve
          references like "It" / "The copper statue" (FEVER §3.2)
Aggregation rules are then compared offline by scripts/eval_aggregation.py without
re-running the model.

Writes data/fever_kb/nli_cache_<model-slug>.jsonl.
Run from backend/:  python scripts/cache_nli_scores.py [--model cross-encoder/nli-deberta-v3-small]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.retriever import retriever  # noqa: E402

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BACKEND_DIR, "data", "fever_kb")


def load_classifier(model):
    from transformers import pipeline
    return pipeline("text-classification", model=model)


def score(classifier, premises, claim):
    batch = classifier([{"text": p, "text_pair": claim} for p in premises], top_k=None)
    return [{s["label"].lower(): round(s["score"], 5) for s in scores} for scores in batch]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    classifier = load_classifier(args.model)
    with open(os.path.join(KB_DIR, "claims.jsonl"), encoding="utf-8") as f:
        claims = [json.loads(line) for line in f]

    out_path = os.path.join(KB_DIR, f"nli_cache_{args.model.replace('/', '__')}.jsonl")
    start = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        for i, c in enumerate(claims, 1):
            spans = retriever.retrieve(c["claim"], top_k=args.top_k, max_distance=float("inf"))
            record = {"id": c["id"], "claim": c["claim"], "label": c["label"],
                      "evidence_sets": c["evidence_sets"], "spans": []}
            if spans:
                plain = score(classifier, [s.text for s in spans], c["claim"])
                titled = score(classifier, [f"{s.page}: {s.text}" for s in spans], c["claim"])
                for s, p, t in zip(spans, plain, titled):
                    record["spans"].append({"span_id": s.span_id, "page": s.page, "distance": s.distance,
                                            "plain": p, "titled": t})
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(claims)} claims ({time.time() - start:.0f}s)", flush=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
