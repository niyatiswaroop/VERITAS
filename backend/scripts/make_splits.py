"""Split the sampled FEVER claims into dev / test (Phase 2, item 2.2, step 1).

Report §4.3 / §6.2: thresholds are calibrated on a held-out development slice and final
results are reported on a separate test slice not seen during calibration. The split is
stratified by label and seeded, so it is reproducible.

Writes data/fever_kb/splits.json: {"dev": [claim ids], "test": [claim ids]}.
Run from backend/:  python scripts/make_splits.py
"""
import argparse
import json
import os
import random
from collections import defaultdict

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BACKEND_DIR, "data", "fever_kb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(os.path.join(KB_DIR, "claims.jsonl"), encoding="utf-8") as f:
        claims = [json.loads(line) for line in f]
    by_label = defaultdict(list)
    for c in claims:
        by_label[c["label"]].append(c["id"])

    rng = random.Random(args.seed)
    splits = {"dev": [], "test": []}
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        rng.shuffle(ids)
        cut = round(len(ids) * args.dev_fraction)
        splits["dev"] += ids[:cut]
        splits["test"] += ids[cut:]
        print(f"{label}: {cut} dev / {len(ids) - cut} test")

    with open(os.path.join(KB_DIR, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "dev_fraction": args.dev_fraction, **splits}, f, indent=2)
    print(f"dev={len(splits['dev'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()
