"""Compare Stage 4 aggregation rules on cached NLI scores (Phase 2, item 2.2).

Reads data/fever_kb/nli_cache_<model>.jsonl (scripts/cache_nli_scores.py) and
data/fever_kb/splits.json (scripts/make_splits.py). For each configuration:
  premise    plain | titled            sentence alone, or "<page title>: <sentence>"
  relevance  none  | title_in_claim    only sentences whose page title (minus any
                                       "(...)" qualifier) appears in the claim may decide
  cutoff     tuned on DEV          drop sentences farther than this retrieval distance, so a
                                       claim with no close match becomes NEI (CUTOFFS grid)
it scores two rules against FEVER gold labels:
  argmax       current app rule: Supported if any deciding sentence's top class is
               entailment, else Refuted if any is contradiction, else NEI
  report_rule  report §4.1: Refuted if any P_contra > τ_c, else Supported if
               max P_entail > τ_e, else NEI; τ_c, τ_e tuned on DEV, accuracy reported on TEST
  entail_first same thresholds, order swapped: Supported if max P_entail > τ_e, else Refuted
               if any P_contra > τ_c, else NEI (tuned on DEV, reported on TEST)

Run from backend/:  python scripts/eval_aggregation.py [--model cross-encoder/nli-deberta-v3-small]
"""
import argparse
import itertools
import json
import os
import re

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BACKEND_DIR, "data", "fever_kb")
GOLD = {"SUPPORTS": "Supported", "REFUTES": "Refuted", "NOT ENOUGH INFO": "Not Enough Information"}
NEI = "Not Enough Information"
CUTOFFS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, None]  # None = no cutoff
TAUS = [round(0.05 * i, 2) for i in range(2, 20)] + [0.97, 0.99]  # 0.10 .. 0.99


def title_core(page):
    return re.sub(r"\s*\(.*?\)\s*", " ", page).strip().lower()


def deciding_spans(record, premise, relevance, cutoff):
    claim = record["claim"].lower()
    spans = []
    for s in record["spans"]:
        if cutoff is not None and s["distance"] > cutoff:
            continue
        if relevance == "title_in_claim" and title_core(s["page"]) not in claim:
            continue
        spans.append(s[premise])
    return spans


def rule_argmax(probs):
    tops = [max(p, key=p.get) for p in probs]
    if "entailment" in tops:
        return "Supported"
    if "contradiction" in tops:
        return "Refuted"
    return NEI


def rule_report(probs, tau_c, tau_e):
    if probs and max(p["contradiction"] for p in probs) > tau_c:
        return "Refuted"
    if probs and max(p["entailment"] for p in probs) > tau_e:
        return "Supported"
    return NEI


def rule_entail_first(probs, tau_c, tau_e):
    if probs and max(p["entailment"] for p in probs) > tau_e:
        return "Supported"
    if probs and max(p["contradiction"] for p in probs) > tau_c:
        return "Refuted"
    return NEI


def tuned(rule, premise, relevance, dev, test, thresholds=True):
    """Pick cutoff (and τ_c, τ_e for thresholded rules) by DEV accuracy; report DEV and TEST."""
    best_key, best_acc = None, -1.0
    for cutoff in CUTOFFS:
        probs = {r["id"]: deciding_spans(r, premise, relevance, cutoff) for r in dev}
        grid = [(tc, te) for tc in TAUS for te in TAUS] if thresholds else [()]
        for taus in grid:
            acc = accuracy(dev, lambda r: rule(probs[r["id"]], *taus))["overall"]
            if acc > best_acc:
                best_key, best_acc = (cutoff, taus), acc
    cutoff, taus = best_key

    def predict(r):
        return rule(deciding_spans(r, premise, relevance, cutoff), *taus)
    result = {"cutoff": cutoff, "dev": accuracy(dev, predict), "test": accuracy(test, predict)}
    if thresholds:
        result.update({"tau_c": taus[0], "tau_e": taus[1]})
    return result


def accuracy(records, predict):
    per_label = {g: [0, 0] for g in GOLD.values()}
    for r in records:
        gold = GOLD[r["label"]]
        per_label[gold][1] += 1
        per_label[gold][0] += predict(r) == gold
    overall = sum(c for c, _ in per_label.values()) / sum(n for _, n in per_label.values())
    return {"overall": round(overall, 4),
            **{g.split()[0]: round(c / n, 4) for g, (c, n) in per_label.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-small")
    args = parser.parse_args()

    with open(os.path.join(KB_DIR, f"nli_cache_{args.model.replace('/', '__')}.jsonl"), encoding="utf-8") as f:
        records = {r["id"]: r for r in map(json.loads, f)}
    with open(os.path.join(KB_DIR, "splits.json"), encoding="utf-8") as f:
        splits = json.load(f)
    dev = [records[i] for i in splits["dev"]]
    test = [records[i] for i in splits["test"]]

    results = []
    for premise, relevance in itertools.product(["plain", "titled"], ["none", "title_in_claim"]):
        results.append({
            "premise": premise, "relevance": relevance,
            "argmax": tuned(rule_argmax, premise, relevance, dev, test, thresholds=False),
            "report_rule": tuned(rule_report, premise, relevance, dev, test),
            "entail_first": tuned(rule_entail_first, premise, relevance, dev, test),
        })

    print(f"model: {args.model}   dev={len(dev)} test={len(test)}")
    def fmt(x):
        t = x["test"]
        return f"{x['dev']['overall']:.3f} {t['overall']:.3f} ({t['Supported']:.2f}/{t['Refuted']:.2f}/{t['Not']:.2f})"

    def params(x):
        return f"cut={x['cutoff']}" + (f",tc={x['tau_c']},te={x['tau_e']}" if "tau_c" in x else "")

    print("each cell: [tuned params] dev test (test Supported/Refuted/NEI accuracy)")
    for r in results:
        print(f"{r['premise']:7} {r['relevance']:15}")
        for rule in ("argmax", "report_rule", "entail_first"):
            print(f"    {rule:13} [{params(r[rule])}] {fmt(r[rule])}")

    with open(os.path.join(KB_DIR, f"aggregation_eval_{args.model.replace('/', '__')}.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
