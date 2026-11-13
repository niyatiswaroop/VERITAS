"""Cross-check Stage 4 verdicts against FEVER gold labels on the sampled claims.

For every claim in data/fever_kb/claims.jsonl it retrieves the top-5 sentences once,
scores each with the NLI model (alone and joined), and compares aggregation variants:

  V0_joined_top5   current pipeline: all retrieved sentences joined into one premise, argmax
  V1_top1          only the closest sentence, argmax
  V2_entail_first  score each sentence alone; Supported if any argmax=entailment,
                   else Refuted if any argmax=contradiction, else NEI
  V3_report_rule   report §4.1 rule with untuned τ_c = τ_e = 0.5: Refuted if any P_contra > τ_c,
                   else Supported if max P_entail > τ_e, else NEI
  ORACLE_gold      NLI on FEVER's own gold evidence (first set, joined) -> NLI quality with
                   perfect retrieval (SUPPORTS/REFUTES claims only)
  COPY_TEST        the gold evidence sentence itself used as the claim, through the full
                   pipeline variants -> should always be Supported

Run from backend/:  python scripts/eval_verification.py
"""
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.nli_verifier import verifier  # noqa: E402
from services.retriever import retriever  # noqa: E402

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BACKEND_DIR, "data", "fever_kb")
GOLD = {"SUPPORTS": "Supported", "REFUTES": "Refuted", "NOT ENOUGH INFO": "Not Enough Information"}
LABEL = {"entailment": "Supported", "contradiction": "Refuted", "neutral": "Not Enough Information"}


def nli(premise, claim):
    scores = verifier.classifier({"text": premise, "text_pair": claim}, top_k=None)
    return {s["label"].lower(): s["score"] for s in scores}


def argmax_label(p):
    return LABEL[max(p, key=p.get)]


def variants(claim):
    spans = retriever.retrieve(claim)
    if not spans:
        nei = "Not Enough Information"
        return {"V0_joined_top5": nei, "V1_top1": nei, "V2_entail_first": nei, "V3_report_rule": nei}
    per = [nli(s.text, claim) for s in spans]
    joined = nli(" ".join(s.text for s in spans), claim)
    tops = [argmax_label(p) for p in per]
    v2 = ("Supported" if "Supported" in tops else "Refuted" if "Refuted" in tops
          else "Not Enough Information")
    v3 = ("Refuted" if max(p["contradiction"] for p in per) > 0.5
          else "Supported" if max(p["entailment"] for p in per) > 0.5
          else "Not Enough Information")
    return {"V0_joined_top5": argmax_label(joined), "V1_top1": tops[0], "V2_entail_first": v2, "V3_report_rule": v3}


def main():
    with open(os.path.join(KB_DIR, "claims.jsonl"), encoding="utf-8") as f:
        claims = [json.loads(line) for line in f]
    con = sqlite3.connect(os.path.join(KB_DIR, "kb.sqlite"))
    text_of = dict(con.execute("SELECT span_id, text FROM sentences WHERE kb_version = 0"))

    correct = defaultdict(Counter)   # variant -> gold label -> correct count
    totals = Counter()
    confusion = defaultdict(Counter)  # variant -> (gold, predicted)
    copy_ok = Counter()
    copy_n = 0
    examples = defaultdict(list)

    for i, c in enumerate(claims, 1):
        gold = GOLD[c["label"]]
        totals[gold] += 1
        preds = variants(c["claim"])
        if c["evidence_sets"]:
            texts = [text_of[s] for s in c["evidence_sets"][0] if s in text_of]
            if texts:
                preds["ORACLE_gold"] = argmax_label(nli(" ".join(texts), c["claim"]))
        for name, pred in preds.items():
            correct[name][gold] += pred == gold
            confusion[name][(gold, pred)] += 1
        if gold == "Supported" and preds["V0_joined_top5"] != gold and len(examples["V0_miss"]) < 8:
            examples["V0_miss"].append({"claim": c["claim"], "V0": preds["V0_joined_top5"], "V2": preds["V2_entail_first"]})

        # Copy test: a gold evidence sentence verbatim as the claim must come out Supported.
        if c["label"] == "SUPPORTS" and len(c["evidence_sets"][0]) == 1 and c["evidence_sets"][0][0] in text_of:
            copy_n += 1
            for name, pred in variants(text_of[c["evidence_sets"][0][0]]).items():
                copy_ok[name] += pred == "Supported"
        if i % 100 == 0:
            print(f"  {i}/{len(claims)} claims", flush=True)

    report = {"n_claims": dict(totals), "accuracy": {}, "copy_test": {}, "confusion": {}, "examples": examples}
    for name in correct:
        n = sum(totals.values()) if name != "ORACLE_gold" else totals["Supported"] + totals["Refuted"]
        report["accuracy"][name] = {
            "overall": round(sum(correct[name].values()) / n, 4),
            **{g: round(correct[name][g] / totals[g], 4) for g in totals
               if not (name == "ORACLE_gold" and g == "Not Enough Information")},
        }
        report["confusion"][name] = {f"{g} -> {p}": n for (g, p), n in sorted(confusion[name].items())}
    report["copy_test"] = {"n": copy_n, **{k: round(v / copy_n, 4) for k, v in copy_ok.items()}}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(os.path.join(KB_DIR, "verification_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
