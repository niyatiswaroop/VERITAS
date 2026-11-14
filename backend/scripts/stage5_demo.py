"""Early Stage 5 prototype: Change-Aware Re-Verification (report §4.3; Phase 2, item 2.6).

NOT YET EVALUATED -- a single-scenario walk-through of the §4.3 mechanism on real FEVER data:
  1. Claim Store       verify claims; store verdict, evidence_span_ids, kb_version_at_check
  2. KB update         edit one sentence and delete another -> KB version 1
  3. Change Detector   content-level diff of cited sentences between versions (edit / delete)
  4. Dependency Mapper only stored claims citing a changed span are "at risk"
  5. Re-verification   re-run Stages 2->4 on the at-risk claims only, against KB version 1;
                       UNCHANGED if the verdict holds, FLAGGED if it changed (Outdated per DyKnow),
                       MANUAL REVIEW if the new verdict's confidence is below α (AutoVU-KG gate)
Runs on a COPY of the knowledge base (data/fever_kb/stage5_demo/); the app's KB is untouched.

Run from backend/:  python scripts/stage5_demo.py
"""
import difflib
import json
import os
import random
import shutil
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.nli_verifier import verifier  # noqa: E402
from services.retriever import EvidenceRetriever, FEVER_COLLECTION  # noqa: E402

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BACKEND_DIR, "data", "fever_kb")
DEMO_DIR = os.path.join(KB_DIR, "stage5_demo")
DELTA = 0.0   # Diff threshold δ: any content change to a cited sentence triggers (report §4.3)
ALPHA = 0.80  # confidence gate α: new verdicts below this go to manual review

# The KB update applied in this demo (v0 -> v1).
UPDATES = {
    "Bermuda_Triangle::0": ("edit", "The Bermuda Triangle, also known as the Devil's Triangle, is a loosely-defined "
                                    "region in the South Pacific Ocean, where a number of aircraft and ships are said "
                                    "to have disappeared under mysterious circumstances."),
    "Chile::0": ("delete", None),
}
FIXED_CLAIMS = [
    "The Bermuda Triangle is in the western part of the North Atlantic Ocean.",
    "The Bermuda Triangle is a loosely-defined region in the Pacific Ocean.",
    "Chile is in South America.",
]


def verify(retriever, claim):
    spans = retriever.retrieve(claim)
    verdict, probs, _source, decisive = verifier.verify(claim, spans)
    cited = [s.span_id for s in spans if s.relevant] or [s.span_id for s in spans]
    confidence = max(probs.entailment, probs.contradiction, probs.neutral) if probs else 0.0
    return verdict, cited, decisive, confidence


def main():
    # Fresh copy of the KB so the demo's update never touches the app's data.
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    shutil.copytree(os.path.join(KB_DIR, "chroma"), os.path.join(DEMO_DIR, "chroma"))
    shutil.copy(os.path.join(KB_DIR, "kb.sqlite"), os.path.join(DEMO_DIR, "kb.sqlite"))
    kb = sqlite3.connect(os.path.join(DEMO_DIR, "kb.sqlite"))
    retriever = EvidenceRetriever(kb_dir=os.path.join(DEMO_DIR, "chroma"))

    # ---- 1. Claim Store (kb_version 0) ----
    with open(os.path.join(KB_DIR, "claims.jsonl"), encoding="utf-8") as f:
        fever = [json.loads(line)["claim"] for line in f]
    claims = FIXED_CLAIMS + random.Random(1).sample(fever, 60)
    kb.execute("""CREATE TABLE claim_store (claim_id INTEGER PRIMARY KEY, text TEXT, verdict TEXT,
                  evidence_span_ids TEXT, decisive_span_id TEXT, kb_version_at_check INTEGER)""")
    print(f"1. Verifying and storing {len(claims)} claims at KB version 0 ...")
    for i, claim in enumerate(claims):
        verdict, cited, decisive, _ = verify(retriever, claim)
        kb.execute("INSERT INTO claim_store VALUES (?, ?, ?, ?, ?, 0)", (i, claim, verdict, json.dumps(cited), decisive))
    kb.commit()

    # ---- 2. KB update -> version 1 ----
    print("\n2. Applying KB update (version 0 -> 1):")
    for span_id, (op, new_text) in UPDATES.items():
        print(f"   {op:6} {span_id}")
        if op == "edit":
            page, idx = span_id.rsplit("::", 1)
            kb.execute("INSERT INTO sentences (span_id, page, sent_idx, text, kb_version) VALUES (?, ?, ?, ?, 1)",
                       (span_id, page, int(idx), new_text))
            retriever.collection.update(ids=[span_id], documents=[new_text],
                                        embeddings=[retriever.embedder.encode(new_text).tolist()])
        else:
            retriever.collection.delete(ids=[span_id])
    kb.commit()

    # ---- 3. Change Detector: content-level diff of every cited span ----
    def text_at(span_id, version):
        if version >= 1 and UPDATES.get(span_id, (None,))[0] == "delete":
            return None
        row = kb.execute("SELECT text FROM sentences WHERE span_id = ? AND kb_version <= ? "
                         "ORDER BY kb_version DESC LIMIT 1", (span_id, version)).fetchone()
        return row[0] if row else None

    stored = kb.execute("SELECT claim_id, text, verdict, evidence_span_ids FROM claim_store").fetchall()
    cited_spans = {s for _, _, _, ev in stored for s in json.loads(ev)}
    changed = {}
    for span_id in cited_spans:
        old, new = text_at(span_id, 0), text_at(span_id, 1)
        if new is None:
            changed[span_id] = ("delete", 1.0)
        else:
            diff = 1 - difflib.SequenceMatcher(None, old, new).ratio()
            if diff > DELTA:
                changed[span_id] = ("edit", round(diff, 3))
    print(f"\n3. Change Detector: {len(changed)} of {len(cited_spans)} cited sentences changed: {changed}")

    # ---- 4. Dependency Mapper ----
    at_risk = [row for row in stored if set(json.loads(row[3])) & changed.keys()]
    print(f"\n4. Dependency Mapper: {len(at_risk)} of {len(stored)} stored claims cite a changed sentence")

    # ---- 5. Selective re-verification ----
    print("\n5. Re-verifying only the at-risk claims against KB version 1:")
    for claim_id, text, old_verdict, _ in at_risk:
        new_verdict, _, decisive, confidence = verify(retriever, text)
        if confidence < ALPHA:
            status = "MANUAL REVIEW"
        elif new_verdict == old_verdict:
            status = "UNCHANGED"
        else:
            status = "FLAGGED (Outdated)" if old_verdict == "Supported" else "FLAGGED"
        print(f"   [{status}] {old_verdict} -> {new_verdict} (conf {confidence:.2f}, decided by {decisive})\n"
              f"       {text}")

    print(f"\nCost: re-verified {len(at_risk)} claims instead of all {len(stored)} "
          f"({len(at_risk) * 5} NLI calls instead of {len(stored) * 5}).")


if __name__ == "__main__":
    main()
