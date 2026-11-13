"""Build the FEVER knowledge base for VERITAS (Phase 2, item 2.1).

Reads the raw FEVER downloads in backend/data/fever/ and writes backend/data/fever_kb/:
  - claims.jsonl  sampled dev claims, balanced across labels, with gold evidence span ids
  - kb.sqlite     one row per Wikipedia sentence (span_id, page, sent_idx, text, kb_version)
  - chroma/       persistent ChromaDB index of the same sentences (MiniLM embeddings)

Knowledge base = every page cited by the sampled claims + the demo-example pages
+ randomly sampled distractor pages, so retrieval is not trivially easy.

Run from backend/:  python scripts/build_fever_kb.py
"""
import argparse
import json
import os
import random
import sqlite3
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.fever_text import detokenize, normalize_page_id, page_title, parse_lines, span_id  # noqa: E402

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BACKEND_DIR, "data", "fever")
OUT_DIR = os.path.join(BACKEND_DIR, "data", "fever_kb")

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
# Pages behind the UI demo examples, so their evidence comes from FEVER too.
DEMO_PAGES = {"Eiffel_Tower", "Great_Wall_of_China", "Statue_of_Liberty", "Taj_Mahal"}
COLLECTION = "fever_kb"


def sample_claims(path, per_label, rng):
    by_label = {label: [] for label in LABELS}
    with open(path, encoding="utf-8") as f:
        for line in f:
            claim = json.loads(line)
            by_label[claim["label"]].append(claim)
    sampled = []
    for label in LABELS:
        sampled.extend(rng.sample(by_label[label], per_label))
    rng.shuffle(sampled)

    claims = []
    for c in sampled:
        # FEVER evidence: list of alternative evidence sets, each a list of
        # [annotation_id, evidence_id, page, sentence]; page is None for NEI.
        evidence_sets = []
        for ev_set in c["evidence"]:
            spans = [span_id(page, sent) for _, _, page, sent in ev_set if page is not None]
            if spans and spans not in evidence_sets:
                evidence_sets.append(spans)
        claims.append({
            "id": c["id"],
            "claim": c["claim"],
            "label": c["label"],
            "evidence_sets": evidence_sets,
        })
    return claims


def scan_wiki(zip_path, target_pages, n_distractors, rng):
    """One pass over the dump: keep target pages, reservoir-sample distractor pages."""
    found, reservoir, seen_candidates = {}, [], 0
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.startswith("wiki-pages/") and n.endswith(".jsonl"))
        for i, name in enumerate(names, 1):
            with zf.open(name) as f:
                for raw in f:
                    page = json.loads(raw)
                    pid, lines = normalize_page_id(page["id"]), page["lines"]
                    if not pid or not lines:
                        continue
                    if pid in target_pages:
                        found[pid] = lines
                        continue
                    # Reservoir sampling (Algorithm R) over all other non-empty pages.
                    seen_candidates += 1
                    if len(reservoir) < n_distractors:
                        reservoir.append((pid, lines))
                    else:
                        j = rng.randrange(seen_candidates)
                        if j < n_distractors:
                            reservoir[j] = (pid, lines)
            print(f"  scanned {i}/{len(names)} files, {len(found)}/{len(target_pages)} target pages found",
                  flush=True)
    return found, reservoir


def build_rows(pages):
    rows = []
    for pid, lines in pages:
        for idx, text in parse_lines(lines):
            rows.append((span_id(pid, idx), pid, idx, detokenize(text)))
    return rows


def write_sqlite(path, rows):
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE sentences (
        span_id TEXT NOT NULL, page TEXT NOT NULL, sent_idx INTEGER NOT NULL,
        text TEXT NOT NULL, kb_version INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (span_id, kb_version))""")
    con.execute("CREATE INDEX idx_sentences_page ON sentences(page)")
    con.executemany("INSERT INTO sentences (span_id, page, sent_idx, text) VALUES (?, ?, ?, ?)", rows)
    con.commit()
    con.close()


def write_chroma(path, rows, batch_size):
    import chromadb
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=path)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)
    start = time.time()
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = [r[3] for r in batch]
        embeddings = embedder.encode(texts, batch_size=64).tolist()
        collection.add(
            ids=[r[0] for r in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"page": r[1], "sent_idx": r[2], "title": page_title(r[1])} for r in batch],
        )
        done = i + len(batch)
        rate = done / (time.time() - start)
        print(f"  embedded {done}/{len(rows)} sentences ({rate:.0f}/s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-label", type=int, default=333, help="dev claims sampled per label")
    parser.add_argument("--distractors", type=int, default=10000, help="random unrelated pages to add")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("1/4 Sampling claims")
    claims = sample_claims(os.path.join(RAW_DIR, "paper_dev.jsonl"), args.per_label, rng)
    with open(os.path.join(OUT_DIR, "claims.jsonl"), "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(c) + "\n")
    cited = {s.rsplit("::", 1)[0] for c in claims for ev in c["evidence_sets"] for s in ev}
    targets = cited | DEMO_PAGES
    print(f"  {len(claims)} claims, {len(cited)} cited pages")

    print("2/4 Scanning wiki-pages.zip")
    found, distractors = scan_wiki(os.path.join(RAW_DIR, "wiki-pages.zip"), targets, args.distractors, rng)
    missing = sorted(targets - found.keys())
    if missing:
        print(f"  WARNING: {len(missing)} target pages not in the dump: {missing[:10]}")

    rows = build_rows(list(found.items()) + distractors)
    print(f"  {len(found)} target + {len(distractors)} distractor pages -> {len(rows)} sentences")

    print("3/4 Writing kb.sqlite")
    write_sqlite(os.path.join(OUT_DIR, "kb.sqlite"), rows)

    print("4/4 Embedding into ChromaDB")
    write_chroma(os.path.join(OUT_DIR, "chroma"), rows, args.batch_size)

    with open(os.path.join(OUT_DIR, "build_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "per_label": args.per_label, "distractors": args.distractors, "seed": args.seed,
            "claims": len(claims), "cited_pages": len(cited), "target_pages_found": len(found),
            "missing_target_pages": missing, "distractor_pages": len(distractors),
            "sentences": len(rows), "kb_version": 0, "embedder": "all-MiniLM-L6-v2",
            "build_seconds": round(time.time() - t0),
        }, f, indent=2)
    print(f"Done in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
