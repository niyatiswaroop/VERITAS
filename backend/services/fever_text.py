"""Helpers for FEVER's tokenised Wikipedia text.

FEVER's wiki-pages dump is PTB-tokenised: brackets are escaped (-LRB-, -RRB-, ...),
quotes are `` and '', and punctuation is space-separated. We store and show
detokenised text so the NLI model and the UI see normal sentences, but keep the
original FEVER page id in every span id so evidence still matches FEVER's labels.
"""
import re
import unicodedata

_TOKEN_MAP = {
    "-LRB-": "(", "-RRB-": ")",
    "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}",
    "-COLON-": ":",
    "``": '"', "''": '"',
}


def detokenize(text: str) -> str:
    for token, char in _TOKEN_MAP.items():
        text = text.replace(token, char)
    text = re.sub(r"\s+([.,;:!?%)\]}])", r"\1", text)   # no space before closing punctuation
    text = re.sub(r"([(\[{$])\s+", r"\1", text)          # no space after opening brackets
    text = re.sub(r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b", r"\1", text)
    text = re.sub(r"\s+--\s+", " – ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def normalize_page_id(page_id: str) -> str:
    """FEVER's claim files and wiki dump encode accented page ids in different Unicode
    forms (decomposed vs composed), e.g. 'Janelle_Monáe'; normalise to NFC so they match."""
    return unicodedata.normalize("NFC", page_id)


def page_title(page_id: str) -> str:
    """'Soul_Food_-LRB-film-RRB-' -> 'Soul Food (film)'."""
    return detokenize(page_id.replace("_", " "))


def span_id(page_id: str, sent_idx: int) -> str:
    """Stable evidence id: the FEVER page id plus sentence index (report §4.3 evidence_span_ids)."""
    return f"{normalize_page_id(page_id)}::{sent_idx}"


def parse_lines(lines_field: str):
    """Yield (sentence_index, raw_text) for each non-empty sentence of a wiki-pages entry.

    Each line is 'idx<TAB>sentence<TAB>anchor<TAB>target...'; only the first two fields matter.
    """
    for line in lines_field.split("\n"):
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        text = parts[1].strip()
        if text:
            yield int(parts[0]), text
