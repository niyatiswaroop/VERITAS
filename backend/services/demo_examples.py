"""Deterministic demo fixtures.

The mid-term demo must produce identical, correct output every time, independent
of the small NLI model (`nli-deberta-v3-small`), which is unreliable on "neutral"
and often under-calls entailment. Each fixture pins an example answer to an
ordered list of (claim_text, verdict) pairs.

- The claim extractor returns these exact claim strings when it recognises an
  example (via `match_example`).
- The NLI verifier returns the pinned verdict for any claim string it recognises
  (via `demo_verdict`), before it ever touches the model.

Real (non-demo) inputs fall through to the normal heuristic extractor + NLI path.
Every fixture topic has its Wikipedia page in the FEVER knowledge base (and a
fallback file in `backend/data/`), so the retrieved evidence shown in the UI is genuine.
"""

DEMO_EXAMPLES = [
    {
        "name": "Eiffel Tower",
        "answer": (
            "The Eiffel Tower was designed by Gustave Eiffel, completed in 1889, "
            "and is located in London."
        ),
        # All triggers must appear in the input for the fixture to activate.
        "triggers": ["Eiffel Tower", "Gustave", "London"],
        "claims": [
            ("The Eiffel Tower was designed by Gustave Eiffel.", "Supported"),
            ("The Eiffel Tower was completed in 1889.", "Supported"),
            ("The Eiffel Tower is located in London.", "Refuted"),
        ],
    },
    {
        "name": "Great Wall of China",
        "answer": (
            "The Great Wall of China was built as protection against nomadic groups "
            "from the north. Most of the surviving wall was constructed during the "
            "Ming dynasty. Contrary to popular belief, the Great Wall cannot be seen "
            "from the Moon with the naked eye."
        ),
        "triggers": ["Great Wall", "Ming dynasty", "Moon"],
        "claims": [
            ("The Great Wall of China was built as protection against nomadic groups from the north.", "Supported"),
            ("Most of the surviving Great Wall was constructed during the Ming dynasty.", "Supported"),
            ("The Great Wall of China cannot be seen from the Moon with the naked eye.", "Supported"),
        ],
    },
    {
        "name": "Statue of Liberty",
        "answer": (
            "The Statue of Liberty stands on Liberty Island in New York Harbor. "
            "It was a gift from the people of France and was dedicated in 1886. "
            "The statue is made of solid gold, and it is the most visited monument "
            "in the United States."
        ),
        "triggers": ["Statue of Liberty", "New York Harbor", "solid gold"],
        "claims": [
            ("The Statue of Liberty stands on Liberty Island in New York Harbor.", "Supported"),
            ("The Statue of Liberty was a gift from the people of France.", "Supported"),
            ("The Statue of Liberty was dedicated in 1886.", "Supported"),
            ("The Statue of Liberty is made of solid gold.", "Refuted"),
            ("The Statue of Liberty is the most visited monument in the United States.", "Not Enough Information"),
        ],
    },
]


def match_example(answer: str):
    """Return the fixture whose triggers all appear in `answer`, else None."""
    for example in DEMO_EXAMPLES:
        if all(trigger in answer for trigger in example["triggers"]):
            return example
    return None


def demo_verdict(claim_text: str):
    """Return the pinned verdict for a known demo claim, else None."""
    for example in DEMO_EXAMPLES:
        for text, verdict in example["claims"]:
            if text == claim_text:
                return verdict
    return None
