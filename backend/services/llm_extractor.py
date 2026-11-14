"""Stage 1 -- Claim Decomposition (report §4.1, §5.2 item 1; Phase 2, item 2.3).

An LLM breaks the answer into atomic, self-contained claims:
  - FActScore (Min et al., 2023, Appendix A.2 / Table 15): few-shot prompt to "break down
    the following sentence into independent facts", one piece of information per fact.
  - FENICE (Scirè et al., 2024, Appendix A): "use a noun as the subject in the claim (try to
    avoid pronouns)" and "do not generate any novel word, be faithful to the provided input"
    -- this is what makes each claim verifiable on its own (report §4.1 Stage 1).
The whole answer is sent in one call (as FENICE does) so pronouns can be resolved from
earlier sentences; FActScore's per-sentence split would lose that context.

Model: open-weights LLM via Groq (default openai/gpt-oss-120b) (GROQ_API_KEY, optional GROQ_MODEL in backend/.env). Without a key,
or if the call fails, the prototype's regex sentence split is used instead.
"""
import json
import os
import re
from typing import List, Tuple

from dotenv import load_dotenv

from models import Claim
from services.demo_examples import match_example

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

DEFAULT_MODEL = "openai/gpt-oss-120b"  # open-weights; Groq offers no Llama chat model on this account
EXTRACTOR_LLM = "llm"
EXTRACTOR_REGEX = "regex_fallback"
EXTRACTOR_PINNED_DEMO = "pinned_demo"

SYSTEM_PROMPT = """You break a text into atomic factual claims for fact-checking.

Rules:
- Each claim is a short sentence conveying exactly one piece of information.
- Each claim must be understandable on its own: use a noun as the subject and replace pronouns and vague references ("it", "he", "the band", "this") with the entity they refer to.
- Do not generate any novel information; be faithful to the provided text.
- Every fact expressed in the text must appear in the output.
- Skip opinions, greetings and text with no factual content.

Reply with JSON only: {"claims": ["...", "..."]}"""

# Worked examples adapted from FActScore Table 15 (with pronouns resolved per FENICE).
FEW_SHOT = [
    (
        "Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot "
        "who was the Command Module Pilot for the Apollo 11 mission in 1969.",
        ["Michael Collins was born on October 31, 1930.", "Michael Collins is retired.",
         "Michael Collins is American.", "Michael Collins was an astronaut.",
         "Michael Collins was a test pilot.",
         "Michael Collins was the Command Module Pilot for the Apollo 11 mission.",
         "The Apollo 11 mission took place in 1969."],
    ),
    (
        "Love and Destiny is a romantic comedy series. It premiered in 2019, and its lead actress "
        "also starred in two films.",
        ["Love and Destiny is a romantic comedy series.", "Love and Destiny premiered in 2019.",
         "The lead actress of Love and Destiny starred in two films."],
    ),
]


def regex_split(text: str) -> List[str]:
    """Prototype fallback: one claim per sentence."""
    return [s.strip() for s in re.split(r"(?<=[.!?]) +", text) if len(s.strip()) > 5]


class ClaimExtractor:
    def __init__(self):
        self.model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.client = None
        if os.getenv("GROQ_API_KEY"):
            from groq import Groq
            self.client = Groq()
            print(f"Stage 1: LLM claim decomposition with {self.model} (Groq).")
        else:
            print("Stage 1: no GROQ_API_KEY set; using regex sentence split.")

    def _llm_claims(self, text: str) -> List[str]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for example_text, example_claims in FEW_SHOT:
            messages.append({"role": "user", "content": example_text})
            messages.append({"role": "assistant", "content": json.dumps({"claims": example_claims})})
        messages.append({"role": "user", "content": text})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        claims = json.loads(response.choices[0].message.content).get("claims", [])
        return [c.strip() for c in claims if isinstance(c, str) and c.strip()]

    def extract_claims(self, text: str) -> Tuple[List[Claim], str]:
        """Return (claims, extractor) where extractor is one of the EXTRACTOR_* values.

        Recognised mid-term demo answers keep their fixed claim list (services/demo_examples.py).
        """
        example = match_example(text)
        if example:
            return [Claim(id=str(i + 1), text=claim_text)
                    for i, (claim_text, _verdict) in enumerate(example["claims"])], EXTRACTOR_PINNED_DEMO

        if self.client:
            try:
                claims = self._llm_claims(text)
                if claims:
                    return [Claim(id=str(i + 1), text=c) for i, c in enumerate(claims)], EXTRACTOR_LLM
            except Exception as e:
                print(f"LLM claim extraction failed, using regex fallback: {e}")

        return [Claim(id=str(i + 1), text=s) for i, s in enumerate(regex_split(text))], EXTRACTOR_REGEX


extractor = ClaimExtractor()
