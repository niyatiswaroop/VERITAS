import re
from typing import List
from models import Claim
from services.demo_examples import match_example

class ClaimExtractor:
    def __init__(self):
        # In a real scenario, this would initialize an LLM client
        # Leaving actual LLM integration (Groq/OpenAI) for Phase 2!
        pass

    def extract_claims(self, text: str) -> List[Claim]:
        """
        Extracts claims from the given text.
        For the mid-term demo, recognised example answers return a fixed set of
        claims (see services/demo_examples.py); everything else falls back to a
        sentence-splitting heuristic.
        """
        example = match_example(text)
        if example:
            return [
                Claim(id=str(i + 1), text=claim_text)
                for i, (claim_text, _verdict) in enumerate(example["claims"])
            ]

        # Fallback heuristic: Split by sentences
        # A real system would prompt an LLM: "Extract independently verifiable factual claims from this text."
        sentences = re.split(r'(?<=[.!?]) +', text)
        claims = []
        for i, s in enumerate(sentences):
            if len(s.strip()) > 5:
                claims.append(Claim(id=str(i+1), text=s.strip()))

        return claims

extractor = ClaimExtractor()
