from pydantic import BaseModel
from typing import List, Optional

# Sentinel returned by the retriever when no chunk clears the similarity threshold.
# The NLI verifier and the API both check for this to short-circuit to "Unsupported".
NO_EVIDENCE_MESSAGE = "No relevant evidence found in the knowledge base."

class VerificationRequest(BaseModel):
    question: Optional[str] = None
    generated_answer: str

class Claim(BaseModel):
    id: str
    text: str

class VerificationResult(BaseModel):
    claim_id: str
    claim_text: str
    verdict: str  # "Supported", "Contradicted", "Unsupported"
    evidence: str

class VerificationResponse(BaseModel):
    claims_verified: List[VerificationResult]
    hallucination_score: float
    faithfulness_score: float
    overall_classification: str
