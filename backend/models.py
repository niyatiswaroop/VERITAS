from pydantic import BaseModel
from typing import List, Optional

# Where a claim's verdict came from, so pinned demo verdicts (report §5.1) are never
# mistaken for model output.
VERDICT_SOURCE_NLI_MODEL = "nli_model"
VERDICT_SOURCE_PINNED_DEMO = "pinned_demo"
VERDICT_SOURCE_NO_EVIDENCE = "no_evidence"
VERDICT_SOURCE_MODEL_UNAVAILABLE = "model_unavailable"

class VerificationRequest(BaseModel):
    question: Optional[str] = None
    generated_answer: str

class Claim(BaseModel):
    id: str
    text: str

class NLIProbabilities(BaseModel):
    # Three-way NLI class probabilities (report §4.1 Stage 4, Table 4.1 f(a,b)).
    entailment: float
    contradiction: float
    neutral: float

class EvidenceSpan(BaseModel):
    # One retrieved knowledge-base sentence. span_id is "<FEVER page id>::<sentence index>",
    # the stable id Stage 5 stores as evidence_span_ids (report §4.3).
    span_id: str
    page: str
    text: str
    distance: float
    # This sentence's own NLI scores against the claim; None when the model did not run.
    nli_probabilities: Optional[NLIProbabilities] = None

class VerificationResult(BaseModel):
    claim_id: str
    claim_text: str
    verdict: str  # "Supported", "Refuted", "Not Enough Information" (FEVER labels, report §3.1 FR-4)
    evidence: List[EvidenceSpan]  # empty when nothing relevant was retrieved
    # Scores of the sentence that decided the verdict (closest sentence for NEI);
    # None when the NLI model did not run (no evidence found, or a pinned demo verdict).
    nli_probabilities: Optional[NLIProbabilities] = None
    verdict_source: str  # one of the VERDICT_SOURCE_* values above
    # Evidence sentence that decided a Supported/Refuted verdict; None otherwise.
    decisive_span_id: Optional[str] = None

class VerificationResponse(BaseModel):
    claims_verified: List[VerificationResult]
    hallucination_score: float
    faithfulness_score: float
    overall_classification: str
