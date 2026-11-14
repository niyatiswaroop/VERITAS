from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import VerificationRequest, VerificationResponse, VerificationResult
from services.llm_extractor import extractor
from services.retriever import retriever
from services.nli_verifier import verifier
import uvicorn

app = FastAPI(title="VERITAS Phase 1 API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_answer(answer: str) -> str:
    """Reject input that isn't a verifiable answer before running the pipeline.

    Keeps the demo honest: fragments and non-text can't produce meaningful
    claim-level verdicts, so we ask for a proper answer instead of scoring noise.
    """
    cleaned = (answer or "").strip()
    if not any(ch.isalpha() for ch in cleaned):
        raise HTTPException(status_code=422, detail="Please enter an answer with some text to verify.")
    if len(cleaned) < 10 or len(cleaned.split()) < 3:
        raise HTTPException(
            status_code=422,
            detail="Please enter a complete answer (a full sentence) to verify, not a short phrase.",
        )
    return cleaned


@app.post("/verify", response_model=VerificationResponse)
def verify_answer(request: VerificationRequest):
    answer = validate_answer(request.generated_answer)

    # 1. Extract Claims
    claims, claim_extractor = extractor.extract_claims(answer)
    
    verified_claims = []
    supported_count = 0
    refuted_count = 0
    nei_count = 0
    
    # 2. For each claim, retrieve evidence and verify
    for claim in claims:
        # Retrieve evidence
        evidence = retriever.retrieve(claim.text)
        
        # Verify claim
        verdict, probabilities, source, decisive_span_id = verifier.verify(claim.text, evidence)
        
        if verdict == "Supported":
            supported_count += 1
        elif verdict == "Refuted":
            refuted_count += 1
        else:
            nei_count += 1
            
        verified_claims.append(VerificationResult(
            claim_id=claim.id,
            claim_text=claim.text,
            verdict=verdict,
            evidence=evidence,
            nli_probabilities=probabilities,
            verdict_source=source,
            decisive_span_id=decisive_span_id
        ))
        
    total_claims = len(claims)
    
    # 3. Calculate Scores
    if total_claims > 0:
        faithfulness_score = (supported_count / total_claims) * 100
        hallucination_score = ((refuted_count + nei_count) / total_claims) * 100
    else:
        faithfulness_score = 0.0
        hallucination_score = 0.0
        
    # 4. Final Classification
    if faithfulness_score == 100:
        overall_class = "Supported"
    elif refuted_count > 0:
        if supported_count > 0:
            overall_class = "Partially Supported / Has Refutations"
        else:
            overall_class = "Refuted"
    elif nei_count > 0 and supported_count > 0:
        overall_class = "Partially Supported"
    elif nei_count == total_claims and total_claims > 0:
        overall_class = "Not Enough Information"
    else:
        overall_class = "Unknown"

    return VerificationResponse(
        claims_verified=verified_claims,
        hallucination_score=round(hallucination_score, 2),
        faithfulness_score=round(faithfulness_score, 2),
        overall_classification=overall_class,
        claim_extractor=claim_extractor
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
