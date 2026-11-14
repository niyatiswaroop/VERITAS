import { useState } from 'react';
import axios from 'axios';
import { ShieldCheck, Loader2, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';
import './index.css';

// CSS class per verdict label ("Not Enough Information" has spaces, so it can't be used directly).
const VERDICT_CLASS = {
  'Supported': 'supported',
  'Refuted': 'refuted',
  'Not Enough Information': 'nei',
};

// Tag shown when a verdict did not come from the NLI model (report §5.1: pinned
// demo verdicts must not be read as model output).
const VERDICT_SOURCE_TAG = {
  pinned_demo: 'Pinned demo verdict, model not run',
  no_evidence: 'No evidence retrieved',
  model_unavailable: 'NLI model unavailable, model not run',
};

function App() {
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const handleVerify = async () => {
    if (!answer.trim()) return;
    
    setIsLoading(true);
    setError('');
    setResults(null);

    try {
      const response = await axios.post('http://localhost:8000/verify', {
        generated_answer: answer
      });
      setResults(response.data);
    } catch (err) {
      console.error(err);
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === 'string'
          ? detail
          : 'Failed to connect to the verification server. Ensure the backend is running.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Examples drawn from the FEVER knowledge base (Phase 2, item 2.1). They run through the
  // real pipeline (nothing pinned); `expected` is FEVER's gold label, `note` says how the
  // current pipeline does on it, so wrong verdicts are visible rather than hidden.
  const examples = [
    {
      label: 'Chile',
      text: 'Chile is in South America.',
      expected: 'Supported',
      note: 'Currently correct.',
    },
    {
      label: 'Starrcade',
      text: 'Starrcade was an annual professional wrestling event.',
      expected: 'Supported',
      note: 'Currently correct.',
    },
    {
      label: 'Bermuda Triangle',
      text: 'Bermuda Triangle is a loosely-defined region in the Pacific Ocean.',
      expected: 'Refuted',
      note: 'Currently correct (the evidence says Atlantic).',
    },
    {
      label: 'Buffy Summers',
      text: 'Buffy Summers appears only in a television series.',
      expected: 'Refuted',
      note: 'Currently correct.',
    },
    {
      label: 'Omar Khadr',
      text: 'Omar Khadr was interrogated by llamas.',
      expected: 'Not Enough Information',
      note: 'Currently correct (no relevant evidence retrieved).',
    },
    {
      label: 'Aarhus',
      text: 'Aarhus is the second-largest city in Germany.',
      expected: 'Refuted',
      note: 'Currently correct (fixed by scoring each sentence separately).',
    },
    {
      label: 'DJ Quik',
      text: 'DJ Quik is from the United States.',
      expected: 'Not Enough Information',
      note: 'Currently correct (fixed by the relevance check: a sentence about a different person can no longer decide).',
    },
    {
      label: 'Janelle Monáe',
      text: "Janelle Monáe's dog's name is Charles Barkley.",
      expected: 'Not Enough Information',
      note: 'Known failure: "not mentioned" is still read as a contradiction (Refuted). Small-model limitation; larger NLI model under test.',
    },
    {
      label: 'Polynesian languages',
      text: 'Polynesian languages include several speakers.',
      expected: 'Not Enough Information',
      note: 'Currently correct (fixed by the tuned thresholds).',
    },
    {
      label: 'Statue of Liberty (2 claims)',
      text: 'The Statue of Liberty was a gift from France. It is made of copper.',
      expected: 'Not a FEVER claim (both facts are true)',
      note: 'Currently correct: Stage 1 rewrites "It is made of copper" as "The Statue of Liberty is made of copper", and the page title lets the model link "The copper statue" to the claim.',
    },
    {
      label: 'Queen (band)',
      text: 'Queen is a British rock band. They formed in London in 1970.',
      expected: 'Both facts are true (FEVER: Supported)',
      note: 'Known failure: claim 2 comes out Refuted because another Queen sentence ("first charted in 1974") is misread as a contradiction, and the report rule checks contradictions first (rule A vs B question for the mentor).',
    },
    // Mid-eval demo answers (report §5.1): verdicts are pinned in the backend, the model is not run.
    {
      label: 'Eiffel Tower (pinned demo)',
      text: 'The Eiffel Tower was designed by Gustave Eiffel, completed in 1889, and is located in London.',
      expected: 'Pinned demo verdicts',
      note: 'Mid-eval demo (report §5.1): verdicts are fixed in the backend; the NLI model is not run.',
    },
    {
      label: 'Great Wall (pinned demo)',
      text: 'The Great Wall of China was built as protection against nomadic groups from the north. Most of the surviving wall was constructed during the Ming dynasty. Contrary to popular belief, the Great Wall cannot be seen from the Moon with the naked eye.',
      expected: 'Pinned demo verdicts',
      note: 'Mid-eval demo (report §5.1): verdicts are fixed in the backend; the NLI model is not run.',
    },
    {
      label: 'Statue of Liberty (pinned demo)',
      text: 'The Statue of Liberty stands on Liberty Island in New York Harbor. It was a gift from the people of France and was dedicated in 1886. The statue is made of solid gold, and it is the most visited monument in the United States.',
      expected: 'Pinned demo verdicts',
      note: 'Mid-eval demo (report §5.1): verdicts are fixed in the backend; the NLI model is not run.',
    },
  ];
  const selectedExample = examples.find((ex) => ex.text === answer);

  return (
    <div className="fade-in">
      <h1 className="title">VERITAS</h1>
      <p className="subtitle">Verified Evidence-based Reliability & Interpretable Trust Assessment System</p>

      <div className="glass-panel">
        <div className="input-area">
          <label style={{ fontWeight: 600, color: '#e2e8f0' }}>LLM Generated Answer</label>
          <textarea 
            className="textarea"
            placeholder="Paste the LLM answer here to verify its factual claims..."
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {examples.map((ex) => (
                <button
                  key={ex.label}
                  onClick={() => setAnswer(ex.text)}
                  style={{ background: 'transparent', border: '1px solid #475569', color: '#cbd5e1', padding: '8px 16px', borderRadius: '6px', cursor: 'pointer' }}
                >
                  {ex.label}
                </button>
              ))}
            </div>
            <button
              className="btn"
              onClick={handleVerify}
              disabled={isLoading || !answer.trim()}
            >
              {isLoading ? <Loader2 className="spinner" size={20} /> : <ShieldCheck size={20} />}
              Verify Claims
            </button>
          </div>
          {selectedExample && (
            <div className="example-note">
              <strong>FEVER label:</strong> {selectedExample.expected} · {selectedExample.note}
            </div>
          )}
          {error && <p style={{ color: '#ef4444', marginTop: '8px' }}>{error}</p>}
        </div>
      </div>

      {results && (
        <div className="dashboard fade-in">
          <div className="scores-grid">
            <div className="glass-panel score-card">
              <h3 style={{ color: '#94a3b8', margin: 0 }}>Overall Status</h3>
              <div style={{ fontSize: '1.5rem', fontWeight: 600, marginTop: '16px', color: '#f8fafc' }}>
                {results.overall_classification}
              </div>
            </div>
            <div className="glass-panel score-card">
              <h3 style={{ color: '#94a3b8', margin: 0 }}>Faithfulness Score</h3>
              <div className={`score-value ${results.faithfulness_score >= 80 ? 'high' : results.faithfulness_score >= 50 ? 'medium' : 'low'}`}>
                {results.faithfulness_score}%
              </div>
              <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>Supported Claims / Total</p>
            </div>
            <div className="glass-panel score-card">
              <h3 style={{ color: '#94a3b8', margin: 0 }}>Hallucination Score</h3>
              <div className={`score-value ${results.hallucination_score <= 20 ? 'high' : results.hallucination_score <= 50 ? 'medium' : 'low'}`}>
                {results.hallucination_score}%
              </div>
              <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>Refuted + Not Enough Information</p>
            </div>
          </div>

          <div className="glass-panel" style={{ marginTop: '12px' }}>
            <h2 style={{ borderBottom: '1px solid #334155', paddingBottom: '16px', marginBottom: '24px' }}>
              Claim-Level Breakdown
              <span className="source-tag" style={{ marginLeft: '12px', verticalAlign: 'middle' }}>
                {{
                  llm: 'Claims split by LLM (Stage 1)',
                  regex_fallback: 'Claims split by sentence (no LLM key set)',
                  pinned_demo: 'Pinned demo claims',
                }[results.claim_extractor]}
              </span>
            </h2>
            <div className="claims-list">
              {results.claims_verified.map((claim, idx) => (
                <div key={idx} className={`glass-panel claim-item ${VERDICT_CLASS[claim.verdict]}`}>
                  <div className="claim-header">
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                      {claim.verdict === 'Supported' && <CheckCircle2 color="#10b981" style={{ marginTop: '2px' }} />}
                      {claim.verdict === 'Refuted' && <XCircle color="#ef4444" style={{ marginTop: '2px' }} />}
                      {claim.verdict === 'Not Enough Information' && <AlertTriangle color="#f59e0b" style={{ marginTop: '2px' }} />}
                      <div>
                        <div className="claim-text">{claim.claim_text}</div>
                        {VERDICT_SOURCE_TAG[claim.verdict_source] && (
                          <div className="source-tag">{VERDICT_SOURCE_TAG[claim.verdict_source]}</div>
                        )}
                      </div>
                    </div>
                    <div className={`badge ${VERDICT_CLASS[claim.verdict]}`}>
                      {claim.verdict}
                    </div>
                  </div>
                  
                  <div style={{ marginTop: '16px' }}>
                    <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Retrieved Evidence</div>
                    <div className="evidence-box">
                      {claim.evidence.length === 0
                        ? "No relevant evidence found in the knowledge base."
                        : claim.evidence.map((span) => (
                          <div
                            key={span.span_id}
                            className={`evidence-span ${span.span_id === claim.decisive_span_id ? 'decisive' : ''}`}
                          >
                            <div className="evidence-source">
                              {span.page} · <code>{span.span_id}</code>
                              {span.span_id === claim.decisive_span_id && <span className="decisive-tag">decided the verdict</span>}
                            </div>
                            <div>{span.text}</div>
                            {span.nli_probabilities && (
                              <div className="span-scores">
                                E {span.nli_probabilities.entailment.toFixed(3)} · C {span.nli_probabilities.contradiction.toFixed(3)} · N {span.nli_probabilities.neutral.toFixed(3)}
                              </div>
                            )}
                          </div>
                        ))}
                    </div>
                    {claim.nli_probabilities && (
                      <div style={{ marginTop: '8px', fontSize: '0.85rem', color: '#94a3b8', fontFamily: 'monospace' }}>
                        NLI ({claim.decisive_span_id ? 'deciding sentence' : 'closest sentence'}): Entailment {claim.nli_probabilities.entailment.toFixed(3)} · Contradiction {claim.nli_probabilities.contradiction.toFixed(3)} · Neutral {claim.nli_probabilities.neutral.toFixed(3)}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
