import { useState } from 'react';
import axios from 'axios';
import { ShieldCheck, Loader2, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';
import './index.css';

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

  const examples = [
    {
      label: 'Eiffel Tower',
      text: 'The Eiffel Tower was designed by Gustave Eiffel, completed in 1889, and is located in London.',
    },
    {
      label: 'Great Wall',
      text: 'The Great Wall of China was built as protection against nomadic groups from the north. Most of the surviving wall was constructed during the Ming dynasty. Contrary to popular belief, the Great Wall cannot be seen from the Moon with the naked eye.',
    },
    {
      label: 'Statue of Liberty',
      text: 'The Statue of Liberty stands on Liberty Island in New York Harbor. It was a gift from the people of France and was dedicated in 1886. The statue is made of solid gold, and it is the most visited monument in the United States.',
    },
  ];

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
              <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>Contradicted + Unsupported</p>
            </div>
          </div>

          <div className="glass-panel" style={{ marginTop: '12px' }}>
            <h2 style={{ borderBottom: '1px solid #334155', paddingBottom: '16px', marginBottom: '24px' }}>Claim-Level Breakdown</h2>
            <div className="claims-list">
              {results.claims_verified.map((claim, idx) => (
                <div key={idx} className={`glass-panel claim-item ${claim.verdict.toLowerCase()}`}>
                  <div className="claim-header">
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                      {claim.verdict === 'Supported' && <CheckCircle2 color="#10b981" style={{ marginTop: '2px' }} />}
                      {claim.verdict === 'Contradicted' && <XCircle color="#ef4444" style={{ marginTop: '2px' }} />}
                      {claim.verdict === 'Unsupported' && <AlertTriangle color="#f59e0b" style={{ marginTop: '2px' }} />}
                      <div className="claim-text">{claim.claim_text}</div>
                    </div>
                    <div className={`badge ${claim.verdict.toLowerCase()}`}>
                      {claim.verdict}
                    </div>
                  </div>
                  
                  <div style={{ marginTop: '16px' }}>
                    <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Retrieved Evidence</div>
                    <div className="evidence-box">
                      {claim.evidence || "No relevant evidence found in the knowledge base."}
                    </div>
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
