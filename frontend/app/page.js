'use client';

import { useState } from 'react';
import styles from './page.module.css';

export default function Home() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('https://prats010-kronector.hf.space/predict/f1', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        throw new Error(`API Error: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to fetch prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className={styles.main}>
      <div className={styles.header}>
        <h1 className={`${styles.title} text-gradient`}>KRONECTOR</h1>
        <p className={styles.subtitle}>F1 Intelligence Terminal. Powered by LLMs and LightGBM.</p>
      </div>

      <div className={styles.queryContainer}>
        <form onSubmit={handleSubmit} className={styles.inputWrapper}>
          <input
            type="text"
            className={styles.input}
            placeholder="Ask anything (e.g., 'What's Max Verstappen's win probability at Monaco 2023?')"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className={styles.submitBtn} disabled={loading || !query.trim()}>
            {loading ? <div className={styles.loader} /> : (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            )}
          </button>
        </form>
        {error && <div style={{ color: 'var(--neon-red)', marginTop: '1rem', textAlign: 'center' }}>{error}</div>}
      </div>

      {result && (
        <div className={`${styles.dashboardGrid} animate-fade-in-up`}>
          {/* Left Column: Gauge */}
          <div className={`${styles.panel} glass-panel`}>
            <h2 className={styles.panelTitle}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
              Win Probability
            </h2>
            <div className={styles.gaugeContainer}>
              <div 
                className={styles.gaugeCircle} 
                style={{ '--prob': `${(result.win_probability * 100).toFixed(1)}%` }}
              >
                <span className={styles.gaugeText}>
                  {(result.win_probability * 100).toFixed(1)}<span style={{fontSize: '1.5rem', color: 'var(--text-secondary)'}}>%</span>
                </span>
              </div>
              <div className={styles.gaugeLabel}>
                {result.metadata?.driver_name || 'Driver'} @ {result.metadata?.season} Round {result.metadata?.round}
              </div>
              {result.confidence_rating && (
                <div className={`${styles.rating} ${result.confidence_rating.includes('Good') ? styles.ratingGood : styles.ratingBad}`}>
                  Data Confidence: {result.confidence_rating}
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Insight & SHAP */}
          <div className={`${styles.panel} glass-panel delay-100`} style={{ gap: '2rem' }}>
            <div>
              <h2 className={styles.panelTitle}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                AI Insight
              </h2>
              <p className={styles.insightText}>
                {result.llm_explanation}
              </p>
            </div>

            <div>
              <h2 className={styles.panelTitle}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                Key Factors (SHAP)
              </h2>
              <div className={styles.shapContainer}>
                {result.shap_values && Object.entries(result.shap_values)
                  .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
                  .slice(0, 6)
                  .map(([key, value]) => {
                    const absVal = Math.abs(value);
                    const isPositive = value > 0;
                    // Normalize to max 50% width
                    const width = Math.min((absVal / 2) * 100, 50);
                    
                    return (
                      <div key={key} className={styles.shapRow}>
                        <div className={styles.shapLabel} title={key}>
                          {key.replace(/_/g, ' ')}
                        </div>
                        <div className={styles.barTrack}>
                          <div 
                            className={`${styles.barFill} ${isPositive ? styles.barPositive : styles.barNegative}`}
                            style={{ width: `${width}%` }}
                          />
                        </div>
                        <div className={`${styles.shapValue} ${isPositive ? 'text-gradient' : ''}`} style={!isPositive ? {color: 'var(--neon-red)'} : {}}>
                          {value > 0 ? '+' : ''}{value.toFixed(3)}
                        </div>
                      </div>
                    );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
