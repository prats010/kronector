'use client';

import { useState } from 'react';
import styles from './page.module.css';

export default function Home() {
  const [mode, setMode] = useState('predict'); // 'predict' | 'compare'
  
  // Single predict state
  const [query, setQuery] = useState('');
  
  // Compare state
  const [driver1, setDriver1] = useState('');
  const [driver2, setDriver2] = useState('');
  const [season, setSeason] = useState('');
  const [round, setRound] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmitPredict = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    await fetchResult('predict/f1', { query });
  };

  const handleSubmitCompare = async (e) => {
    e.preventDefault();
    if (!driver1.trim() || !driver2.trim()) return;
    const body = { driver1, driver2 };
    if (season) body.season = parseInt(season);
    if (round) body.round = parseInt(round);
    await fetchResult('predict/compare', body);
  };

  const fetchResult = async (endpoint, body) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`https://prats010-kronector.hf.space/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `API Error: ${res.status}`);
      }
      const data = await res.json();
      setResult({ ...data, _type: endpoint });
    } catch (err) {
      setError(err.message || 'Failed to fetch prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const renderShapBars = (shapValues, isCompare = false) => {
    if (!shapValues) return null;
    return Object.entries(shapValues)
      .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
      .slice(0, 6)
      .map(([key, value]) => {
        const absVal = Math.abs(value);
        const isPositive = value > 0;
        
        let width = Math.min((absVal / 2) * 100, 50);
        if (isCompare) {
            // For delta: positive = driver 1 (green, left side), negative = driver 2 (red, right side)
            // Wait, standard UI: positive is driver 1.
            width = Math.min((absVal / 5) * 100, 50); // Scale down slightly for deltas since they can be larger
        }

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
      });
  };

  return (
    <main className={styles.main}>
      <div className={styles.header}>
        <h1 className={`${styles.title} text-gradient`}>KRONECTOR</h1>
        <p className={styles.subtitle}>F1 Intelligence Terminal. Powered by LLMs and LightGBM.</p>
      </div>

      <div className={styles.tabsContainer}>
        <button 
          className={`${styles.tabBtn} ${mode === 'predict' ? styles.tabActive : ''}`}
          onClick={() => { setMode('predict'); setResult(null); setError(null); }}
        >
          Single Prediction
        </button>
        <button 
          className={`${styles.tabBtn} ${mode === 'compare' ? styles.tabActive : ''}`}
          onClick={() => { setMode('compare'); setResult(null); setError(null); }}
        >
          Head-to-Head Compare
        </button>
      </div>

      <div className={styles.queryContainer}>
        {mode === 'predict' ? (
          <form onSubmit={handleSubmitPredict} className={styles.inputWrapper}>
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
        ) : (
          <form onSubmit={handleSubmitCompare} className={styles.compareForm}>
            <div className={styles.compareInputsRow}>
              <div className={styles.inputWrapper}>
                  <input type="text" className={styles.input} placeholder="Driver 1 (e.g., Max Verstappen)" value={driver1} onChange={e=>setDriver1(e.target.value)} required disabled={loading} />
              </div>
              <span className={styles.vsText}>VS</span>
              <div className={styles.inputWrapper}>
                  <input type="text" className={styles.input} placeholder="Driver 2 (e.g., Kimi Antonelli)" value={driver2} onChange={e=>setDriver2(e.target.value)} required disabled={loading} />
              </div>
            </div>
            <div className={styles.compareInputsRowSmall}>
              <div className={styles.inputWrapper} style={{maxWidth: '200px'}}>
                  <input type="number" className={styles.input} style={{padding: '1rem'}} placeholder="Season (Optional)" value={season} onChange={e=>setSeason(e.target.value)} disabled={loading} />
              </div>
              <div className={styles.inputWrapper} style={{maxWidth: '200px'}}>
                  <input type="number" className={styles.input} style={{padding: '1rem'}} placeholder="Round (Optional)" value={round} onChange={e=>setRound(e.target.value)} disabled={loading} />
              </div>
            </div>
            <button type="submit" className={styles.compareSubmitBtn} disabled={loading || !driver1.trim() || !driver2.trim()}>
              {loading ? 'Analyzing...' : 'Run Head-to-Head Comparison'}
            </button>
          </form>
        )}
        {error && <div style={{ color: 'var(--neon-red)', marginTop: '1rem', textAlign: 'center' }}>{error}</div>}
      </div>

      {result && result._type === 'predict/f1' && (
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
                style={{ '--prob': `${result.win_probability.toFixed(1)}%` }}
              >
                <span className={styles.gaugeText}>
                  {result.win_probability.toFixed(1)}<span style={{fontSize: '1.5rem', color: 'var(--text-secondary)'}}>%</span>
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
                {renderShapBars(result.shap_values)}
              </div>
            </div>
          </div>
        </div>
      )}

      {result && result._type === 'predict/compare' && (
        <div className={`${styles.dashboardGrid} animate-fade-in-up`} style={{gridTemplateColumns: '1fr'}}>
          
          <div className={`${styles.panel} glass-panel`} style={{marginBottom: '2rem'}}>
            <h2 className={styles.panelTitle} style={{justifyContent: 'center', fontSize: '1.5rem'}}>
              {result.driver1.driver_name} vs {result.driver2.driver_name}
            </h2>
            <div className={styles.gaugeContainer} style={{flexDirection: 'row', gap: '4rem'}}>
              {/* Driver 1 */}
              <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                <div className={styles.gaugeCircle} style={{ '--prob': `${result.driver1_win_probability.toFixed(1)}%` }}>
                  <span className={styles.gaugeText}>
                    {result.driver1_win_probability.toFixed(1)}<span style={{fontSize: '1.5rem', color: 'var(--text-secondary)'}}>%</span>
                  </span>
                </div>
                <div className={styles.gaugeLabel}>{result.driver1.driver_name}</div>
              </div>
              
              {/* VS */}
              <div className={styles.vsText} style={{fontSize: '2rem'}}>VS</div>

              {/* Driver 2 */}
              <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                <div className={styles.gaugeCircle} style={{ '--prob': `${result.driver2_win_probability.toFixed(1)}%`, '--neon-cyan': 'var(--neon-red)' }}>
                  <span className={styles.gaugeText}>
                    {result.driver2_win_probability.toFixed(1)}<span style={{fontSize: '1.5rem', color: 'var(--text-secondary)'}}>%</span>
                  </span>
                </div>
                <div className={styles.gaugeLabel}>{result.driver2.driver_name}</div>
              </div>
            </div>
          </div>

          <div className={`${styles.dashboardGrid}`} style={{gap: '2rem'}}>
            {/* LLM Insight */}
            <div className={`${styles.panel} glass-panel delay-100`}>
              <h2 className={styles.panelTitle}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                Tale of the Tape
              </h2>
              <div className={styles.insightText} style={{whiteSpace: 'pre-wrap'}}>
                {result.llm_analysis}
              </div>
            </div>

            {/* SHAP Deltas */}
            <div className={`${styles.panel} glass-panel delay-200`}>
              <h2 className={styles.panelTitle}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                Mathematical Edges (Tug of War)
              </h2>
              <div className={styles.shapContainer}>
                {renderShapBars(result.shap_deltas, true)}
                <div style={{display: 'flex', justifyContent: 'space-between', marginTop: '1rem', color: 'var(--text-secondary)', fontSize: '0.8rem'}}>
                    <span>← Advantage {result.driver2.driver_name}</span>
                    <span>Advantage {result.driver1.driver_name} →</span>
                </div>
              </div>
            </div>
          </div>
          
        </div>
      )}
    </main>
  );
}
