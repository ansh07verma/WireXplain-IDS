import { useState, useEffect, useRef } from 'react'
import { usePipelineStatus } from '../hooks/useApi'
import MetricCard from '../components/MetricCard'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import {
  Terminal, Play, CheckCircle, XCircle,
  Cpu, Activity, Target, Zap, BarChart2, Database
} from 'lucide-react'
import './Pipeline.css'

const DEFAULT_CONFIG = {
  n_estimators: 100,
  test_size: 0.2,
  contamination: 0.05,
  top_n_features: 15,
  dataset: 'cicids2018',
}

export default function Pipeline() {
  const pipelineStatus = usePipelineStatus()

  const [config, setConfig] = useState(DEFAULT_CONFIG)
  const [runId, setRunId] = useState(null)
  const [runStatus, setRunStatus] = useState(null) // "running"|"done"|"error"|null
  const [logs, setLogs] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [features, setFeatures] = useState(null)
  const [isStarting, setIsStarting] = useState(false)

  const logEndRef = useRef(null)
  const esRef = useRef(null)

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  // Load existing metrics + features on mount
  useEffect(() => {
    fetch('/api/pipeline/metrics')
      .then(r => r.ok ? r.json() : null)
      .then(m => { if (m) setMetrics(m) })
      .catch(() => {})

    fetch('/api/pipeline/features')
      .then(r => r.ok ? r.json() : null)
      .then(f => { if (f) setFeatures(f) })
      .catch(() => {})
  }, [])

  // Connect to SSE stream for a given run
  function connectStream(id) {
    if (esRef.current) esRef.current.close()

    const es = new EventSource(`/api/pipeline/stream/${id}`)
    esRef.current = es

    es.onmessage = (e) => {
      const item = JSON.parse(e.data)
      if (item.type === 'log') {
        setLogs(prev => [...prev, item.data])
      } else if (item.type === 'done') {
        setRunStatus('done')
        setMetrics(item.data)
        es.close()
        // Reload feature importance
        fetch('/api/pipeline/features')
          .then(r => r.ok ? r.json() : null)
          .then(f => { if (f) setFeatures(f) })
          .catch(() => {})
      } else if (item.type === 'error') {
        setRunStatus('error')
        setLogs(prev => [...prev, `❌ ${item.data}`])
        es.close()
      }
    }
    es.onerror = () => {
      if (runStatus !== 'done') setRunStatus('error')
      es.close()
    }
  }

  async function handleRun() {
    setIsStarting(true)
    setLogs([])
    setMetrics(null)
    setRunStatus('running')

    try {
      const res = await fetch('/api/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed to start pipeline')
      }
      const { run_id } = await res.json()
      setRunId(run_id)
      connectStream(run_id)
    } catch (err) {
      setRunStatus('error')
      setLogs(prev => [...prev, `❌ ${err.message}`])
    } finally {
      setIsStarting(false)
    }
  }

  const isRunning = runStatus === 'running'
  const isDone    = runStatus === 'done'
  const isError   = runStatus === 'error'
  const trained   = pipelineStatus?.any_model_trained || isDone
  const dataReady = pipelineStatus?.dataset_ready

  return (
    <div>
      <h1 className="page-title">Pipeline</h1>
      <p className="page-subtitle">Train ML models on CICIDS2018 — watch it happen live</p>

      {/* Dataset warning */}
      {!dataReady && (
        <div className="pipeline-alert pipeline-alert--warn">
          <Database size={16} />
          <span>
            <strong>Dataset missing.</strong> Download <code>02-14-2018.csv</code> from{' '}
            <a href="https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv" target="_blank" rel="noreferrer">
              Kaggle
            </a>{' '}
            and place it in <code>backend/data/raw/</code>
          </span>
        </div>
      )}

      <div className="pipeline-layout">
        {/* ── Left column: controls ── */}
        <div className="pipeline-left">

          {/* Config card */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="section-title" style={{ marginBottom: 16 }}>
              <Cpu size={15} color="var(--accent-cyan)" /> Configuration
            </div>

            <div className="pipeline-form">
              <label className="pipeline-label">
                <span>Dataset</span>
                <select
                  className="input"
                  value={config.dataset}
                  disabled={isRunning}
                  onChange={e => setConfig(p => ({ ...p, dataset: e.target.value }))}
                >
                  <option value="cicids2018">CICIDS2018</option>
                  <option value="unsw_nb15">UNSW-NB15</option>
                </select>
              </label>

              <label className="pipeline-label">
                <span>Trees (n_estimators)</span>
                <input
                  className="input"
                  type="number" min={10} max={500} step={10}
                  value={config.n_estimators}
                  disabled={isRunning}
                  onChange={e => setConfig(p => ({ ...p, n_estimators: +e.target.value }))}
                />
              </label>

              <label className="pipeline-label">
                <span>Test size</span>
                <input
                  className="input"
                  type="number" min={0.1} max={0.4} step={0.05}
                  value={config.test_size}
                  disabled={isRunning}
                  onChange={e => setConfig(p => ({ ...p, test_size: +e.target.value }))}
                />
              </label>

              <label className="pipeline-label">
                <span>Contamination (Isolation)</span>
                <input
                  className="input"
                  type="number" min={0.01} max={0.2} step={0.01}
                  value={config.contamination}
                  disabled={isRunning}
                  onChange={e => setConfig(p => ({ ...p, contamination: +e.target.value }))}
                />
              </label>

              <label className="pipeline-label">
                <span>Top N features (MI)</span>
                <input
                  className="input"
                  type="number" min={5} max={40} step={1}
                  value={config.top_n_features}
                  disabled={isRunning}
                  onChange={e => setConfig(p => ({ ...p, top_n_features: +e.target.value }))}
                />
              </label>
            </div>

            <button
              className={`btn btn-primary pipeline-run-btn ${isRunning ? 'btn--pulse' : ''}`}
              onClick={handleRun}
              disabled={isRunning || isStarting || !dataReady}
              style={{ width: '100%', marginTop: 16, justifyContent: 'center' }}
            >
              {isStarting ? (
                <LoadingSpinner size={14} text="Starting..." />
              ) : isRunning ? (
                <><LoadingSpinner size={14} /> Running pipeline...</>
              ) : (
                <><Play size={15} /> Run Pipeline</>
              )}
            </button>
          </div>

          {/* Model status */}
          <div className="card">
            <div className="section-title" style={{ marginBottom: 12 }}>
              <Activity size={15} color="var(--accent-cyan)" /> Model Status
            </div>
            {pipelineStatus ? (
              <div className="pipeline-model-list">
                {[
                  { key: 'random_forest',    label: 'RandomForest' },
                  { key: 'isolation_forest', label: 'IsolationForest' },
                  { key: 'xgboost',          label: 'XGBoost' },
                  { key: 'lightgbm',         label: 'LightGBM' },
                ].map(m => (
                  <div key={m.key} className="pipeline-model-row">
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                      {m.label}
                    </span>
                    <StatusBadge type={
                      (pipelineStatus.models[m.key] || (isDone && (m.key === 'random_forest' || m.key === 'isolation_forest')))
                        ? 'normal' : 'warning'
                    }>
                      {(pipelineStatus.models[m.key] || (isDone && (m.key === 'random_forest' || m.key === 'isolation_forest')))
                        ? 'Trained' : 'Not trained'}
                    </StatusBadge>
                  </div>
                ))}
              </div>
            ) : <LoadingSpinner text="Checking..." />}
          </div>
        </div>

        {/* ── Right column: logs + results ── */}
        <div className="pipeline-right">

          {/* Terminal log */}
          <div className="card pipeline-terminal-card">
            <div className="section-header">
              <div className="section-title">
                <Terminal size={15} color="var(--accent-cyan)" /> Live Log
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                {isRunning && <StatusBadge type="info"><span className="pulse">●</span> Running</StatusBadge>}
                {isDone    && <StatusBadge type="normal"><CheckCircle size={11} /> Complete</StatusBadge>}
                {isError   && <StatusBadge type="attack"><XCircle size={11} /> Failed</StatusBadge>}
                {runId     && <span className="mono" style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>{runId}</span>}
              </div>
            </div>

            <div className="pipeline-terminal">
              {logs.length === 0 && !isRunning && (
                <div className="pipeline-terminal-empty">
                  <Terminal size={32} color="var(--text-muted)" />
                  <p>Click <strong>Run Pipeline</strong> to start training.</p>
                  <p style={{ fontSize: '0.75rem', marginTop: 4 }}>
                    Expected time: ~5–10 min for CICIDS2018 (1M rows)
                  </p>
                </div>
              )}
              {logs.map((line, i) => (
                <div key={i} className="pipeline-log-line">
                  <span className="pipeline-log-prefix">{'>'}</span>
                  <span>{line}</span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>

          {/* Metrics (shown after training) */}
          {metrics && (
            <div style={{ marginTop: 16 }}>
              <div className="section-title" style={{ marginBottom: 12 }}>
                <Target size={15} color="var(--accent-cyan)" /> Evaluation Results
              </div>
              <div className="metric-grid">
                <MetricCard value={`${(metrics.accuracy * 100).toFixed(2)}%`}   label="Accuracy"  icon={CheckCircle} color="emerald" />
                <MetricCard value={`${(metrics.precision * 100).toFixed(2)}%`}  label="Precision" icon={Target}      color="cyan"    />
                <MetricCard value={`${(metrics.recall * 100).toFixed(2)}%`}     label="Recall"    icon={Zap}         color="cyan"    />
                <MetricCard value={`${(metrics.f1_score * 100).toFixed(2)}%`}   label="F1-Score"  icon={Activity}    color="purple"  />
              </div>
              {metrics.model_comparison && (
                <div className="card" style={{ marginTop: 16 }}>
                  <div className="section-title" style={{ marginBottom: 12 }}>Model Comparison</div>
                  <table className="data-table">
                    <thead>
                      <tr><th>Model</th><th>Accuracy</th><th>F1</th><th>AUC</th></tr>
                    </thead>
                    <tbody>
                      {Object.entries(metrics.model_comparison).map(([name, m]) => (
                        <tr key={name}>
                          <td style={{ textTransform: 'capitalize' }}>{name.replace('_', ' ')}</td>
                          <td>{(m.accuracy * 100).toFixed(2)}%</td>
                          <td>{(m.f1_score * 100).toFixed(2)}%</td>
                          <td>{m.roc_auc?.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <div className="metric-grid" style={{ marginTop: 12 }}>
                <MetricCard value={metrics.roc_auc?.toFixed(4)} label="ROC-AUC" icon={BarChart2} color="cyan" />
                <MetricCard value={`${metrics.test_samples?.toLocaleString()}`} label="Test Samples" icon={Database} color="orange" />
              </div>

              {/* Confusion Matrix */}
              <div className="card" style={{ marginTop: 16 }}>
                <div className="section-title" style={{ marginBottom: 14 }}>Confusion Matrix</div>
                <div className="cm-grid">
                  <div className="cm-cell cm-tn">
                    <div className="cm-value">{metrics.true_negatives?.toLocaleString()}</div>
                    <div className="cm-label">True Negatives</div>
                    <div className="cm-sub">Correctly identified normal</div>
                  </div>
                  <div className="cm-cell cm-fp">
                    <div className="cm-value">{metrics.false_positives?.toLocaleString()}</div>
                    <div className="cm-label">False Positives</div>
                    <div className="cm-sub">Normal → wrongly flagged</div>
                  </div>
                  <div className="cm-cell cm-fn">
                    <div className="cm-value">{metrics.false_negatives?.toLocaleString()}</div>
                    <div className="cm-label">False Negatives</div>
                    <div className="cm-sub">Attacks → missed</div>
                  </div>
                  <div className="cm-cell cm-tp">
                    <div className="cm-value">{metrics.true_positives?.toLocaleString()}</div>
                    <div className="cm-label">True Positives</div>
                    <div className="cm-sub">Correctly detected attacks</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Feature importance */}
          {features && (
            <div className="card" style={{ marginTop: 16 }}>
              <div className="section-title" style={{ marginBottom: 14 }}>
                <BarChart2 size={15} color="var(--accent-cyan)" /> Feature Importance (RandomForest Gini)
              </div>
              <div className="feature-bars">
                {features.importance.map(({ rank, feature, importance }) => {
                  const pct = (importance / features.importance[0].importance) * 100
                  return (
                    <div key={feature} className="feature-bar-row">
                      <span className="feature-rank">#{rank}</span>
                      <span className="feature-name">{feature}</span>
                      <div className="feature-bar-bg">
                        <div
                          className="feature-bar-fill"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="feature-score">{(importance * 100).toFixed(2)}%</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
