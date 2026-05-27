import { useState, useEffect } from 'react'
import { useHealth, usePipelineStatus } from '../hooks/useApi'
import { api } from '../api'
import MetricCard from '../components/MetricCard'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import {
  Shield, Cpu, Database, AlertTriangle,
  CheckCircle, XCircle, Clock, Activity
} from 'lucide-react'

export default function Dashboard() {
  const { health, loading: healthLoading } = useHealth()
  const pipelineStatus = usePipelineStatus()
  const [sysInfo, setSysInfo] = useState(null)
  const [alertStats, setAlertStats] = useState(null)

  useEffect(() => {
    api.info().then(setSysInfo).catch(() => {})
    const loadAlerts = () => api.alertStats().then((r) => setAlertStats(r.data)).catch(() => {})
    loadAlerts()
    const interval = setInterval(loadAlerts, 10000)
    return () => clearInterval(interval)
  }, [])

  const isOnline = health.status === 'ok'

  return (
    <div>
      <h1 className="page-title">Dashboard</h1>
      <p className="page-subtitle">System overview and status</p>

      <div className="card" style={{
        marginBottom: 24,
        background: isOnline
          ? 'linear-gradient(135deg, rgba(0,255,135,0.05), rgba(0,212,255,0.05))'
          : 'rgba(255,71,87,0.05)',
        borderColor: isOnline ? 'rgba(0,255,135,0.2)' : 'rgba(255,71,87,0.2)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {healthLoading ? (
            <LoadingSpinner text="Connecting to backend..." />
          ) : (
            <>
              {isOnline
                ? <CheckCircle size={24} color="var(--accent-emerald)" />
                : <XCircle size={24} color="var(--accent-red)" />
              }
              <div>
                <div style={{ fontWeight: 600, fontSize: '1rem', color: isOnline ? 'var(--accent-emerald)' : 'var(--accent-red)' }}>
                  Backend {isOnline ? 'Online' : 'Offline'}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                  {isOnline
                    ? `http://localhost:8000 • Python ${health.python_version} • Uptime: ${Math.floor((health.uptime_seconds || 0) / 60)}m`
                    : 'Start backend: cd backend && uvicorn main:app --reload'
                  }
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        <MetricCard
          value={isOnline ? 'Active' : 'Offline'}
          label="Backend Status"
          icon={Activity}
          color={isOnline ? 'emerald' : 'red'}
        />
        <MetricCard
          value={pipelineStatus
            ? Object.values(pipelineStatus.models).filter(Boolean).length + '/4'
            : '—'}
          label="Models Trained"
          icon={Cpu}
          color="cyan"
          sublabel="RF • XGB • LGB • ISO"
        />
        <MetricCard
          value={alertStats?.total ?? '—'}
          label="Total Alerts"
          icon={AlertTriangle}
          color="orange"
          sublabel={`Critical: ${alertStats?.severity_counts?.critical ?? 0}`}
        />
        <MetricCard
          value={pipelineStatus?.dataset_ready ? 'Ready' : 'Missing'}
          label="Dataset"
          icon={Database}
          color={pipelineStatus?.dataset_ready ? 'emerald' : 'orange'}
          sublabel={pipelineStatus?.default_dataset || 'cicids2018'}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div className="card">
          <div className="section-header">
            <div className="section-title">
              <Cpu size={16} color="var(--accent-cyan)" />
              Model Status
            </div>
          </div>
          {pipelineStatus ? (
            <table className="data-table">
              <thead>
                <tr><th>Model</th><th>Status</th></tr>
              </thead>
              <tbody>
                {[
                  { key: 'random_forest', name: 'RandomForest' },
                  { key: 'isolation_forest', name: 'IsolationForest' },
                  { key: 'xgboost', name: 'XGBoost' },
                  { key: 'lightgbm', name: 'LightGBM' },
                ].map((m) => (
                  <tr key={m.key}>
                    <td>{m.name}</td>
                    <td>
                      <StatusBadge type={pipelineStatus.models[m.key] ? 'normal' : 'warning'}>
                        {pipelineStatus.models[m.key] ? 'Trained' : 'Not trained'}
                      </StatusBadge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <LoadingSpinner text="Checking models..." />
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="card">
            <div className="section-header">
              <div className="section-title">
                <AlertTriangle size={16} color="var(--accent-red)" />
                Recent Alerts
              </div>
            </div>
            {alertStats?.recent?.length > 0 ? (
              <table className="data-table">
                <tbody>
                  {alertStats.recent.map((a) => (
                    <tr key={a.id}>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>{a.src_ip}</td>
                      <td><StatusBadge type={a.status === 'attack' ? 'danger' : 'warning'}>{a.status}</StatusBadge></td>
                      <td style={{ fontSize: '0.75rem' }}>{a.severity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No alerts yet. Run detection or live capture.</div>
            )}
          </div>

          <div className="card">
            <div className="section-title" style={{ marginBottom: 12 }}>
              <Database size={16} color="var(--accent-cyan)" />
              Packages
            </div>
            {sysInfo ? (
              <table className="data-table">
                <tbody>
                  {Object.entries(sysInfo.packages || {}).slice(0, 6).map(([pkg, ver]) => (
                    <tr key={pkg}>
                      <td>{pkg}</td>
                      <td style={{ fontSize: '0.8rem' }}>{ver}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <LoadingSpinner text="Loading..." />
            )}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <div className="section-title" style={{ marginBottom: 16 }}>
          <Clock size={16} color="var(--accent-cyan)" />
          Getting Started
        </div>
        {[
          { step: '1', done: isOnline, text: 'Backend running', detail: 'uvicorn main:app --reload' },
          { step: '2', done: pipelineStatus?.dataset_ready, text: 'Dataset in backend/data/raw/', detail: 'CICIDS2018 or UNSW-NB15' },
          { step: '3', done: pipelineStatus?.any_model_trained, text: 'Train models via Pipeline', detail: 'RF + XGB + LGB + IsolationForest' },
          { step: '4', done: true, text: 'Live capture or CSV detection', detail: 'Live Capture / Predict pages' },
        ].map(({ step, done, text, detail }) => (
          <div key={step} style={{
            display: 'flex', alignItems: 'center', gap: 14, padding: '12px 16px', marginBottom: 8,
            background: done ? 'rgba(0,255,135,0.04)' : 'rgba(0,0,0,0.2)',
            borderRadius: 'var(--radius-sm)',
            border: `1px solid ${done ? 'rgba(0,255,135,0.15)' : 'var(--bg-border)'}`,
          }}>
            <div style={{
              width: 28, height: 28, borderRadius: '50%',
              background: done ? 'rgba(0,255,135,0.15)' : 'var(--bg-border)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.75rem', fontWeight: 700,
              color: done ? 'var(--accent-emerald)' : 'var(--text-muted)',
            }}>
              {done ? '✓' : step}
            </div>
            <div>
              <div style={{ fontSize: '0.875rem', fontWeight: 500 }}>{text}</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{detail}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
