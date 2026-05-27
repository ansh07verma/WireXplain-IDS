import { useState, useEffect } from 'react'
import { api } from '../api'
import { Bell, ShieldAlert, AlertTriangle, Info, Clock, RefreshCw } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'

export default function Alerts() {
  const [alerts, setAlerts] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)
  const [severityFilter, setSeverityFilter] = useState('all')

  const fetchData = async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true)
    else setLoading(true)
    
    setError(null)
    try {
      const [alertsRes, statsRes] = await Promise.all([
        api.getAlerts(severityFilter !== 'all' ? { severity: severityFilter } : {}),
        api.alertStats()
      ])
      setAlerts(alertsRes.data)
      setStats(statsRes.data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(() => fetchData(true), 15000)
    return () => clearInterval(interval)
  }, [severityFilter])

  const handleLifecycle = async (alertId, state) => {
    try {
      await api.updateAlert(alertId, state)
      fetchData(true)
    } catch (e) {
      setError(e.message)
    }
  }

  const getSeverityColor = (sev) => {
    switch (sev) {
      case 'critical': return 'var(--accent-red)'
      case 'high': return '#ff6b6b'
      case 'medium': return 'var(--accent-orange)'
      default: return 'var(--accent-cyan)'
    }
  }

  const getStatsChartData = () => {
    if (!stats || !stats.severity_counts) return []
    return [
      { name: 'Critical', value: stats.severity_counts['critical'] || 0, color: 'var(--accent-red)' },
      { name: 'High', value: stats.severity_counts['high'] || 0, color: '#ff6b6b' },
      { name: 'Medium', value: stats.severity_counts['medium'] || 0, color: 'var(--accent-orange)' },
      { name: 'Info', value: stats.severity_counts['info'] || 0, color: 'var(--accent-cyan)' }
    ].filter(d => d.value > 0)
  }

  if (loading) return (
    <div style={{ display: 'flex', justifyContent: 'center', padding: '100px 0' }}>
      <LoadingSpinner text="Loading SIEM events..." />
    </div>
  )

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 className="page-title">Alerts & Incidents</h1>
          <p className="page-subtitle">SIEM-style view of detected threats and zero-day anomalies</p>
        </div>
        <button 
          className="btn btn-secondary" 
          onClick={() => fetchData(true)}
          disabled={refreshing}
          style={{ display: 'flex', alignItems: 'center', gap: 8 }}
        >
          <RefreshCw size={16} className={refreshing ? "spin" : ""} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="card" style={{ backgroundColor: 'rgba(255,71,87,0.1)', borderColor: 'var(--accent-red)', marginBottom: 24 }}>
          <div style={{ color: 'var(--accent-red)', fontWeight: 600 }}>Failed to load alerts</div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: 4 }}>{error}</div>
        </div>
      )}

      {/* Stats Overview */}
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: 8 }}>Total Alerts</div>
            <div style={{ fontSize: '2rem', fontWeight: 700 }}>{stats.total}</div>
          </div>
          <div className="card" style={{ textAlign: 'center', borderColor: 'rgba(255,71,87,0.3)' }}>
            <div style={{ color: 'var(--accent-red)', fontSize: '0.9rem', marginBottom: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
              <ShieldAlert size={16} /> Critical
            </div>
            <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-red)' }}>{stats.severity_counts['critical'] || 0}</div>
          </div>
          <div className="card" style={{ textAlign: 'center', borderColor: 'rgba(255,107,107,0.3)' }}>
            <div style={{ color: '#ff6b6b', fontSize: '0.9rem', marginBottom: 8 }}>High</div>
            <div style={{ fontSize: '2rem', fontWeight: 700, color: '#ff6b6b' }}>{stats.severity_counts['high'] || 0}</div>
          </div>
          <div className="card" style={{ textAlign: 'center', borderColor: 'rgba(255,165,2,0.3)' }}>
            <div style={{ color: 'var(--accent-orange)', fontSize: '0.9rem', marginBottom: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
              <AlertTriangle size={16} /> Medium
            </div>
            <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-orange)' }}>{stats.severity_counts['medium'] || 0}</div>
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 24 }}>
        
        {/* Alerts Table */}
        <div className="card" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div className="section-header" style={{ marginBottom: 16 }}>
            <div className="section-title">
              <Bell size={18} color="var(--accent-cyan)" />
              Recent Events
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              {['all', 'critical', 'high', 'medium', 'info'].map(f => (
                <button
                  key={f}
                  onClick={() => setSeverityFilter(f)}
                  style={{
                    padding: '4px 10px',
                    borderRadius: 12,
                    fontSize: '0.8rem',
                    background: severityFilter === f ? 'var(--accent-cyan)' : 'transparent',
                    color: severityFilter === f ? '#000' : 'var(--text-muted)',
                    border: `1px solid ${severityFilter === f ? 'var(--accent-cyan)' : 'var(--bg-border)'}`,
                    cursor: 'pointer',
                    textTransform: 'capitalize'
                  }}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>

          <div style={{ overflowX: 'auto', flex: 1 }}>
            {alerts.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-muted)' }}>
                No alerts found.
              </div>
            ) : (
              <table className="data-table" style={{ width: '100%', minWidth: 800 }}>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Source</th>
                    <th>Dest</th>
                    <th>Status</th>
                    <th>Severity</th>
                    <th>Details</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((alert) => (
                    <tr key={alert.id}>
                      <td style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <Clock size={12} />
                          {new Date(alert.timestamp).toLocaleString()}
                        </div>
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{alert.src_ip}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>
                        {alert.dst_ip}:{alert.dst_port}
                      </td>
                      <td>
                        {alert.status === 'attack' ? (
                          <StatusBadge type="danger">Attack</StatusBadge>
                        ) : alert.status === 'anomaly' ? (
                          <StatusBadge type="warning">Anomaly</StatusBadge>
                        ) : (
                          <StatusBadge type="info">{alert.status}</StatusBadge>
                        )}
                      </td>
                      <td>
                        <span style={{ 
                          fontSize: '0.75rem', 
                          fontWeight: 600,
                          textTransform: 'uppercase',
                          color: getSeverityColor(alert.severity), 
                          background: `color-mix(in srgb, ${getSeverityColor(alert.severity)} 15%, transparent)`, 
                          padding: '3px 8px', 
                          borderRadius: 4 
                        }}>
                          {alert.severity}
                        </span>
                      </td>
                      <td style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        {alert.rule_name || `${alert.detection_source || 'ml'} • ${(alert.confidence * 100).toFixed(1)}%`}
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: 4 }}>
                          {['acknowledged', 'false_positive', 'closed'].map((s) => (
                            <button
                              key={s}
                              onClick={() => handleLifecycle(alert.id, s)}
                              style={{
                                padding: '2px 6px', fontSize: '0.7rem', borderRadius: 4,
                                border: '1px solid var(--bg-border)', background: 'transparent',
                                color: 'var(--text-muted)', cursor: 'pointer',
                              }}
                              title={s}
                            >
                              {s.split('_')[0]}
                            </button>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Severity Chart */}
        <div className="card">
          <div className="section-title" style={{ marginBottom: 20 }}>Severity Breakdown</div>
          <div style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={getStatsChartData()}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" vertical={false} />
                <XAxis dataKey="name" stroke="var(--text-muted)" tick={{fontSize: 12}} />
                <YAxis stroke="var(--text-muted)" tick={{fontSize: 12}} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8 }}
                  cursor={{fill: 'rgba(255,255,255,0.05)'}}
                />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {getStatsChartData().map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  )
}
