import { useState, useEffect } from 'react'
import { api } from '../api'
import LoadingSpinner from '../components/LoadingSpinner'
import { Settings as SettingsIcon, Save, Key } from 'lucide-react'

export default function Settings() {
  const [settings, setSettings] = useState(null)
  const [form, setForm] = useState({
    abuseipdb_api_key: '',
    virustotal_api_key: '',
    model_type: 'rf',
    default_dataset: 'cicids2018',
    syslog_host: '',
    webhook_url: '',
  })
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState(null)

  useEffect(() => {
    api.getSettings()
      .then((s) => {
        setSettings(s)
        setForm((f) => ({
          ...f,
          model_type: s.model_type || 'rf',
          default_dataset: s.default_dataset || 'cicids2018',
        }))
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    try {
      const payload = {}
      if (form.abuseipdb_api_key) payload.abuseipdb_api_key = form.abuseipdb_api_key
      if (form.virustotal_api_key) payload.virustotal_api_key = form.virustotal_api_key
      payload.model_type = form.model_type
      payload.default_dataset = form.default_dataset
      if (form.syslog_host) payload.syslog_host = form.syslog_host
      if (form.webhook_url) payload.webhook_url = form.webhook_url
      await api.updateSettings(payload)
      setMessage('Settings saved. Restart detection to pick up model changes.')
      const s = await api.getSettings()
      setSettings(s)
    } catch (e) {
      setMessage(`Error: ${e.message}`)
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: 100 }}>
        <LoadingSpinner text="Loading settings..." />
      </div>
    )
  }

  return (
    <div>
      <h1 className="page-title">Settings</h1>
      <p className="page-subtitle">API keys, model selection, and SIEM export configuration</p>

      {message && (
        <div className="card" style={{ marginBottom: 16, borderColor: 'var(--accent-cyan)' }}>
          {message}
        </div>
      )}

      <div className="card" style={{ maxWidth: 600 }}>
        <div className="section-title" style={{ marginBottom: 20 }}>
          <Key size={16} /> Threat Intelligence
        </div>
        <div style={{ marginBottom: 16 }}>
          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            AbuseIPDB API Key {settings?.abuseipdb_configured && '(configured)'}
          </label>
          <input
            type="password"
            placeholder="Enter AbuseIPDB key"
            value={form.abuseipdb_api_key}
            onChange={(e) => setForm({ ...form, abuseipdb_api_key: e.target.value })}
            style={{ width: '100%', marginTop: 6, padding: 10, borderRadius: 6, background: 'var(--bg-input)', border: '1px solid var(--bg-border)', color: 'var(--text-primary)' }}
          />
        </div>
        <div style={{ marginBottom: 16 }}>
          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            VirusTotal API Key {settings?.virustotal_configured && '(configured)'}
          </label>
          <input
            type="password"
            placeholder="Enter VirusTotal key"
            value={form.virustotal_api_key}
            onChange={(e) => setForm({ ...form, virustotal_api_key: e.target.value })}
            style={{ width: '100%', marginTop: 6, padding: 10, borderRadius: 6, background: 'var(--bg-input)', border: '1px solid var(--bg-border)', color: 'var(--text-primary)' }}
          />
        </div>

        <div className="section-title" style={{ margin: '24px 0 16px' }}>
          <SettingsIcon size={16} /> ML & Pipeline
        </div>
        <div style={{ marginBottom: 16 }}>
          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Active ML Model</label>
          <select
            value={form.model_type}
            onChange={(e) => setForm({ ...form, model_type: e.target.value })}
            style={{ width: '100%', marginTop: 6, padding: 10, borderRadius: 6, background: 'var(--bg-input)', border: '1px solid var(--bg-border)', color: 'var(--text-primary)' }}
          >
            <option value="rf">Random Forest</option>
            <option value="xgb">XGBoost</option>
            <option value="lgb">LightGBM</option>
          </select>
        </div>
        <div style={{ marginBottom: 16 }}>
          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Default Dataset</label>
          <select
            value={form.default_dataset}
            onChange={(e) => setForm({ ...form, default_dataset: e.target.value })}
            style={{ width: '100%', marginTop: 6, padding: 10, borderRadius: 6, background: 'var(--bg-input)', border: '1px solid var(--bg-border)', color: 'var(--text-primary)' }}
          >
            <option value="cicids2018">CICIDS2018</option>
            <option value="unsw_nb15">UNSW-NB15</option>
          </select>
        </div>

        <div className="section-title" style={{ margin: '24px 0 16px' }}>SIEM Export</div>
        <div style={{ marginBottom: 16 }}>
          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Syslog Host</label>
          <input
            placeholder="e.g. 127.0.0.1"
            value={form.syslog_host}
            onChange={(e) => setForm({ ...form, syslog_host: e.target.value })}
            style={{ width: '100%', marginTop: 6, padding: 10, borderRadius: 6, background: 'var(--bg-input)', border: '1px solid var(--bg-border)', color: 'var(--text-primary)' }}
          />
        </div>
        <div style={{ marginBottom: 20 }}>
          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Webhook URL</label>
          <input
            placeholder="https://..."
            value={form.webhook_url}
            onChange={(e) => setForm({ ...form, webhook_url: e.target.value })}
            style={{ width: '100%', marginTop: 6, padding: 10, borderRadius: 6, background: 'var(--bg-input)', border: '1px solid var(--bg-border)', color: 'var(--text-primary)' }}
          />
        </div>

        <button className="btn btn-primary" onClick={handleSave} disabled={saving} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Save size={16} /> {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  )
}
