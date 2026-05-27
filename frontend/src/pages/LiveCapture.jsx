import { useState, useEffect, useRef } from 'react'
import { api } from '../api'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import { Wifi, Play, Square, Upload, Activity, AlertTriangle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function LiveCapture() {
  const [interfaces, setInterfaces] = useState([])
  const [selectedIface, setSelectedIface] = useState('')
  const [status, setStatus] = useState(null)
  const [events, setEvents] = useState([])
  const [packetHistory, setPacketHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const esRef = useRef(null)
  const fileRef = useRef(null)

  useEffect(() => {
    Promise.all([api.getInterfaces(), api.captureStatus()])
      .then(([ifaces, st]) => {
        setInterfaces(ifaces.interfaces || [])
        if (ifaces.interfaces?.length) setSelectedIface(ifaces.interfaces[0].name)
        setStatus(st)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))

    return () => esRef.current?.close()
  }, [])

  useEffect(() => {
    if (!status?.running) return
    connectStream()
    const poll = setInterval(() => {
      api.captureStatus().then(setStatus).catch(() => {})
    }, 3000)
    return () => clearInterval(poll)
  }, [status?.running])

  function connectStream() {
    esRef.current?.close()
    const es = new EventSource('/api/capture/stream')
    esRef.current = es
    es.onmessage = (e) => {
      const item = JSON.parse(e.data)
      if (item.type === 'flow' || item.type === 'alert') {
        setEvents((prev) => [item.data, ...prev].slice(0, 50))
      }
      if (item.type === 'stats') {
        setStatus((s) => ({ ...s, ...item.data }))
        setPacketHistory((prev) => [
          ...prev,
          { time: new Date().toLocaleTimeString(), packets: item.data.packet_count },
        ].slice(-30))
      }
      if (item.type === 'stopped' || item.type === 'replay_done') {
        setStatus(item.data)
        api.captureStatus().then(setStatus)
      }
      if (item.type === 'error') setError(item.data?.message || 'Capture error')
    }
    es.onerror = () => es.close()
  }

  const handleStart = async () => {
    setError(null)
    try {
      const res = await api.startCapture(selectedIface || null)
      setStatus(res)
      connectStream()
    } catch (e) {
      setError(e.message)
    }
  }

  const handleStop = async () => {
    try {
      const res = await api.stopCapture()
      setStatus(res)
    } catch (e) {
      setError(e.message)
    }
  }

  const handleReplay = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError(null)
    const fd = new FormData()
    fd.append('file', file)
    try {
      await api.replayPcap(fd)
      connectStream()
      const st = await api.captureStatus()
      setStatus(st)
    } catch (err) {
      setError(err.message)
    }
    e.target.value = ''
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: 100 }}>
        <LoadingSpinner text="Loading capture module..." />
      </div>
    )
  }

  return (
    <div>
      <h1 className="page-title">Live Capture</h1>
      <p className="page-subtitle">Real-time packet capture, PCAP replay, and hybrid detection</p>

      {error && (
        <div className="card" style={{ borderColor: 'var(--accent-red)', marginBottom: 16 }}>
          <div style={{ color: 'var(--accent-red)' }}>{error}</div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <div className="card">
          <div className="section-title" style={{ marginBottom: 16 }}>
            <Wifi size={16} /> Capture Controls
          </div>
          <div style={{ marginBottom: 12 }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Network Interface</label>
            <select
              value={selectedIface}
              onChange={(e) => setSelectedIface(e.target.value)}
              style={{ width: '100%', marginTop: 6, padding: 8, borderRadius: 6, background: 'var(--bg-input)', border: '1px solid var(--bg-border)', color: 'var(--text-primary)' }}
              disabled={status?.running}
            >
              {interfaces.map((i) => (
                <option key={i.name} value={i.name}>{i.name} ({i.address})</option>
              ))}
            </select>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            {!status?.running ? (
              <button className="btn btn-primary" onClick={handleStart} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Play size={16} /> Start Capture
              </button>
            ) : (
              <button className="btn btn-secondary" onClick={handleStop} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Square size={16} /> Stop
              </button>
            )}
            <button className="btn btn-secondary" onClick={() => fileRef.current?.click()} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Upload size={16} /> Replay PCAP
            </button>
            <input ref={fileRef} type="file" accept=".pcap,.pcapng,.cap" hidden onChange={handleReplay} />
          </div>
        </div>

        <div className="card">
          <div className="section-title" style={{ marginBottom: 16 }}>
            <Activity size={16} /> Live Stats
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Status</div>
              <StatusBadge type={status?.running ? 'normal' : 'warning'}>
                {status?.running ? `${status.mode} (active)` : 'Idle'}
              </StatusBadge>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Packets</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{status?.packet_count ?? 0}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Flows Processed</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{status?.flows_detected ?? 0}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Threats</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-red)' }}>{status?.threats_detected ?? 0}</div>
            </div>
          </div>
        </div>
      </div>

      {packetHistory.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="section-title" style={{ marginBottom: 12 }}>Packet Rate</div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={packetHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" />
              <XAxis dataKey="time" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
              <YAxis stroke="var(--text-muted)" />
              <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }} />
              <Line type="monotone" dataKey="packets" stroke="var(--accent-cyan)" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="card">
        <div className="section-title" style={{ marginBottom: 16 }}>
          <AlertTriangle size={16} color="var(--accent-orange)" /> Live Events
        </div>
        {events.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
            Start capture or replay a PCAP to see live flow events.
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr><th>Source</th><th>Destination</th><th>Status</th><th>Source</th><th>Severity</th></tr>
            </thead>
            <tbody>
              {events.map((ev, i) => (
                <tr key={i}>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{ev.metadata?.['Src IP']}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{ev.metadata?.['Dst IP']}:{ev.metadata?.['Dst Port']}</td>
                  <td><StatusBadge type={ev.status === 'attack' ? 'danger' : ev.status === 'anomaly' ? 'warning' : 'normal'}>{ev.status}</StatusBadge></td>
                  <td>{ev.detection_source}</td>
                  <td>{ev.severity}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
