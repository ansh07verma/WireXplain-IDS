import { useState, useRef } from 'react'
import { api } from '../api'
import { Upload, FileText, CheckCircle, AlertTriangle, ShieldAlert, PieChart as PieChartIcon } from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'

export default function Predict() {
  const [file, setFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0])
      setError(null)
      setResult(null)
    }
  }

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
      setError(null)
      setResult(null)
    }
  }

  const handleAnalyze = async () => {
    if (!file) return

    setIsAnalyzing(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const data = await api.detectCsv(formData)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getChartData = () => {
    if (!result) return []
    const normal = result.total_flows - result.threats_detected - result.anomalies_detected
    return [
      { name: 'Normal', value: normal, color: 'var(--accent-emerald)' },
      { name: 'Attack', value: result.threats_detected, color: 'var(--accent-red)' },
      { name: 'Anomaly', value: result.anomalies_detected, color: 'var(--accent-orange)' }
    ].filter(d => d.value > 0)
  }

  return (
    <div>
      <h1 className="page-title">Predict</h1>
      <p className="page-subtitle">Upload PCAP/CSV data for hybrid detection</p>

      {/* Upload Area */}
      <div 
        className="card"
        style={{
          border: isDragging ? '2px dashed var(--accent-cyan)' : '2px dashed var(--bg-border)',
          backgroundColor: isDragging ? 'rgba(0, 212, 255, 0.05)' : 'var(--bg-card)',
          textAlign: 'center',
          padding: '40px 20px',
          cursor: 'pointer',
          transition: 'all 0.2s',
          marginBottom: 24
        }}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleFileDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input 
          type="file" 
          ref={fileInputRef} 
          onChange={handleFileSelect} 
          style={{ display: 'none' }}
          accept=".csv"
        />
        
        {file ? (
          <div>
            <FileText size={48} color="var(--accent-cyan)" style={{ margin: '0 auto 16px' }} />
            <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>{file.name}</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: 4 }}>
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </div>
            <button 
              className="btn btn-primary" 
              style={{ marginTop: 20 }}
              onClick={(e) => {
                e.stopPropagation()
                handleAnalyze()
              }}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? <LoadingSpinner text="Analyzing..." /> : 'Run Detection Pipeline'}
            </button>
          </div>
        ) : (
          <div>
            <Upload size={48} color="var(--text-muted)" style={{ margin: '0 auto 16px' }} />
            <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Drag & drop your CSV file here
            </div>
            <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: 8 }}>
              or click to browse from your computer
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="card" style={{ backgroundColor: 'rgba(255,71,87,0.1)', borderColor: 'var(--accent-red)', marginBottom: 24 }}>
          <div style={{ color: 'var(--accent-red)', fontWeight: 600 }}>Analysis Failed</div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: 4 }}>{error}</div>
        </div>
      )}

      {/* Results Section */}
      {result && (
        <div style={{ animation: 'fadeIn 0.5s ease-out' }}>
          <h2 className="section-title" style={{ marginBottom: 16 }}>Analysis Results</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 20, marginBottom: 24 }}>
            
            {/* Stats Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div className="card" style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <div style={{ padding: 12, borderRadius: 8, background: 'rgba(255,255,255,0.05)' }}>
                  <FileText size={24} color="var(--text-primary)" />
                </div>
                <div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Total Flows Processed</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{result.total_flows}</div>
                </div>
              </div>
              
              <div className="card" style={{ display: 'flex', alignItems: 'center', gap: 16, borderColor: result.threats_detected > 0 ? 'rgba(255,71,87,0.3)' : 'var(--bg-border)' }}>
                <div style={{ padding: 12, borderRadius: 8, background: 'rgba(255,71,87,0.1)' }}>
                  <ShieldAlert size={24} color="var(--accent-red)" />
                </div>
                <div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Threats Detected</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: result.threats_detected > 0 ? 'var(--accent-red)' : 'var(--text-primary)' }}>
                    {result.threats_detected}
                  </div>
                </div>
              </div>

              <div className="card" style={{ display: 'flex', alignItems: 'center', gap: 16, borderColor: result.anomalies_detected > 0 ? 'rgba(255,165,2,0.3)' : 'var(--bg-border)' }}>
                <div style={{ padding: 12, borderRadius: 8, background: 'rgba(255,165,2,0.1)' }}>
                  <AlertTriangle size={24} color="var(--accent-orange)" />
                </div>
                <div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Zero-Day Anomalies</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: result.anomalies_detected > 0 ? 'var(--accent-orange)' : 'var(--text-primary)' }}>
                    {result.anomalies_detected}
                  </div>
                </div>
              </div>

              <div className="card" style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <div style={{ padding: 12, borderRadius: 8, background: 'rgba(0,255,135,0.1)' }}>
                  <CheckCircle size={24} color="var(--accent-emerald)" />
                </div>
                <div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Normal Traffic</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-emerald)' }}>
                    {result.total_flows - result.threats_detected - result.anomalies_detected}
                  </div>
                </div>
              </div>
            </div>

            {/* Donut Chart */}
            <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                <PieChartIcon size={16} /> Distribution
              </div>
              <div style={{ flex: 1, minHeight: 180 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={getChartData()}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={70}
                      paddingAngle={5}
                      dataKey="value"
                      stroke="none"
                    >
                      {getChartData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8 }}
                      itemStyle={{ color: 'var(--text-primary)' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Detailed Results Table */}
          <div className="card">
            <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 16 }}>Flow Details</h3>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table" style={{ width: '100%', minWidth: 800 }}>
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Source IP</th>
                    <th>Dest IP</th>
                    <th>Port</th>
                    <th>Status</th>
                    <th>Engine</th>
                    <th>Confidence/Score</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.slice(0, 50).map((row) => (
                    <tr key={row.id}>
                      <td style={{ color: 'var(--text-muted)' }}>{row.metadata.Timestamp || row.metadata.timestamp || '—'}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{row.metadata['Src IP'] || row.metadata.src_ip || '—'}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{row.metadata['Dst IP'] || row.metadata.dst_ip || '—'}</td>
                      <td>{row.metadata['Dst Port'] || row.metadata.dst_port || '—'}</td>
                      <td>
                        {row.status === 'attack' ? (
                          <StatusBadge type="danger">Attack</StatusBadge>
                        ) : row.status === 'anomaly' ? (
                          <StatusBadge type="warning">Anomaly</StatusBadge>
                        ) : (
                          <StatusBadge type="success">Normal</StatusBadge>
                        )}
                      </td>
                      <td>
                        {row.signature_match ? (
                          <span style={{ fontSize: '0.75rem', color: 'var(--accent-purple)', background: 'rgba(168, 85, 247, 0.1)', padding: '2px 6px', borderRadius: 4 }}>Signature</span>
                        ) : row.status === 'anomaly' ? (
                          <span style={{ fontSize: '0.75rem', color: 'var(--accent-orange)', background: 'rgba(255, 165, 2, 0.1)', padding: '2px 6px', borderRadius: 4 }}>Anomaly</span>
                        ) : (
                          <span style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)', background: 'rgba(0, 212, 255, 0.1)', padding: '2px 6px', borderRadius: 4 }}>ML Model</span>
                        )}
                      </td>
                      <td style={{ fontSize: '0.85rem' }}>
                        {row.signature_match 
                          ? row.rule_name 
                          : row.status === 'anomaly'
                            ? `Score: ${row.anomaly_score.toFixed(2)}`
                            : `${(row.ml_confidence * 100).toFixed(1)}%`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {result.results.length > 50 && (
              <div style={{ textAlign: 'center', padding: '16px 0 0', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Showing top 50 rows.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
