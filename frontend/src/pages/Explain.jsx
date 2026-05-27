import { useState, useRef } from 'react'
import { api } from '../api'
import { Upload, Shield, BarChart2, Info, FileText } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import LoadingSpinner from '../components/LoadingSpinner'

export default function Explain() {
  const [file, setFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isExplaining, setIsExplaining] = useState(false)
  const [globalResult, setGlobalResult] = useState(null)
  const [localResult, setLocalResult] = useState(null)
  const [error, setError] = useState(null)
  const [mode, setMode] = useState('global') // 'global' | 'local'
  const [rowIndex, setRowIndex] = useState(0)
  
  const fileInputRef = useRef(null)

  const handleFileDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0])
      setError(null)
      setGlobalResult(null)
      setLocalResult(null)
    }
  }

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
      setError(null)
      setGlobalResult(null)
      setLocalResult(null)
    }
  }

  const handleExplain = async () => {
    if (!file) return

    setIsExplaining(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      if (mode === 'global') {
        const res = await api.explainGlobal(formData)
        setGlobalResult(res.data)
      } else {
        formData.append('row_index', rowIndex.toString())
        const res = await api.explainLocal(formData)
        setLocalResult(res.data)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setIsExplaining(false)
    }
  }

  return (
    <div>
      <h1 className="page-title">Explainability</h1>
      <p className="page-subtitle">Understand why the model makes decisions using SHAP</p>

      {/* Mode Selector */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
        <button 
          className={`btn ${mode === 'global' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => { setMode('global'); setLocalResult(null) }}
        >
          Global Explanations
        </button>
        <button 
          className={`btn ${mode === 'local' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => { setMode('local'); setGlobalResult(null) }}
        >
          Local Explanations
        </button>
      </div>

      <div className="card" style={{ marginBottom: 24, padding: 24, background: 'rgba(0, 212, 255, 0.05)', borderColor: 'rgba(0, 212, 255, 0.2)' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
          <Info size={24} color="var(--accent-cyan)" style={{ flexShrink: 0 }} />
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--accent-cyan)', marginBottom: 4 }}>
              {mode === 'global' ? 'Global Feature Importance' : 'Local Prediction Explanation'}
            </h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              {mode === 'global' 
                ? 'Upload a dataset to see which features the model relies on the most overall. This shows the average impact of each feature across many flows.'
                : 'Upload a dataset and select a specific row to see exactly why the model classified that specific flow as an attack or normal.'}
            </p>
          </div>
        </div>
      </div>

      {/* Upload Area */}
      <div 
        className="card"
        style={{
          border: isDragging ? '2px dashed var(--accent-cyan)' : '2px dashed var(--bg-border)',
          backgroundColor: isDragging ? 'rgba(0, 212, 255, 0.05)' : 'var(--bg-card)',
          textAlign: 'center',
          padding: '30px 20px',
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
            <FileText size={36} color="var(--accent-cyan)" style={{ margin: '0 auto 12px' }} />
            <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>{file.name}</div>
            
            {mode === 'local' && (
              <div style={{ marginTop: 20 }} onClick={e => e.stopPropagation()}>
                <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', display: 'block', marginBottom: 8 }}>
                  Row Index to Explain:
                </label>
                <input 
                  type="number" 
                  min="0"
                  value={rowIndex}
                  onChange={(e) => setRowIndex(parseInt(e.target.value) || 0)}
                  style={{ 
                    background: 'rgba(0,0,0,0.2)', 
                    border: '1px solid var(--bg-border)', 
                    color: 'white',
                    padding: '8px 12px',
                    borderRadius: 4,
                    width: 100,
                    textAlign: 'center'
                  }}
                />
              </div>
            )}

            <button 
              className="btn btn-primary" 
              style={{ marginTop: 20 }}
              onClick={(e) => {
                e.stopPropagation()
                handleExplain()
              }}
              disabled={isExplaining}
            >
              {isExplaining ? <LoadingSpinner text="Explaining..." /> : 'Generate Explanation'}
            </button>
          </div>
        ) : (
          <div>
            <Upload size={36} color="var(--text-muted)" style={{ margin: '0 auto 12px' }} />
            <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Drag & drop CSV data here
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

      {/* Global Results */}
      {globalResult && (
        <div className="card" style={{ animation: 'fadeIn 0.5s ease-out' }}>
          <div className="section-header">
            <div className="section-title">
              <BarChart2 size={18} color="var(--accent-cyan)" />
              Top Features (Mean |SHAP|)
            </div>
          </div>
          <div style={{ height: 400, marginTop: 20 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={globalResult.feature_importance.slice(0, 15)}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" horizontal={false} />
                <XAxis type="number" stroke="var(--text-muted)" tick={{fill: 'var(--text-muted)', fontSize: 12}} />
                <YAxis 
                  dataKey="feature" 
                  type="category" 
                  stroke="var(--text-muted)" 
                  tick={{fill: 'var(--text-primary)', fontSize: 11}} 
                  width={150}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8 }}
                  itemStyle={{ color: 'var(--accent-cyan)' }}
                />
                <Bar dataKey="importance" fill="var(--accent-cyan)" radius={[0, 4, 4, 0]} barSize={16} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Local Results */}
      {localResult && (
        <div style={{ animation: 'fadeIn 0.5s ease-out' }}>
          
          <div className="card" style={{ marginBottom: 24, borderLeft: `4px solid ${localResult.predicted_class === 1 ? 'var(--accent-red)' : 'var(--accent-emerald)'}` }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              {localResult.predicted_class === 1 ? (
                <Shield size={24} color="var(--accent-red)" />
              ) : (
                <Shield size={24} color="var(--accent-emerald)" />
              )}
              <div>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 600 }}>
                  Predicted: {localResult.predicted_class === 1 ? 'Attack' : 'Normal'}
                </h3>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Confidence: {(localResult.ml_confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
            
            <div style={{ 
              padding: 16, 
              background: 'rgba(255,255,255,0.03)', 
              borderRadius: 8,
              fontSize: '1.05rem',
              lineHeight: 1.6,
              color: 'var(--text-primary)'
            }}>
              {localResult.explanation}
            </div>
          </div>

          <div className="card">
             <div className="section-header" style={{ marginBottom: 20 }}>
              <div className="section-title">
                <BarChart2 size={18} color="var(--accent-cyan)" />
                Feature Contributions
              </div>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table" style={{ width: '100%' }}>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Value in Flow</th>
                    <th>SHAP Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {localResult.contributions.map((feat, i) => (
                    <tr key={i}>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem' }}>{feat.feature}</td>
                      <td>{feat.value.toFixed(4)}</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ 
                            color: feat.contribution > 0 ? 'var(--accent-red)' : 'var(--accent-emerald)',
                            fontWeight: 600,
                            minWidth: 60
                          }}>
                            {feat.contribution > 0 ? '+' : ''}{feat.contribution.toFixed(4)}
                          </span>
                          {/* Visual bar */}
                          <div style={{ width: 100, height: 6, background: 'rgba(255,255,255,0.1)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{ 
                              width: `${Math.min(100, Math.abs(feat.contribution) * 100)}%`, 
                              height: '100%', 
                              background: feat.contribution > 0 ? 'var(--accent-red)' : 'var(--accent-emerald)'
                            }} />
                          </div>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}
    </div>
  )
}
