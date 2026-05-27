/**
 * MetricCard — shows a large number with label and optional icon
 * Props: value, label, icon, color ("cyan"|"emerald"|"red"|"orange")
 */
export default function MetricCard({ value, label, icon: Icon, color = 'cyan', sublabel }) {
  const colorMap = {
    cyan:    'var(--accent-cyan)',
    emerald: 'var(--accent-emerald)',
    red:     'var(--accent-red)',
    orange:  'var(--accent-orange)',
    purple:  'var(--accent-purple)',
  }
  const c = colorMap[color] || colorMap.cyan

  return (
    <div className="metric-card">
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <div className="metric-value" style={{ color: c }}>{value ?? '—'}</div>
          <div className="metric-label">{label}</div>
          {sublabel && (
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
              {sublabel}
            </div>
          )}
        </div>
        {Icon && (
          <div style={{
            width: 36, height: 36,
            background: `${c}18`,
            border: `1px solid ${c}30`,
            borderRadius: 'var(--radius-sm)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Icon size={18} color={c} />
          </div>
        )}
      </div>
    </div>
  )
}
