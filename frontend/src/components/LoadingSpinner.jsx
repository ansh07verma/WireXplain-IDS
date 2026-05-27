export default function LoadingSpinner({ size = 20, text }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-secondary)' }}>
      <div
        className="spinner"
        style={{ width: size, height: size, borderWidth: size > 24 ? 3 : 2 }}
      />
      {text && <span style={{ fontSize: '0.875rem' }}>{text}</span>}
    </div>
  )
}
