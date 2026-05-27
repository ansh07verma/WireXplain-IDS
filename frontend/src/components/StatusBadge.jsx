/**
 * StatusBadge — colored pill badge
 * type: "attack"|"normal"|"anomaly"|"critical"|"warning"|"info"|"online"|"offline"
 */
export default function StatusBadge({ type, children }) {
  return (
    <span className={`badge badge-${type}`}>
      <span className="status-dot" style={{
        width: 5, height: 5,
        background: 'currentColor',
        borderRadius: '50%',
        display: 'inline-block',
      }} />
      {children}
    </span>
  )
}
