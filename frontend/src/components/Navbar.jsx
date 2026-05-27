import { useHealth } from '../hooks/useApi'
import {
  Shield, Activity, Upload, Terminal,
  Wifi, Bell, Settings as SettingsIcon, ChevronRight
} from 'lucide-react'
import { NavLink } from 'react-router-dom'
import './Navbar.css'

const NAV_ITEMS = [
  { to: '/',          icon: Activity,     label: 'Dashboard' },
  { to: '/pipeline',  icon: Terminal,     label: 'Pipeline' },
  { to: '/predict',   icon: Upload,       label: 'Predict' },
  { to: '/explain',   icon: Shield,       label: 'Explain' },
  { to: '/capture',   icon: Wifi,         label: 'Live Capture' },
  { to: '/alerts',    icon: Bell,         label: 'Alerts' },
  { to: '/settings',  icon: SettingsIcon, label: 'Settings' },
]

export default function Navbar() {
  const { health } = useHealth()
  const isOnline = health.status === 'ok'

  return (
    <nav className="navbar">
      {/* Logo */}
      <div className="navbar-logo">
        <div className="logo-icon">
          <Shield size={22} color="#00d4ff" />
        </div>
        <div className="logo-text">
          <span className="logo-name">WireXplain</span>
          <span className="logo-sub">IDS v2.0</span>
        </div>
      </div>

      {/* Backend status */}
      <div className="navbar-status">
        <span className={`status-dot ${isOnline ? 'online' : 'offline'} pulse`} />
        <span className="status-text">
          Backend {isOnline ? 'Online' : 'Offline'}
        </span>
        {isOnline && health.uptime_seconds && (
          <span className="status-uptime">
            {Math.floor(health.uptime_seconds / 60)}m up
          </span>
        )}
      </div>

      <div className="divider" style={{ margin: '12px 16px' }} />

      {/* Nav links */}
      <ul className="navbar-links">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `nav-link ${isActive ? 'nav-link--active' : ''}`
              }
            >
              <Icon size={16} />
              <span>{label}</span>
              <ChevronRight size={12} className="nav-arrow" />
            </NavLink>
          </li>
        ))}
      </ul>

      {/* Footer */}
      <div className="navbar-footer">
        <p className="navbar-footer-text">Final Year Project</p>
        <p className="navbar-footer-text" style={{ opacity: 0.4 }}>ECE Department</p>
      </div>
    </nav>
  )
}
