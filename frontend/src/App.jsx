import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Pipeline from './pages/Pipeline'
import Predict from './pages/Predict'
import Explain from './pages/Explain'
import LiveCapture from './pages/LiveCapture'
import Alerts from './pages/Alerts'
import Settings from './pages/Settings'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/"         element={<Dashboard />} />
            <Route path="/pipeline" element={<Pipeline />} />
            <Route path="/predict"  element={<Predict />} />
            <Route path="/explain"  element={<Explain />} />
            <Route path="/capture"  element={<LiveCapture />} />
            <Route path="/alerts"   element={<Alerts />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
