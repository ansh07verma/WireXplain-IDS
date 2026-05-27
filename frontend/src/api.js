/**
 * api.js — All fetch() calls to the FastAPI backend
 */
const BASE = '/api'

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`)
  return res.json()
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`)
  return res.json()
}

async function put(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`PUT ${path} failed: ${res.status}`)
  return res.json()
}

async function patch(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`PATCH ${path} failed: ${res.status}`)
  return res.json()
}

async function upload(path, formData) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) throw new Error(`UPLOAD ${path} failed: ${res.status}`)
  return res.json()
}

export const api = {
  health: () => get('/health'),
  info: () => get('/info'),

  pipelineStatus: () => get('/pipeline/status'),
  pipelineDatasets: () => get('/pipeline/datasets'),
  runPipeline: (config) => post('/pipeline/run', config),

  detectCsv: (formData) => upload('/detect/csv', formData),

  explainGlobal: (formData) => upload('/explain/global', formData),
  explainLocal: (formData) => upload('/explain/local', formData),

  getInterfaces: () => get('/capture/interfaces'),
  captureStatus: () => get('/capture/status'),
  startCapture: (iface) => post('/capture/start', { interface: iface }),
  stopCapture: () => post('/capture/stop', {}),
  replayPcap: (formData) => upload('/capture/replay', formData),

  getAlerts: (params = {}) => get('/alerts?' + new URLSearchParams(params)),
  alertStats: () => get('/alerts/stats'),
  updateAlert: (id, lifecycle_state) => patch(`/alerts/${id}`, { lifecycle_state }),

  getSettings: () => get('/settings'),
  updateSettings: (body) => put('/settings', body),

  getRules: () => get('/rules/'),
}
