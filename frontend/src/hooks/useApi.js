import { useState, useEffect } from 'react'
import { api } from '../api'

/**
 * useHealth — polls backend health every 15 seconds
 * Returns { status, uptime, loading }
 */
export function useHealth() {
  const [health, setHealth] = useState({ status: 'checking', uptime: null })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true

    async function check() {
      try {
        const data = await api.health()
        if (mounted) setHealth({ status: 'ok', ...data })
      } catch {
        if (mounted) setHealth({ status: 'offline' })
      } finally {
        if (mounted) setLoading(false)
      }
    }

    check()
    const interval = setInterval(check, 15000)
    return () => { mounted = false; clearInterval(interval) }
  }, [])

  return { health, loading }
}

/**
 * usePipelineStatus — checks which models are trained
 */
export function usePipelineStatus() {
  const [status, setStatus] = useState(null)

  useEffect(() => {
    api.pipelineStatus().then(setStatus).catch(console.error)
  }, [])

  return status
}
