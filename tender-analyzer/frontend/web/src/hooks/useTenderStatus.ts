import { useEffect, useState } from "react"

import { getStatus, TenderStatusResponse } from "../api/tenders"

export const useTenderStatus = (tenderId: string | null) => {
  const [status, setStatus] = useState<TenderStatusResponse | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!tenderId) {
      setStatus(null)
      return
    }

    let isMounted = true
    let interval: ReturnType<typeof setInterval>

    const poll = async () => {
      setLoading(true)
      try {
        const data = await getStatus(tenderId)
        if (isMounted) {
          setStatus(data)
        }
      } catch (error) {
        console.error("Failed to fetch status", error)
      } finally {
        if (isMounted) {
          setLoading(false)
        }
      }
    }

    poll()
    interval = setInterval(poll, 3000)

    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [tenderId])

  return { status, loading }
}
