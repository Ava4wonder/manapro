import { FormEvent, useEffect, useState } from "react"

import { useAuth } from "../context/AuthContext"

const LoginPanel = () => {
  const {
    loginStage,
    pendingEmail,
    requestCode,
    verifyCode,
    error,
    infoMessage,
    resetLogin,
  } = useAuth()
  const [email, setEmail] = useState("")
  const [code, setCode] = useState("")
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (pendingEmail) {
      setEmail(pendingEmail)
    }
  }, [pendingEmail])

  const handleRequest = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!email) {
      return
    }
    setLoading(true)
    try {
      await requestCode(email)
    } finally {
      setLoading(false)
    }
  }

  const handleVerify = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!code) {
      return
    }
    setLoading(true)
    try {
      await verifyCode(code)
    } finally {
      setLoading(false)
    }
  }

  if (loginStage === "code-sent") {
    return (
      <div className="login-shell">
        <div className="login-panel">
          <h1>Enter the verification code</h1>
          <p className="login-subtitle">
            We just sent a code to <strong>{pendingEmail}</strong>. Paste it below to finish logging in.
          </p>
          <form onSubmit={handleVerify}>
            <label>
              Code
              <input
                value={code}
                onChange={(event) => setCode(event.target.value)}
                placeholder="000000"
                autoFocus
              />
            </label>
            <div className="login-actions">
              <button type="submit" disabled={loading}>
                {loading ? "Verifying…" : "Verify code"}
              </button>
              <button type="button" className="ghost" onClick={resetLogin} disabled={loading}>
                Use another email
              </button>
            </div>
          </form>
          {infoMessage && <p className="login-info">{infoMessage}</p>}
          {error && <p className="login-error">{error}</p>}
        </div>
      </div>
    )
  }

  return (
    <div className="login-shell">
      <div className="login-panel">
        <h1>Sign in</h1>
        <p className="login-subtitle">
          Request a one-time login code for your <strong>@gruner.ch</strong> email to continue.
        </p>
        <form onSubmit={handleRequest}>
          <label>
            Email
            <input
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@gruner.ch"
              type="email"
              autoFocus
            />
          </label>
          <button type="submit" disabled={loading}>
            {loading ? "Sending…" : "Send login code"}
          </button>
        </form>
        {infoMessage && <p className="login-info">{infoMessage}</p>}
        {error && <p className="login-error">{error}</p>}
      </div>
    </div>
  )
}

export default LoginPanel
