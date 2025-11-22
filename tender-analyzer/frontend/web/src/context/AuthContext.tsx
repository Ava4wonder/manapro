// frontend/src/context/AuthContext.tsx

import { createContext, ReactNode, useContext, useEffect, useState } from "react"

import { verifyLoginCode, requestLoginCode } from "../api/auth"
import { setAuthToken } from "../api/client"

type AuthState = {
  token: string | null
  userId: string | null
  tenantId: string | null
  email: string | null
}

type LoginStage = "idle" | "code-sent"

type AuthContextValue = {
  authState: AuthState
  isAuthenticated: boolean
  loginStage: LoginStage
  pendingEmail: string | null
  error: string | null
  infoMessage: string | null
  requestCode: (email: string) => Promise<void>
  verifyCode: (code: string) => Promise<void>
  resetLogin: () => void
  logout: () => void
}

const LOCAL_STORAGE_KEY = "tender_analyzer_auth"

const createEmptyAuthState = (): AuthState => ({
  token: null,
  userId: null,
  tenantId: null,
  email: null,
})

const loadStoredAuthState = (): AuthState => {
  if (typeof window === "undefined") {
    return createEmptyAuthState()
  }

  const stored = window.localStorage.getItem(LOCAL_STORAGE_KEY)
  if (!stored) {
    return createEmptyAuthState()
  }

  try {
    return JSON.parse(stored) as AuthState
  } catch {
    return createEmptyAuthState()
  }
}

const noopAsync = async () => {}
const noop = () => {}

const AuthContext = createContext<AuthContextValue>({
  authState: createEmptyAuthState(),
  isAuthenticated: false,
  loginStage: "idle",
  pendingEmail: null,
  error: null,
  infoMessage: null,
  requestCode: noopAsync,
  verifyCode: noopAsync,
  resetLogin: noop,
  logout: noop,
})

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [authState, setAuthState] = useState<AuthState>(() => loadStoredAuthState())
  const [loginStage, setLoginStage] = useState<LoginStage>("idle")
  const [pendingEmail, setPendingEmail] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [infoMessage, setInfoMessage] = useState<string | null>(null)

  // Keep token in localStorage and in the API client
  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(authState))
    }
    setAuthToken(authState.token)
  }, [authState])

  const isAuthenticated = Boolean(authState.token)

  const sanitizeError = (value: unknown, fallback: string) => {
    if (value instanceof Error) {
      return value.message
    }
    return fallback
  }

  const requestCode = async (email: string) => {
    setError(null)
    setInfoMessage(null)

    try {
      const trimmed = email.trim().toLowerCase()
      // We normalize the backend response in api/auth.ts,
      // but we mainly care about which email we used and a user-friendly message.
      const resp = await requestLoginCode(trimmed)

      // Use the email we know (either from response or the argument)
      const effectiveEmail = resp.email || trimmed

      setPendingEmail(effectiveEmail)
      setLoginStage("code-sent")
      setInfoMessage(resp.message || `Code sent to ${effectiveEmail}. It expires soon.`)
    } catch (err) {
      const message = sanitizeError(err, "Unable to send verification code")
      setError(message)
      throw err
    }
  }

  const verifyCode = async (code: string) => {
    if (!pendingEmail) {
      const message = "Please request a code first"
      setError(message)
      throw new Error(message)
    }

    setError(null)
    setInfoMessage(null)

    try {
      const trimmedCode = code.trim()
      const response = await verifyLoginCode(pendingEmail, trimmedCode)

      setAuthState({
        token: response.token, // ✅ normalized in api/auth.ts
        userId: response.user_id,
        tenantId: response.tenant_id,
        email: response.email,
      })

      setLoginStage("idle")
      setPendingEmail(null)
      setInfoMessage(`Welcome, ${response.email}`)
    } catch (err) {
      const message = sanitizeError(err, "Unable to verify code")
      setError(message)
      throw err
    }
  }

  const resetLogin = () => {
    setError(null)
    setInfoMessage(null)
    setLoginStage("idle")
    setPendingEmail(null)
  }

  const logout = () => {
    setAuthState(createEmptyAuthState())
    resetLogin()
  }

  const contextValue: AuthContextValue = {
    authState,
    isAuthenticated,
    loginStage,
    pendingEmail,
    error,
    infoMessage,
    requestCode,
    verifyCode,
    resetLogin,
    logout,
  }

  return <AuthContext.Provider value={contextValue}>{children}</AuthContext.Provider>
}

export const useAuth = () => useContext(AuthContext)
