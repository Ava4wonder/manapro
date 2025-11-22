// frontend/src/api/auth.ts

import { apiRequest } from "./client"

/**
 * Shape we want to expose to the rest of the app.
 * You can adapt this to match real backend fields if you later change them.
 */
export interface RequestLoginCodeResponse {
  email: string
  message: string
  expires_at: string
}

/**
 * Normalized shape used in AuthContext and elsewhere.
 * Note: backend currently returns `access_token`, we normalize it to `token`.
 */
export interface VerifyLoginCodeResponse {
  token: string
  user_id: string
  tenant_id: string
  email: string
}

/**
 * Raw shape returned by the backend from /api/auth/verify-code.
 * This matches the FastAPI TokenResponse we sketched:
 *
 * {
 *   "access_token": "...",
 *   "token_type": "bearer",
 *   "user_id": "...",
 *   "tenant_id": "...",
 *   "email": "..."
 * }
 */
interface RawVerifyLoginCodeResponse {
  access_token: string
  token_type: string
  user_id: string
  tenant_id: string
  email: string
}

/**
 * Request a one-time login code.
 *
 * We normalize the response into RequestLoginCodeResponse, even if the backend
 * only returns { detail: "..." } for now. This keeps the rest of the app stable.
 */
export async function requestLoginCode(email: string): Promise<RequestLoginCodeResponse> {
  const trimmed = email.trim().toLowerCase()

  // We don't strictly depend on backend shape; we normalize to our interface.
  const raw = await apiRequest<{
    email?: string
    message?: string
    expires_at?: string
    detail?: string
  }>("/auth/request-code", {
    method: "POST",
    body: JSON.stringify({ email: trimmed }),
  })

  const normalizedEmail = raw.email ?? trimmed
  const message =
    raw.message ?? raw.detail ?? "If this email is registered, a login code has been sent."
  const expiresAt = raw.expires_at ?? ""

  return {
    email: normalizedEmail,
    message,
    expires_at: expiresAt,
  }
}

/**
 * Verify a login code and return a normalized token payload.
 *
 * This function:
 *  - calls /api/auth/verify-code
 *  - receives { access_token, user_id, tenant_id, email, ... }
 *  - returns { token, user_id, tenant_id, email } for the rest of the app
 */
export async function verifyLoginCode(
  email: string,
  code: string
): Promise<VerifyLoginCodeResponse> {
  const trimmedEmail = email.trim().toLowerCase()
  const trimmedCode = code.trim()

  const raw = await apiRequest<RawVerifyLoginCodeResponse>("/auth/verify-code", {
    method: "POST",
    body: JSON.stringify({ email: trimmedEmail, code: trimmedCode }),
  })

  return {
    token: raw.access_token, // normalized name for the rest of the app
    user_id: raw.user_id,
    tenant_id: raw.tenant_id,
    email: raw.email,
  }
}
