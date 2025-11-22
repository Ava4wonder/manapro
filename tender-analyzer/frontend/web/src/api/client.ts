// frontend/web/src/api/client.ts

// Safely read Vite env var if available, otherwise fall back to a default.
const API_BASE =
  (typeof import.meta !== "undefined" &&
    (import.meta as any).env &&
    (import.meta as any).env.VITE_API_BASE_URL) ||
  "http://localhost:8000/api"

let authToken: string | null = null

export const setAuthToken = (token: string | null) => {
  authToken = token
}

/**
 * Generic API request helper using fetch.
 *
 * - Prefixes all paths with API_BASE (e.g., http://localhost:8000/api)
 * - Attaches Authorization: Bearer <token> if setAuthToken() was called
 * - Sends cookies / credentials with every request (for session-based auth)
 * - Automatically sets Content-Type: application/json for non-FormData bodies
 * - Throws Error on non-2xx responses, with response text as message if present
 */
export async function apiRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(options.headers as Record<string, string> | undefined),
  }

  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`
  }

  // If body is not FormData, default to JSON content type
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = headers["Content-Type"] ?? "application/json"
  }

  const response = await fetch(`${API_BASE}${path}`, {
    // Always include credentials so session cookies are sent to the API.
    credentials: "include",
    ...options,
    headers,
  })

  if (!response.ok) {
    const text = await response.text()
    // Keep error message readable; backend often returns JSON like {"detail": "..."}
    let message = text || "API request failed"
    try {
      const parsed = JSON.parse(text)
      if (parsed && typeof parsed.detail === "string") {
        message = parsed.detail
      }
    } catch {
      // ignore JSON parse errors, fall back to raw text
    }
    throw new Error(message)
  }

  if (response.status === 204) {
    // no content
    return undefined as T
  }

  return (await response.json()) as T
}
