import { useEffect, useMemo, useState } from "react"

import { AuthProvider, useAuth } from "./context/AuthContext"
import UploadPage from "./pages/UploadPage"
import SummaryPage from "./pages/SummaryPage"
import DetailsPage from "./pages/DetailsPage"
import EvaluationPage from "./pages/EvaluationPage"
import ProjectCardsPage from "./pages/ProjectCardsPage"
import TenderStatusIndicator from "./components/TenderStatusIndicator"
import LoadingBar from "./components/LoadingBar"
import LoginPanel from "./components/LoginPanel"
import Sidebar from "./components/Sidebar"
import {
  SummaryResponse,
  DetailsResponse,
  EvaluationResponse,
  uploadTender,
  startAnalysis,
  getSummary,
  getDetails,
  getEvaluation,
} from "./api/tenders"
import { useTenderStatus } from "./hooks/useTenderStatus"
import { ProjectCardInfo } from "./types/project"
import { buildAnalysisStatus, buildProjectCardFields, buildSummaryPreview } from "./utils/projects"

type NavId = "upload" | "projects" | "summary"

const navItems: { id: NavId; label: string; description?: string }[] = [
  { id: "upload", label: "Upload", description: "Phase I" },
  { id: "projects", label: "Project cards", description: "Browse summaries" },
  { id: "summary", label: "Summary hub", description: "Insights & QA" },
]

const Dashboard = () => {
  const { isAuthenticated, authState, logout } = useAuth()
  const [tenderId, setTenderId] = useState<string | null>(null)
  const [isUploading, setUploading] = useState(false)
  const [uploadComplete, setUploadComplete] = useState(false)
  const [isStarting, setStarting] = useState(false)
  const { status, loading: statusLoading } = useTenderStatus(tenderId)
  const [summary, setSummary] = useState<SummaryResponse | null>(null)
  const [details, setDetails] = useState<DetailsResponse | null>(null)
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null)
  const [projects, setProjects] = useState<ProjectCardInfo[]>([])
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null)
  const [activeNav, setActiveNav] = useState<NavId>("upload")

  // 🔴 New: Track if analysis has already started (prevents duplicates)
  const [hasStartedAnalysis, setHasStartedAnalysis] = useState(false)
  // 🔴 New: Retry counter for failed auto-start attempts
  const [retryCount, setRetryCount] = useState(0)

  // 🔴 New: Reset analysis state when tenderId changes (e.g., new project selected)
  useEffect(() => {
    setHasStartedAnalysis(false)
    setRetryCount(0)
  }, [tenderId])

  // 🔴 New: Auto-start analysis when state becomes INGESTED
  useEffect(() => {
    const MAX_RETRIES = 3 // Limit retries to avoid spamming the backend
    const RETRY_DELAY_MS = 2000 // Wait 2s between retries

    // Only trigger if:
    // 1. Tender exists
    // 2. Status is loaded and state is INGESTED
    // 3. Analysis hasn't started yet
    // 4. Haven't exceeded max retries
    if (
      tenderId &&
      status &&
      status.state === "INGESTED" && // Match backend state (uppercase)
      !hasStartedAnalysis &&
      retryCount < MAX_RETRIES
    ) {
      const autoStartAnalysis = async () => {
        setStarting(true) // Disable manual button during auto-start
        try {
          await startAnalysis(tenderId)
          console.log(`✅ [AUTO-ANALYSIS] Started for tender: ${tenderId}`)
          setHasStartedAnalysis(true) // Mark as started to prevent re-runs
          setRetryCount(0) // Reset retries on success
        } catch (error) {
          const nextRetry = retryCount + 1
          console.error(
            `❌ [AUTO-ANALYSIS] Attempt ${nextRetry}/${MAX_RETRIES} failed for tender ${tenderId}:`,
            error
          )
          setRetryCount(nextRetry)

          // Optional: Add delay before next retry
          if (nextRetry < MAX_RETRIES) {
            console.log(`⏳ [AUTO-ANALYSIS] Retrying in ${RETRY_DELAY_MS / 1000}s...`)
          }
        } finally {
          setStarting(false) // Re-enable button after attempt
        }
      }

      autoStartAnalysis()
    }
  }, [tenderId, status, hasStartedAnalysis, retryCount])

  // Existing: Fetch summary when ready
  useEffect(() => {
    if (!tenderId || !status?.summary_ready) {
      setSummary(null)
      return
    }

    getSummary(tenderId)
      .then(setSummary)
      .catch(() => setSummary(null))
  }, [tenderId, status?.summary_ready])

  // Existing: Reset upload complete flag when navigating away
  useEffect(() => {
    if (activeNav !== "upload" && uploadComplete) {
      const timer = setTimeout(() => setUploadComplete(false), 1000)
      return () => clearTimeout(timer)
    }
  }, [activeNav, uploadComplete])

  // Existing: Fetch details when ready
  useEffect(() => {
    if (!tenderId || !status?.full_ready) {
      setDetails(null)
      return
    }

    getDetails(tenderId)
      .then(setDetails)
      .catch(() => setDetails(null))
  }, [tenderId, status?.full_ready])

  // Existing: Fetch evaluation when ready
  useEffect(() => {
    if (!tenderId || !status?.eval_ready) {
      setEvaluation(null)
      return
    }

    getEvaluation(tenderId)
      .then(setEvaluation)
      .catch(() => setEvaluation(null))
  }, [tenderId, status?.eval_ready])

  // Existing: Update project cards when status changes
  useEffect(() => {
    if (!tenderId || !status) {
      return
    }

    const analysisStatus = buildAnalysisStatus(status.summary_ready)

    setProjects((prev) => {
      const existing = prev.find((project) => project.id === tenderId)
      const updated: ProjectCardInfo = {
        id: tenderId,
        name: existing?.name ?? `Project ${tenderId.slice(0, 6)}`,
        createdAt: existing?.createdAt ?? new Date().toISOString(),
        documents: status.documents?.length ?? existing?.documents ?? 0,
        summaryPreview: existing?.summaryPreview,
        cardFields: existing?.cardFields ?? buildProjectCardFields(status.project_fields ?? null),
        analysisStatus,
      }

      if (existing) {
        return prev.map((project) => (project.id === tenderId ? updated : project))
      }

      return [updated, ...prev]
    })
  }, [status, tenderId])

  useEffect(() => {
    if (!tenderId || !status?.project_fields) {
      return
    }
    if (Object.keys(status.project_fields).length === 0) {
      return
    }

    const nextCardFields = buildProjectCardFields(status.project_fields)
    setProjects((prev) =>
      prev.map((project) =>
        project.id === tenderId
          ? {
              ...project,
              cardFields: nextCardFields,
            }
          : project,
      ),
    )
  }, [status?.project_fields, tenderId])

  // Existing: Update project cards when summary is ready
  useEffect(() => {
    if (!tenderId || !summary) {
      return
    }

    setProjects((prev) =>
      prev.map((project) =>
        project.id === tenderId
          ? {
              ...project,
              summaryPreview: buildSummaryPreview(summary),
              analysisStatus: buildAnalysisStatus(summary),
            }
          : project,
      ),
    )
  }, [summary, tenderId])

  // Existing: Handle file upload
  const handleUpload = async (name: string, files: File[]) => {
    console.log(`🚀 [UPLOAD START] Uploading package: "${name}" with ${files.length} file(s)`);
    console.log(`📁 Files:`, files.map((f) => ({ name: f.name, size: `${(f.size / 1024).toFixed(2)} KB` })));
    
    setUploading(true);
    setUploadComplete(false);
    try {
      console.log(`⏳ [UPLOAD] Sending files to server...`);
      const response = await uploadTender(name, files);
      console.log(`✅ [UPLOAD SUCCESS] Tender created with ID: ${response.id}`);

      setTenderId(response.id);
      setSelectedProjectId(response.id);
      setUploadComplete(true);

      // Optional: Navigate to projects page after upload
      setActiveNav("projects");
    } catch (error) {
      console.error(`❌ [UPLOAD ERROR] Upload failed:`, error);
      throw error;
    } finally {
      setUploading(false);
    }
  };

  // Existing: Manual start analysis (kept for re-runs)
  const handleStartAnalysis = async () => {
    if (!tenderId) {
      return
    }

    setStarting(true)
    try {
      await startAnalysis(tenderId)
    } finally {
      setStarting(false)
    }
  }

  // Existing: Button label logic (still useful for re-runs)
  const isReadyForAnalysis = useMemo(() => Boolean(tenderId && status), [tenderId, status])
  const buttonLabel = status?.state ? "Re-run analysis" : "Start analysis"
  const documents = status?.documents ?? []

  // Existing: Project selection logic
  const handleProjectSelect = (projectId: string) => {
    setSelectedProjectId(projectId)
    setTenderId(projectId)
    setActiveNav("projects")
  }

  // Existing: Clear project selection
  const handleProjectClear = () => {
    setSelectedProjectId(null)
  }

  if (!isAuthenticated) {
    return <LoginPanel />
  }

  return (
    <div className="app-shell">
      <Sidebar items={navItems} activeItem={activeNav} onChange={(id: string) => setActiveNav(id as NavId)} />
      <div className="app-content">
        <header className="content-header">
          <div>
            <h1>Tender Analyzer</h1>
            <p>Phase I: upload documents · Phase II/III: question answering & evaluation</p>
          </div>
          <div className="auth-meta">
            <div className="auth-meta__details">
              <strong>{authState.email ?? "gruner.ch user"}</strong>
              <span>{authState.tenantId ?? "tenant"}</span>
            </div>
            <button className="logout-button" type="button" onClick={logout}>
              Sign out
            </button>
          </div>
        </header>

        <div className="status-area">
          <TenderStatusIndicator status={status} loading={Boolean(tenderId) && statusLoading} />
          <LoadingBar progress={status?.progress ?? 0} />
          <div className="page-actions">
            {/* Keep button for manual re-runs (optional but useful) */}
            {activeNav === "upload" && (
              <button onClick={handleStartAnalysis} disabled={!isReadyForAnalysis || isStarting}>
                {isStarting ? "Starting analysis..." : buttonLabel}
              </button>
            )}
          </div>
        </div>

        <div className="page-body">
          {activeNav === "upload" && (
            <UploadPage isUploading={isUploading} onSubmit={handleUpload} tenderId={tenderId} uploadComplete={uploadComplete} />
          )}

          {activeNav === "projects" && (
            <ProjectCardsPage
              projects={projects}
              selectedProjectId={selectedProjectId}
              onSelectProject={handleProjectSelect}
              onClearSelection={handleProjectClear}
              summary={summary}
              details={details}
              documents={documents}
              tenderId={tenderId}
            />
          )}

          {activeNav === "summary" && (
            <div className="summary-grid">
              {/* <SummaryPage summary={summary} documents={documents} tenderId={tenderId} /> */}
              {/* <DetailsPage details={details} /> */}
              {/* <EvaluationPage evaluation={evaluation} /> */}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const App = () => (
  <AuthProvider>
    <Dashboard />
  </AuthProvider>
)

export default App
