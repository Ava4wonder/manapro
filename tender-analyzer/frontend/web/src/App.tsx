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

  useEffect(() => {
    if (!tenderId || !status?.summary_ready) {
      setSummary(null)
      return
    }

    getSummary(tenderId)
      .then(setSummary)
      .catch(() => setSummary(null))
  }, [tenderId, status?.summary_ready])

  useEffect(() => {
    // Reset uploadComplete flag when navigating away from upload page or after success
    if (activeNav !== "upload" && uploadComplete) {
      const timer = setTimeout(() => setUploadComplete(false), 1000)
      return () => clearTimeout(timer)
    }
  }, [activeNav, uploadComplete])

  useEffect(() => {
    if (!tenderId || !status?.full_ready) {
      setDetails(null)
      return
    }

    getDetails(tenderId)
      .then(setDetails)
      .catch(() => setDetails(null))
  }, [tenderId, status?.full_ready])

  useEffect(() => {
    if (!tenderId || !status?.eval_ready) {
      setEvaluation(null)
      return
    }

    getEvaluation(tenderId)
      .then(setEvaluation)
      .catch(() => setEvaluation(null))
  }, [tenderId, status?.eval_ready])

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
        cardFields: existing?.cardFields ?? buildProjectCardFields(null),
        analysisStatus,
      }

      if (existing) {
        return prev.map((project) => (project.id === tenderId ? updated : project))
      }

      return [updated, ...prev]
    })
  }, [status, tenderId])

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
              cardFields: buildProjectCardFields(summary),
              analysisStatus: buildAnalysisStatus(summary),
            }
          : project,
      ),
    )
  }, [summary, tenderId])

  const handleUpload = async (name: string, files: File[]) => {
    console.log(`🚀 [UPLOAD START] Uploading package: "${name}" with ${files.length} file(s)`)
    console.log(`📁 Files:`, files.map((f) => ({ name: f.name, size: `${(f.size / 1024).toFixed(2)} KB` })))
    
    setUploading(true)
    setUploadComplete(false)
    try {
      console.log(`⏳ [UPLOAD] Sending files to server...`)
      const response = await uploadTender(name, files)
      console.log(`✅ [UPLOAD SUCCESS] Tender created with ID: ${response.id}`)

      setTenderId(response.id)
      setSelectedProjectId(response.id)
      setUploadComplete(true)

      // 🔴 Start analysis immediately:
      console.log(`🔄 [ANALYSIS] Starting analysis for tender: ${response.id}`)
      await startAnalysis(response.id)
      console.log(`✅ [ANALYSIS] Analysis started successfully`)

      // optional: only navigate to "projects" after analysis kicks off
      setActiveNav("projects")
    } catch (error) {
      console.error(`❌ [UPLOAD ERROR] Upload failed:`, error)
      throw error
    } finally {
      setUploading(false)
    }
  }

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

  const isReadyForAnalysis = useMemo(() => Boolean(tenderId && status), [tenderId, status])
  const buttonLabel = status?.state ? "Re-run analysis" : "Start analysis"
  const documents = status?.documents ?? []

  const handleProjectSelect = (projectId: string) => {
    setSelectedProjectId(projectId)
    setTenderId(projectId)
    setActiveNav("projects")
  }

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
            {activeNav === "upload" && (
              <button onClick={handleStartAnalysis} disabled={!isReadyForAnalysis || isStarting}>
                {isStarting ? "Starting analysis." : buttonLabel}
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
            />
          )}

          {activeNav === "summary" && (
            <div className="summary-grid">
              <SummaryPage summary={summary} documents={documents} />
              <DetailsPage details={details} />
              <EvaluationPage evaluation={evaluation} />
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
