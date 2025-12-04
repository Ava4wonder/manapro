export const PROJECT_CARD_FIELD_KEYS = [
  "projectprocurementType",
  "ProjectRole",
  "ProjectType",
  "location",
  "deadline",
  "submission_format",
  "budgetRange",
  "evaluationMethod",
  "weighting",
] as const

export type ProjectCardFieldKey = typeof PROJECT_CARD_FIELD_KEYS[number]

export type ProjectCardFields = Record<ProjectCardFieldKey, string>

export type ProjectCardAnalysisStatus = {
  state: "ingesting" | "ingested" | "summarizing" | "completed"
  label: string
  color: "orange" | "blue" | "purple" | "green"
}

export type ProjectCardInfo = {
  id: string
  name: string
  createdAt: string
  documents: number
  summaryPreview?: string
  cardFields: ProjectCardFields
  analysisStatus: ProjectCardAnalysisStatus
}

// Optional: per-project workflow status from backend state machine
export type ProjectWorkflowStatus = {
  state: string
  progress: number
}
