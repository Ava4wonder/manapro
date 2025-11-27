export const PROJECT_CARD_FIELD_KEYS = [
  "projectType",
  "projectScope",
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
  state: "in-process" | "completed"
  label: string
  color: "orange" | "green"
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
