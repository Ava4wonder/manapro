export type ProjectCardFields = {
  type: string
  location: string
  logisticsVariant: "ok" | "challenging" | "unknown"
  deadline: string
  submission: string
  budget: string
  budgetTag: string
  evaluation: string
  eligibilityChips: string[]
  riskChips: string[]
}

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
