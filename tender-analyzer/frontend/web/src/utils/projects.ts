import { SummaryResponse, TenderStatusResponse } from "../api/tenders"
import {
  ProjectCardAnalysisStatus,
  ProjectCardFields,
  ProjectCardFieldKey,
  PROJECT_CARD_FIELD_KEYS,
} from "../types/project"

const DEFAULT_FIELD_VALUE = "TBD"

export const PROJECT_CARD_FIELD_LABELS: Record<ProjectCardFieldKey, string> = {
  projectType: "Project Type",
  projectScope: "Project Scope",
  location: "Location",
  deadline: "Deadline",
  submission_format: "Submission Format",
  budgetRange: "Budget Range",
  evaluationMethod: "Evaluation Method",
  weighting: "Weighting",
}

export function buildProjectCardFields(source: Record<string, string> | ProjectCardFields | null): ProjectCardFields {
  return PROJECT_CARD_FIELD_KEYS.reduce((acc, key) => {
    const rawValue = source?.[key]
    const normalized = typeof rawValue === "string" ? rawValue.trim() : ""
    acc[key] = normalized || DEFAULT_FIELD_VALUE
    return acc
  }, {} as ProjectCardFields)
}

export function buildAnalysisStatus(
  source?: SummaryResponse | TenderStatusResponse | null | boolean,
): ProjectCardAnalysisStatus {
  // Boolean / summary-only fallback (e.g. existing callers)
  if (typeof source === "boolean" || (source && !("state" in source))) {
    const completed =
      typeof source === "boolean" ? source : Boolean((source as SummaryResponse | null | undefined)?.ready)

    return completed
      ? { state: "completed", label: "Completed", color: "green" }
      : { state: "summarizing", label: "Summarizing", color: "purple" }
  }

  const status = source as TenderStatusResponse | undefined
  const rawState = status?.state?.toUpperCase?.() ?? ""

  // Map backend state machine -> UI states
  if (
    rawState === "PENDING" ||
    rawState === "INGESTING" ||
    rawState === "UPLOADING" ||
    rawState === "PARSING" ||
    rawState === "QUEUED"
  ) {
    return { state: "ingesting", label: "In queue", color: "orange" }
  }

  if (rawState === "INGESTED") {
    return { state: "ingested", label: "Ingested", color: "blue" }
  }

  if (rawState === "SUMMARIZING") {
    return { state: "summarizing", label: "Summarizing", color: "purple" }
  }

  // Treat ready / done-like states as completed
  const isCompleted =
    rawState === "READY" ||
    rawState === "DONE" ||
    status?.summary_ready ||
    status?.full_ready ||
    status?.eval_ready

  if (isCompleted) {
    return { state: "completed", label: "Completed", color: "green" }
  }

  // Fallback: still in ingesting / processing
  return { state: "ingesting", label: "Ingesting", color: "orange" }
}

export function buildSummaryPreview(summary: SummaryResponse | null): string {
  if (!summary?.ready) {
    return "Summary pending."
  }

  const preview = summary.questions.map((item) => item.answer).join(" ").trim()
  if (!preview) {
    return "Summary ready - data pending"
  }

  const excerpt = preview.slice(0, 160).trim()
  return excerpt.length < preview.length ? `${excerpt}.` : excerpt
}

export function formatProjectDate(dateString: string): string {
  if (!dateString) return "Unknown"
  try {
    const date = new Date(dateString)
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    })
  } catch {
    return dateString
  }
}
