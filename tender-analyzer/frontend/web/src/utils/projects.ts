import { SummaryResponse } from "../api/tenders"
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
  summary?: SummaryResponse | null | boolean,
): ProjectCardAnalysisStatus {
  const completed = typeof summary === "boolean" ? summary : Boolean(summary?.ready)
  return {
    state: completed ? "completed" : "in-process",
    label: completed ? "Completed" : "In process",
    color: completed ? "green" : "orange",
  }
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
