import { SummaryResponse } from "../api/tenders"
import { ProjectCardAnalysisStatus, ProjectCardFields } from "../types/project"

const CHUNK_MAX_LENGTH = 36

const highlightQuestionsByField = {
  projectType:
    "What is the type and nature of the project (e.g., new construction, rehabilitation, study, supervision)?",
  location:
    "Where is the project located (country, region, city) and are there any site-specific constraints?",
  logistics: "Is the project location logistically feasible?",
  deadline: "What is the submission deadline?",
  submission:
    "What is the required submission format (physical, electronic, or both)?",
  budgetRange:
    "Is there an indication of the budget range, and does it match our expectations?",
  budgetReality: "Is the budget realistic compared to the scope?",
  evaluationMethod:
    "How will the proposals be evaluated (e.g., lowest price, quality-based, mixed)?",
  weighting: "Is there a weighting system for price vs. quality?",
}

const eligibilityQuestions = [
  "What are the tendering eligiblity criterias in terms of financial?",
  "What are the tendering eligiblity criterias in terms of technical?",
  "What are the tendering eligiblity criterias in terms of language skills?",
]

const riskQuestions = [
  "Are the payment terms reasonable (e.g., advance payments, milestones)?",
  "Does the contract require the bidder to provide performance guarantees?",
  "Would we be required to pre-finance a significant portion of the work before payments start?",
  "Is the client's financial stability confirmed, or is there a risk of non-payment?",
  "What are the liability caps, if any?",
  "Are we required to take on unlimited liability?",
  "Are there unreasonable termination clauses?",
  "Are there indications that the project is underfunded or at risk of cancellation?",
]

const defaultCardFields: ProjectCardFields = {
  type: "Loading…",
  location: "Unknown",
  logisticsVariant: "unknown",
  deadline: "Pending",
  submission: "Pending",
  budget: "TBD",
  budgetTag: "Realistic ?",
  evaluation: "Pending",
  eligibilityChips: [],
  riskChips: [],
}

export function buildProjectCardFields(summary: SummaryResponse | null): ProjectCardFields {
  if (!summary?.questions?.length) {
    return { ...defaultCardFields }
  }

  const answers = new Map(summary.questions.map((item) => [item.question, item.answer.trim()]))

  const get = (question: string) => answers.get(question) ?? ""

  const logisticsAnswer = get(highlightQuestionsByField.logistics).toLowerCase()
  const logisticsVariant = logisticsAnswer
    ? logisticsAnswer.includes("challeng") ||
      logisticsAnswer.includes("not") ||
      logisticsAnswer.includes("tight")
      ? "challenging"
      : "ok"
    : "unknown"

  const budgetStep = get(highlightQuestionsByField.budgetReality) || get(highlightQuestionsByField.budgetRange)
  const budgetTag = buildBudgetTag(get(highlightQuestionsByField.budgetReality))

  const eligibilityChips = eligibilityQuestions
    .map((question) => get(question))
    .filter(Boolean)
    .map((text) => shortenChip(text))
    .slice(0, 3)

  const riskChips = riskQuestions
    .map((question) => get(question))
    .filter(Boolean)
    .map((text) => shortenChip(text))
    .slice(0, 3)

  return {
    type: get(highlightQuestionsByField.projectType) || "TBD",
    location: get(highlightQuestionsByField.location) || "Unknown",
    logisticsVariant,
    deadline: get(highlightQuestionsByField.deadline) || "TBD",
    submission: get(highlightQuestionsByField.submission) || "TBD",
    budget: budgetStep || "TBD",
    budgetTag,
    evaluation: get(highlightQuestionsByField.evaluationMethod) || get(highlightQuestionsByField.weighting) || "TBD",
    eligibilityChips,
    riskChips,
  }
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
    return "Summary pending…"
  }

  const preview = summary.questions.map((item) => item.answer).join(" ").trim()
  if (!preview) {
    return "Summary ready – data pending"
  }

  const excerpt = preview.slice(0, 160).trim()
  return excerpt.length < preview.length ? `${excerpt}…` : excerpt
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

const buildBudgetTag = (input: string) => {
  const text = input.trim().toLowerCase()
  if (!text) {
    return "Realistic ?"
  }
  if (text.includes("realistic") || text.includes("match") || text.includes("reasonable")) {
    return "Realistic ✓"
  }
  if (text.includes("unrealistic") || text.includes("tight") || text.includes("constrained")) {
    return "Realistic ✗"
  }
  return "Realistic ?"
}

const shortenChip = (text: string) => {
  const trimmed = text.trim()
  if (trimmed.length <= CHUNK_MAX_LENGTH) {
    return trimmed
  }
  return `${trimmed.slice(0, CHUNK_MAX_LENGTH).trim()}…`
}
