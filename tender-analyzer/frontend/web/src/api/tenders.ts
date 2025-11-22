import { apiRequest } from "./client"

export interface QuestionAnswer {
  question: string
  answer: string
}

export interface TenderStatusResponse {
  id: string
  state: string
  progress: number
  summary_ready: boolean
  full_ready: boolean
  eval_ready: boolean
  documents: string[]
  created_at: string
}

export interface SummaryResponse {
  id: string
  ready: boolean
  questions: QuestionAnswer[]
}

export interface DetailsResponse extends SummaryResponse {}

export interface EvaluationRecord {
  summary: string
  risk_level: string
  recommendation: string
}

export interface EvaluationResponse {
  id: string
  ready: boolean
  evaluation: EvaluationRecord | null
}

export async function uploadTender(name: string, files: File[]) {
  const form = new FormData();
  form.append("name", name);
  files.forEach((file) => form.append("files", file));

  return apiRequest<{ id: string }>("/tenders", {
    method: "POST",
    body: form,
  });
  }

export function startAnalysis(tenderId: string) {
  return apiRequest<TenderStatusResponse>(`/tenders/${tenderId}/start-analysis`, {
    method: "POST",
  })
}

export function getStatus(tenderId: string) {
  return apiRequest<TenderStatusResponse>(`/tenders/${tenderId}/status`)
}

export function getSummary(tenderId: string) {
  return apiRequest<SummaryResponse>(`/tenders/${tenderId}/summary`)
}

export function getDetails(tenderId: string) {
  return apiRequest<DetailsResponse>(`/tenders/${tenderId}/details`)
}

export function getEvaluation(tenderId: string) {
  return apiRequest<EvaluationResponse>(`/tenders/${tenderId}/evaluation`)
}
