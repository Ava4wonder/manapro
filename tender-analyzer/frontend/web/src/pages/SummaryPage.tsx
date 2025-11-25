import { useCallback, useMemo } from "react"

import PdfPreviewWithHighlights from "../components/PdfPreviewWithHighlights"
import TenderList from "../components/TenderList"
import { QuestionAnswer, SummaryResponse } from "../api/tenders"

import ReactMarkdown from "react-markdown"


type Props = {
  summary: SummaryResponse | null
  documents: string[]
}

type SummarySubgroup = {
  name: string
  items: QuestionAnswer[]
}

type SummaryGroup = {
  category: string
  subgroups: SummarySubgroup[]
}

const buildSummaryGroups = (questions: QuestionAnswer[]): SummaryGroup[] => {
  const data = new Map<string, Map<string, QuestionAnswer[]>>()

  questions.forEach((question) => {
    const category = question.category?.trim() || "General"
    const subcategory = question.subcategory?.trim() || "General"

    if (!data.has(category)) {
      data.set(category, new Map())
    }

    const subMap = data.get(category)!
    if (!subMap.has(subcategory)) {
      subMap.set(subcategory, [])
    }

    subMap.get(subcategory)!.push(question)
  })

  return Array.from(data.entries()).map(([category, subgroups]) => ({
    category,
    subgroups: Array.from(subgroups.entries()).map(([name, items]) => ({
      name,
      items,
    })),
  }))
}

const deriveStatusClass = (answer: QuestionAnswer) => {
  const explicitStatus = (answer.status ?? "").trim().toLowerCase()
  if (explicitStatus === "error") return "error"
  if (explicitStatus === "mentioned") return "mentioned"
  if (explicitStatus === "no mention" || explicitStatus === "no-mention") return "no-mention"

  const text = (answer.answer ?? "").trim().toLowerCase()
  if (text.startsWith("mentioned")) return "mentioned"
  if (text.startsWith("no mention") || text.startsWith("no-mention")) return "no-mention"
  if (text.startsWith("error")) return "error"

  return "neutral"
}

const statusLabel = (answer: QuestionAnswer, status: string) => {
  if (answer.status) {
    return answer.status
  }

  switch (status) {
    case "mentioned":
      return "Mentioned"
    case "no-mention":
      return "No mention"
    case "error":
      return "Error"
    default:
      return "Neutral"
  }
}

const SummaryPage = ({ summary, documents }: Props) => {
  const groups = useMemo(() => buildSummaryGroups(summary?.questions ?? []), [summary])

  const toggleDetails = useCallback((open: boolean) => {
    if (typeof document === "undefined") {
      return
    }
    document.querySelectorAll(".summary-page details").forEach((el) => {
      (el as HTMLDetailsElement).open = open
    })
  }, [])

  return (
    <section className="phase-pane summary-page">
      <header className="summary-page__header">
        <div>
          <h2>Phase II — Summary</h2>
          <p>
            Highlights are grouped by <strong>category</strong> and <strong>subcategory</strong>.
            Expand any bucket to read the agent answers and find quick status cues.
          </p>
        </div>
        <div className="summary-page__toolbar">
          <button type="button" onClick={() => toggleDetails(true)}>
            Expand all
          </button>
          <button type="button" onClick={() => toggleDetails(false)}>
            Collapse all
          </button>
        </div>
      </header>

      {summary?.ready ? (
        groups.length > 0 ? (
          <div className="summary-page__groups">
            {groups.map((group) => (
              <details key={group.category} open className="summary-group">
                <summary>
                  {group.category} <span>({group.subgroups.reduce((sum, sub) => sum + sub.items.length, 0)} answers)</span>
                </summary>
                <div className="summary-group__body">
                  {group.subgroups.map((subgroup) => (
                    <details key={`${group.category}-${subgroup.name}`} open className="summary-subgroup">
                      <summary>
                        {subgroup.name} <span>({subgroup.items.length})</span>
                      </summary>
                      <div className="summary-cards">
                        {subgroup.items.map((answer) => {
                          const status = deriveStatusClass(answer)
                          const badge = statusLabel(answer, status)
                          return (
                            <article
                              key={`${subgroup.name}-${answer.question}`}
                              className={["summary-card", `summary-card--${status}`].join(" ")}
                            >
                              <div className="summary-card__header">
                                <h3>{answer.question}</h3>
                                <span className={["summary-badge", `summary-badge--${status}`].join(" ")}>
                                  {badge}
                                </span>
                              </div>
                              <div className="summary-card__answer">
                                <ReactMarkdown
                                  components={{
                                    a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
                                  }}
                                >
                                  {answer.answer ?? ""}
                                </ReactMarkdown>
                              </div>
                              <div className="summary-card__meta">
                                {answer.processing_time_sec ? (
                                  <span>Processed in {answer.processing_time_sec.toFixed(2)}s</span>
                                ) : (
                                  <span>Processing time unavailable</span>
                                )}
                                {answer.error_message && (
                                  <span className="summary-card__error">{answer.error_message}</span>
                                )}
                              </div>
                            </article>
                          )
                        })}
                      </div>
                    </details>
                  ))}
                </div>
              </details>
            ))}
          </div>
        ) : (
          <p className="summary-page__empty">No highlight answers are available yet.</p>
        )
      ) : (
        <p className="summary-page__pending">
          Highlight answers are still being generated. Please check back shortly or rerun the analysis.
        </p>
      )}

      <div className="summary-page__meta">
        <TenderList documents={documents} />
        <PdfPreviewWithHighlights documents={documents} questions={summary?.questions ?? []} />
      </div>
    </section>
  )
}

export default SummaryPage
