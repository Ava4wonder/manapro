// frontend/web/src/pages/ProjectCardsPage.tsx
import { FC, useEffect, useMemo, useState } from "react"
import { ProjectCardInfo } from "../types/project"
import { SummaryResponse, DetailsResponse, QuestionAnswerReference } from "../api/tenders"
import ProjectCard from "../components/ProjectCard"
import ReactMarkdown from "react-markdown"
import PdfPreviewWithHighlights from "../components/PdfPreviewWithHighlights"


type ProjectCardsPageProps = {
  projects: ProjectCardInfo[]
  selectedProjectId: string | null
  onSelectProject: (projectId: string) => void
  onClearSelection: () => void
  summary: SummaryResponse | null
  details: DetailsResponse | null
  documents: string[]
  tenderId: string | null
}

type DetailTab = "summary" | "details"

const ProjectCardsPage: FC<ProjectCardsPageProps> = ({
  projects,
  selectedProjectId,
  onSelectProject,
  onClearSelection,
  summary,
  details,
  documents,
  tenderId,
}) => {
  const [activeTab, setActiveTab] = useState<DetailTab>("summary")
  const [activeReference, setActiveReference] = useState<QuestionAnswerReference | null>(null)
  const [isPdfPanelOpen, setIsPdfPanelOpen] = useState(true)

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  )

  useEffect(() => {
    setActiveReference(null)
  }, [summary?.id])

  const handleCardClick = (id: string) => {
    if (id === selectedProjectId) return
    setActiveTab("summary")
    onSelectProject(id)
  }

  const handleReferenceSelect = (reference: QuestionAnswerReference) => {
    setActiveReference(reference)
    setIsPdfPanelOpen(true)
  }

  return (
    <div className="project-cards-page">
      <header>
        <h2>Project cards</h2>
        <p>Browse all uploaded tenders and jump into their Q&A views.</p>
      </header>

      {projects.length === 0 ? (
        <div className="project-cards-empty">
          <p className="project-cards-empty__title">No projects yet</p>
          <p className="project-cards-empty__subtitle">
            Upload a tender in the <strong>Upload</strong> section to see it appear here.
          </p>
        </div>
      ) : (
        <div className="project-cards-grid">
          {projects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              isActive={project.id === selectedProjectId}
              onClick={() => handleCardClick(project.id)}
            />
          ))}
        </div>
      )}

      <section className="project-detail">
        {selectedProject ? (
          <>
            <header className="project-detail__header">
              <div>
                <h2>{selectedProject.name}</h2>
                <p>
                  {selectedProject.documents} document(s) · analysis {selectedProject.analysisStatus.label}
                </p>
              </div>
              <button type="button" className="project-detail__close" onClick={onClearSelection}>
                Close
              </button>
            </header>

            <div className="project-detail__tabs">
              <button
                type="button"
                className={`project-detail__tab ${activeTab === "summary" ? "is-active" : ""}`}
                onClick={() => setActiveTab("summary")}
              >
                Summary
              </button>
              <button
                type="button"
                className={`project-detail__tab ${activeTab === "details" ? "is-active" : ""}`}
                onClick={() => setActiveTab("details")}
              >
                Details
              </button>
            </div>

            {activeTab === "summary" ? (
              summary && summary.ready ? (
                <div className="project-detail__content">
                  <div className="project-detail__summary">
                    <div className="question-grid">
                      {summary.questions.map((item, idx) => (
                        <article key={idx}>
                          <h4>{item.question}</h4>
                          <div className="markdown-answer">
                            <ReactMarkdown
                              components={{
                                a: ({ node, ...props }) => (
                                  <a {...props} target="_blank" rel="noopener noreferrer" />
                                ),
                              }}
                            >
                              {item.answer}
                            </ReactMarkdown>
                          </div>
                          {item.references && item.references.length > 0 && (
                            <div className="reference-links">
                              {item.references.map((reference, referenceIdx) => (
                                <button
                                  key={`${item.question}-${reference.chunk_id ?? referenceIdx}`}
                                  type="button"
                                  className="reference-link"
                                  onClick={() => handleReferenceSelect(reference)}
                                >
                                  Source {referenceIdx + 1}: {reference.file_name}
                                  {reference.page ? ` · p.${reference.page}` : ""}
                                </button>
                              ))}
                            </div>
                          )}
                        </article>
                      ))}
                    </div>
                  </div>
                  <aside className={`pdf-panel ${isPdfPanelOpen ? "is-open" : "is-collapsed"}`}>
                    <header className="pdf-panel__header">
                      <div>
                        <h3>PDF panel</h3>
                        <p>Inspect the source documents and highlighted references.</p>
                      </div>
                      <button
                        type="button"
                        className="pdf-panel__toggle"
                        onClick={() => setIsPdfPanelOpen((prev) => !prev)}
                        aria-expanded={isPdfPanelOpen}
                      >
                        {isPdfPanelOpen ? "Collapse" : "Expand"}
                      </button>
                    </header>
                    <div className="pdf-panel__body" aria-hidden={!isPdfPanelOpen}>
                      <PdfPreviewWithHighlights
                        documents={documents}
                        tenderId={tenderId}
                        activeReference={activeReference}
                        onClearReference={() => setActiveReference(null)}
                      />
                    </div>
                  </aside>
                </div>
              ) : (
                <p className="project-detail-placeholder">
                  Summary not ready yet. Start or re-run analysis from the Upload page.
                </p>
              )
            ) : details && details.ready ? (
              <div className="question-grid">
                {details.questions.map((item, idx) => (
                  <article key={idx}>
                    <h4>{item.question}</h4>
                    <p>{item.answer}</p>
                  </article>
                ))}
              </div>
            ) : (
              <p className="project-detail-placeholder">
                Detailed Q&A not ready yet. Start or re-run analysis from the Upload page.
              </p>
            )}
          </>
        ) : (
          <p className="project-detail-placeholder">
            Select a project card to see its summary and detailed questions here.
          </p>
        )}
      </section>
    </div>
  )
}

export default ProjectCardsPage
