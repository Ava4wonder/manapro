// frontend/web/src/pages/ProjectCardsPage.tsx
import { FC, useEffect, useMemo, useState } from "react"
import { ProjectCardInfo } from "../types/project"
import { SummaryResponse, DetailsResponse, QuestionAnswerReference } from "../api/tenders"
import ProjectCard from "../components/ProjectCard"
import ReactMarkdown, { defaultUrlTransform } from "react-markdown"
import remarkGfm from "remark-gfm"
import { usePdfViewerStore } from "../store/pdfViewerStore"

const CHUNKREF_PREFIX = "chunkref://"


type ProjectCardsPageProps = {
  projects: ProjectCardInfo[]
  selectedProjectId: string | null
  onSelectProject: (projectId: string) => void
  onClearSelection: () => void
  summary: SummaryResponse | null
  details: DetailsResponse | null
  documents: string[]
  tenderId: string | null
  onDeleteProject?: (projectId: string) => void
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
  onDeleteProject,
}) => {
  const [activeTab, setActiveTab] = useState<DetailTab>("summary")
  const { show, clearActiveReference } = usePdfViewerStore()

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  )

  useEffect(() => {
    clearActiveReference()
  }, [summary?.id, clearActiveReference])

  const handleCardClick = (id: string) => {
    if (id === selectedProjectId) return
    setActiveTab("summary")
    onSelectProject(id)
  }

  const handleReferenceSelect = (reference: QuestionAnswerReference) => {
    show(reference, tenderId)
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
              onDelete={onDeleteProject ? () => onDeleteProject(project.id) : undefined}
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
                    <p className="project-detail__hint">
                      {documents.length > 0
                        ? `${documents.length} document(s) available.`
                        : "No documents have finished uploading yet."}{" "}
                      Use the floating PDF viewer (bottom-right) to inspect highlighted references.
                    </p>
                    <div className="question-grid">
                      {summary.questions.map((item, idx) => {
                        const filteredReferences =
                          item.references?.filter((reference) => {
                            // keep if tenderId not set on ref (old data)
                            // or matches current tenderId
                            if (!reference.tender_id || !tenderId) return true
                            return reference.tender_id === tenderId
                          }) ?? []

                        console.log("[ProjectCardsPage] question", item.question, {
                            tenderId,
                            rawRefs: item.references?.map((r) => ({
                              file_name: r.file_name,
                              page: r.page,
                              tender_id: r.tender_id,
                            })),
                            filteredRefs: filteredReferences.map((r) => ({
                              file_name: r.file_name,
                              page: r.page,
                              tender_id: r.tender_id,
                            })),
                          })
                        const referenceMap = new Map<string, QuestionAnswerReference>()
                        filteredReferences.forEach((reference) => {
                          if (reference.chunk_id) {
                            referenceMap.set(String(reference.chunk_id), reference)
                          }
                        })

                        return (
                          <article key={idx}>
                            <h4>{item.question}</h4>
                            <div className="markdown-answer">
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                urlTransform={(url) => {
                                  // 对我们自定义的协议，直接放行，不做安全裁剪
                                  if (url.startsWith(CHUNKREF_PREFIX)) {
                                    return url
                                  }
                                  // 其它的还是走官方的默认 transform，保持安全性
                                  return defaultUrlTransform(url)
                                }}
                                components={{
                                  a: ({ href, children, node, ...linkProps }: any) => {
                                    console.log("Markdown link href = ", JSON.stringify(href))
                                    if (href?.startsWith(CHUNKREF_PREFIX)) {
                                      const chunkId = href.slice(CHUNKREF_PREFIX.length)
                                      console.log("ChunkId = ", JSON.stringify(chunkId))
                                      const reference = referenceMap.get(chunkId)
                                      console.log("Found reference? ", !!reference, reference)
                                      if (reference) {
                                        return (
                                          <button
                                            type="button"
                                            className="reference-inline-link"
                                            onClick={() => handleReferenceSelect(reference)}
                                          >
                                            {children}
                                          </button>
                                        )
                                      } else {
                                        // 异常处理：找不到引用时，渲染为不可点击文本（避免无效交互）
                                        return <span className="reference-inline-link--invalid">{children}</span>
                                      }
                                    }
                                    return (
                                      <a
                                        {...linkProps}
                                        href={href}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                      >
                                        {children}
                                      </a>
                                    )
                                  },
                                }}
                              >
                                {item.answer}
                              </ReactMarkdown>
                            </div>
                            {filteredReferences.length > 0 && (
                              <div className="reference-links">
                                {filteredReferences.map((reference, referenceIdx) => (
                                  <button
                                    key={`${item.question}-${reference.chunk_id ?? referenceIdx}`}
                                    type="button"
                                    className="reference-link"
                                    onClick={() => handleReferenceSelect(reference)}
                                  >
                                    Source {referenceIdx + 1}: {reference.file_name}
                                    {reference.page ? ` • p.${reference.page}` : ""}
                                  </button>
                                ))}
                              </div>
                            )}
                          </article>
                        )
                      })}
                    </div>
                  </div>
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
