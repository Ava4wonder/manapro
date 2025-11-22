// frontend/web/src/pages/ProjectCardsPage.tsx
import { FC, useMemo, useState } from "react"
import { ProjectCardInfo } from "../types/project"
import { SummaryResponse, DetailsResponse } from "../api/tenders"
import ProjectCard from "../components/ProjectCard"

type ProjectCardsPageProps = {
  projects: ProjectCardInfo[]
  selectedProjectId: string | null
  onSelectProject: (projectId: string) => void
  onClearSelection: () => void
  summary: SummaryResponse | null
  details: DetailsResponse | null
}

type DetailTab = "summary" | "details"

const ProjectCardsPage: FC<ProjectCardsPageProps> = ({
  projects,
  selectedProjectId,
  onSelectProject,
  onClearSelection,
  summary,
  details,
}) => {
  const [activeTab, setActiveTab] = useState<DetailTab>("summary")

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  )

  const handleCardClick = (id: string) => {
    if (id === selectedProjectId) return
    setActiveTab("summary")
    onSelectProject(id)
  }

  return (
    <div className="project-cards-page">
      <header>
        <h2>Project cards</h2>
        <p>Browse all uploaded tenders and jump into their Q&amp;A views.</p>
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
              <button
                type="button"
                className="project-detail__close"
                onClick={onClearSelection}
              >
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
                <div className="question-grid">
                  {summary.questions.map((item, idx) => (
                    <article key={idx}>
                      <h4>{item.question}</h4>
                      <p>{item.answer}</p>
                    </article>
                  ))}
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
                Detailed Q&amp;A not ready yet. Start or re-run analysis from the Upload page.
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
