import { FC } from "react"
import { ProjectCardInfo } from "../types/project"
import { formatProjectDate } from "../utils/projects"

type ProjectCardProps = {
  project: ProjectCardInfo
  isActive?: boolean
  onClick: () => void
}

const ProjectCard: FC<ProjectCardProps> = ({ project, isActive = false, onClick }) => {
  const { name, createdAt, documents, summaryPreview, cardFields, analysisStatus } = project
  const {
    type,
    location,
    logisticsVariant,
    deadline,
    submission,
    budget,
    budgetTag,
    evaluation,
    eligibilityChips,
    riskChips,
  } = cardFields

  const logisticsLabel =
    logisticsVariant === "challenging" ? "Challenging" : logisticsVariant === "ok" ? "OK" : "Unknown"

  return (
    <article
      className={`project-card ${isActive ? "is-active" : ""}`}
      onClick={onClick}
      aria-pressed={isActive}
    >
      <header className="project-card__header">
        <div>
          <p className={`project-card__status-pill project-card__status-pill--${analysisStatus.color}`}>
            <span className="project-card__status-dot" />
            {analysisStatus.label}
          </p>
          <h3>{name}</h3>
        </div>
        <div className="project-card__dates">
          <span>{documents} docs</span>
          <span>{formatProjectDate(createdAt)}</span>
        </div>
      </header>

      <div className="project-card__body">
        <div className="project-card__line">
          <span className="project-card__label">Type</span>
          <strong>{type}</strong>
        </div>
        <div className="project-card__line project-card__line--location">
          <span className="project-card__label">Location</span>
          <div>
            <strong>{location}</strong>
            <span className={`project-card__logistics project-card__logistics--${logisticsVariant}`}>
              Logistics: {logisticsLabel}
            </span>
          </div>
        </div>
        <div className="project-card__line">
          <span className="project-card__label">Deadline</span>
          <span>{deadline}</span>
        </div>
        <div className="project-card__line">
          <span className="project-card__label">Submission</span>
          <span>{submission}</span>
        </div>
        <div className="project-card__line">
          <span className="project-card__label">Budget</span>
          <div className="project-card__budget-row">
            <strong>{budget}</strong>
            <span className="project-card__budget-tag">{budgetTag}</span>
          </div>
        </div>
        <div className="project-card__line">
          <span className="project-card__label">Evaluation</span>
          <span>{evaluation}</span>
        </div>
      </div>

      <div className="project-card__chips">
        {eligibilityChips.length > 0 && (
          <div className="project-card__chip-group">
            {eligibilityChips.map((chip) => (
              <span key={chip} className="project-card__chip project-card__chip--eligibility">
                {chip}
              </span>
            ))}
          </div>
        )}
        {riskChips.length > 0 && (
          <div className="project-card__chip-group">
            {riskChips.map((chip) => (
              <span key={chip} className="project-card__chip project-card__chip--risk">
                {chip}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="project-card__tooltip">
        {summaryPreview ? (
          <>{summaryPreview}</>
        ) : (
          <>No summary available yet. Run analysis to generate highlights.</>
        )}
      </div>
    </article>
  )
}

export default ProjectCard
