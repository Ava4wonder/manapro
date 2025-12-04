import { FC } from "react"
import { ProjectCardInfo, PROJECT_CARD_FIELD_KEYS } from "../types/project"
import { formatProjectDate, PROJECT_CARD_FIELD_LABELS } from "../utils/projects"

type ProjectCardProps = {
  project: ProjectCardInfo
  isActive?: boolean
  onClick: () => void
  onDelete?: () => void
}

const ProjectCard: FC<ProjectCardProps> = ({ project, isActive = false, onClick, onDelete }) => {
  const { name, createdAt, documents, summaryPreview, cardFields, analysisStatus } = project
  const fieldRows = PROJECT_CARD_FIELD_KEYS.map((key) => ({
    key,
    label: PROJECT_CARD_FIELD_LABELS[key],
    value: cardFields[key],
  }))

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
        {onDelete && (
          <button
            type="button"
            className="project-card__delete"
            onClick={(e) => {
              e.stopPropagation()
              onDelete()
            }}
          >
            Delete
          </button>
        )}
      </header>

      <div className="project-card__body">
        {fieldRows.map(({ key, label, value }) => (
          <div key={key} className="project-card__line">
            <span className="project-card__label">{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
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
