import { FC } from "react"
import { ProjectCardInfo, PROJECT_CARD_FIELD_KEYS } from "../types/project"
import { formatProjectDate, PROJECT_CARD_FIELD_LABELS } from "../utils/projects"

// Trashcan icon
// import { FaTrashCan } from 'react-icons/fa';

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
            {/* 直接嵌入SVG */}
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#598580" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
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
