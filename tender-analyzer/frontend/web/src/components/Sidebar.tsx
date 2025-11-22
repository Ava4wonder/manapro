// frontend/web/src/components/Sidebar.tsx
import { FC } from "react"

type SidebarItem = {
  id: string
  label: string
  description?: string
}

type SidebarProps = {
  items: SidebarItem[]
  activeItem: string
  onChange: (id: string) => void
}

const Sidebar: FC<SidebarProps> = ({ items, activeItem, onChange }) => {
  return (
    <aside className="app-sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand__logo">TA</div>
        <div>
          <p>Tender Analyzer</p>
          <small>Go / No-Go assistant</small>
        </div>
      </div>

      <nav className="sidebar-nav">
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`sidebar-nav__item ${
              item.id === activeItem ? "is-active" : ""
            }`}
            onClick={() => onChange(item.id)}
          >
            <span>{item.label}</span>
            {item.description && <small>{item.description}</small>}
          </button>
        ))}
      </nav>

      <footer className="sidebar-footer">
        <span>Phase I–III pipeline</span>
        <p className="sidebar-footer__hint">
          Upload tenders, review project cards, then dive into summaries &amp; details.
        </p>
      </footer>
    </aside>
  )
}

export default Sidebar
