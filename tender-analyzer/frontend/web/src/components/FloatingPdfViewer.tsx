import { createPortal } from "react-dom"

import PdfPreviewWithHighlights from "./PdfPreviewWithHighlights"
import { usePdfViewerStore } from "../store/pdfViewerStore"

const FloatingPdfViewer = () => {
  const { isVisible, activeReference, tenderId, hide, toggle, clearActiveReference } =
    usePdfViewerStore()

  if (typeof document === "undefined") {
    console.log("[FloatingPdfViewer] document is undefined (SSR), render null")
    return null
  }

  console.log(
    "[FloatingPdfViewer] render, isVisible=",
    isVisible,
    "tenderId=",
    tenderId,
    "activeReference=",
    activeReference,
  )

  return (
    <>
      {createPortal(
        <button
          type="button"
          className="floating-pdf-toggle"
          onClick={() => {
            console.log("[FloatingPdfViewer] toggle clicked, next isVisible:", !isVisible)
            toggle()
          }}
          aria-pressed={isVisible}
          aria-label={isVisible ? "Hide floating PDF viewer" : "Show floating PDF viewer"}
        >
          {isVisible ? "Hide PDF viewer" : "Open PDF viewer"}
        </button>,
        document.body,
      )}
      {isVisible &&
        createPortal(
          <div
            className="floating-pdf-viewer"
            role="dialog"
            aria-modal="true"
            aria-label="Document preview"
          >
            <div className="floating-pdf-viewer__header">
              <div>
                <p className="floating-pdf-viewer__title">PDF preview</p>
                <p className="floating-pdf-viewer__subtitle">
                  Inspect highlighted references in context.
                </p>
              </div>
              <button
                type="button"
                className="floating-pdf-viewer__close"
                onClick={() => {
                  console.log("[FloatingPdfViewer] close button clicked")
                  hide()
                }}
                aria-label="Close PDF viewer"
              >
                ×
              </button>
            </div>
            <div className="floating-pdf-viewer__body">
              <PdfPreviewWithHighlights
                documents={[]}
                tenderId={tenderId}
                activeReference={activeReference}
                onClearReference={() => {
                  console.log("[FloatingPdfViewer] clearActiveReference called")
                  clearActiveReference()
                }}
              />
            </div>
          </div>,
          document.body,
        )}
    </>
  )
}

export default FloatingPdfViewer
