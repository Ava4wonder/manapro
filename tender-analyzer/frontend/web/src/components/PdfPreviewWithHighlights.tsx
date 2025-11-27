import { useEffect, useMemo, useRef, useState } from "react"
import { Document, Page, pdfjs } from "react-pdf"
import type { PDFPageProxy } from "pdfjs-dist/types/src/display/api"

import { API_BASE_URL } from "../api/client"
import { QuestionAnswerReference } from "../api/tenders"

pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`

const MIN_VIEWER_WIDTH = 280
const MAX_VIEWER_WIDTH = 520

type Props = {
  documents: string[]
  tenderId: string | null
  activeReference: QuestionAnswerReference | null
  onClearReference?: () => void
}

const PdfPreviewWithHighlights = ({ documents, tenderId, activeReference, onClearReference }: Props) => {
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [numPages, setNumPages] = useState(0)
  const [viewerWidth, setViewerWidth] = useState(MAX_VIEWER_WIDTH)
  const [pageHeight, setPageHeight] = useState<number | null>(null)
  const cacheRef = useRef<Map<string, string>>(new Map())
  const viewerRef = useRef<HTMLDivElement | null>(null)
  const pageRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    return () => {
      cacheRef.current.forEach((url) => URL.revokeObjectURL(url))
      cacheRef.current.clear()
    }
  }, [])

  useEffect(() => {
    setNumPages(0)
    setPageHeight(null)
    if (!activeReference || !tenderId) {
      setPdfUrl(null)
      setError(null)
      setLoading(false)
      return
    }

    const cacheKey = `${tenderId}:${activeReference.file_name}`
    const cachedUrl = cacheRef.current.get(cacheKey)
    if (cachedUrl) {
      setPdfUrl(cachedUrl)
      setError(null)
      return
    }

    const controller = new AbortController()
    let cancelled = false
    setLoading(true)
    setError(null)

    ;(async () => {
      try {
        const encodedName = encodeURIComponent(activeReference.file_name)
        const response = await fetch(`${API_BASE_URL}/tenders/${tenderId}/documents/${encodedName}`, {
          credentials: "include",
          signal: controller.signal,
        })
        if (!response.ok) {
          const message = await response.text()
          throw new Error(message || "Unable to fetch source document.")
        }
        const blob = await response.blob()
        const objectUrl = URL.createObjectURL(blob)
        cacheRef.current.set(cacheKey, objectUrl)
        if (!cancelled) {
          setPdfUrl(objectUrl)
        } else {
          URL.revokeObjectURL(objectUrl)
        }
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : "Failed to load document."
          setError(message)
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    })()

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [activeReference, tenderId])

  useEffect(() => {
    const node = viewerRef.current
    if (!node || typeof window === "undefined") {
      return
    }

    const updateWidth = () => {
      const width = node.clientWidth
      const clamped = Math.min(MAX_VIEWER_WIDTH, Math.max(MIN_VIEWER_WIDTH, width))
      setViewerWidth(clamped)
    }

    updateWidth()

    let observer: ResizeObserver | null = null
    if ("ResizeObserver" in window) {
      observer = new ResizeObserver(updateWidth)
      observer.observe(node)
    } else {
      window.addEventListener("resize", updateWidth)
    }

    return () => {
      if (observer) {
        observer.disconnect()
      } else {
        window.removeEventListener("resize", updateWidth)
      }
    }
  }, [])

  useEffect(() => {
    if (!activeReference?.page || numPages === 0 || !viewerRef.current) {
      return
    }

    const target = viewerRef.current.querySelector(`#pdf-page-${activeReference.page}`)
    if (target) {
      // Calculate the height of the current page container
      const rect = target.getBoundingClientRect()
      setPageHeight(rect.height)
      
      // Scroll to the specific page
      target.scrollIntoView({ behavior: "smooth", block: "start" })
    }
  }, [activeReference, numPages])

  const pageNumbers = useMemo(() => Array.from({ length: numPages }, (_, idx) => idx + 1), [numPages])

  const buildHighlightStyle = (pageNumber: number) => {
    if (!activeReference) {
      return null
    }
    if (activeReference.page !== pageNumber) {
      return null
    }
    const bbox = activeReference.bbox
    if (!bbox || bbox.length < 4) {
      return null
    }
    const origSize = activeReference.orig_size
    if (!origSize || origSize.length < 2) {
      return null
    }
    const [origWidth, origHeight] = origSize
    if (!origWidth || !origHeight) {
      return null
    }
    const scale = viewerWidth / origWidth
    const [x0, y0, x1, y1] = bbox
    return {
      left: x0 * scale,
      top: y0 * scale,
      width: (x1 - x0) * scale,
      height: (y1 - y0) * scale,
    }
  }

  const handleDocumentLoadSuccess = ({ numPages: loadedPages }: { numPages: number }) => {
    setNumPages(loadedPages)
  }

  const handlePageRenderSuccess = (pageProxy: PDFPageProxy) => {
    if (pageProxy.pageNumber === activeReference?.page) {
      // Calculate actual rendered page height
      const viewport = pageProxy.getViewport({ scale: viewerWidth / (activeReference.orig_size?.[0] || 1) })
      setPageHeight(viewport.height)
    }
  }

  return (
    <div className="pdf-preview">
      <div className="pdf-preview__topbar">
        <span className="pdf-preview__label">PDF</span>
        {onClearReference && (
          <button type="button" onClick={onClearReference} className="pdf-preview__close">
            ×
          </button>
        )}
      </div>

      <div 
        className="pdf-viewer" 
        ref={viewerRef}
        style={{ maxHeight: pageHeight ? `${pageHeight + 80}px` : 'auto' }} // Add padding for top bar
      >
        {!activeReference || !tenderId ? (
          <span className="preview-placeholder">Select any reference badge to load the PDF page here.</span>
        ) : loading ? (
          <span className="preview-placeholder">Loading PDF.</span>
        ) : error ? (
          <span className="preview-placeholder error">{error}</span>
        ) : pdfUrl ? (
          <Document file={pdfUrl} loading={null} onLoadSuccess={handleDocumentLoadSuccess}>
            {pageNumbers.map((pageNumber) => {
              const highlightStyle = buildHighlightStyle(pageNumber)
              return (
                <div 
                  key={`page-${pageNumber}`} 
                  id={`pdf-page-${pageNumber}`} 
                  className="pdf-page-wrapper"
                  ref={pageNumber === activeReference.page ? pageRef : null}
                >
                  <Page
                    pageNumber={pageNumber}
                    width={viewerWidth}
                    renderAnnotationLayer={false}
                    renderTextLayer={false}
                    onRenderSuccess={handlePageRenderSuccess}
                  />
                  {highlightStyle && <div className="pdf-highlight" style={highlightStyle} />}
                </div>
              )
            })}
          </Document>
        ) : (
          <span className="preview-placeholder">PDF preview unavailable.</span>
        )}
      </div>
    </div>
  )
}

export default PdfPreviewWithHighlights



