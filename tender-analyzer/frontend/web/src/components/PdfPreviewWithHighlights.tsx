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
  const [renderedPages, setRenderedPages] = useState<Set<number>>(new Set())
  const cacheRef = useRef<Map<string, string>>(new Map())
  const viewerRef = useRef<HTMLDivElement | null>(null)
  const pageRef = useRef<HTMLDivElement | null>(null)
  const scrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // 用 tenderId + file_name 作为当前 PDF 的唯一 key（只代表“哪个文件”）
  const docKey = useMemo(
    () =>
      activeReference && tenderId
        ? `${tenderId}:${activeReference.file_name}`
        : null,
    [tenderId, activeReference?.file_name],
  )

  // 看看 activeReference / docKey 的变化
  useEffect(() => {
    console.log("[PdfPreview] activeReference changed:", activeReference)
    console.log("[PdfPreview] docKey changed:", docKey)
  }, [activeReference, docKey])

  useEffect(() => {
    console.log("[PdfPreview] unmount, revoke all cached object URLs")
    return () => {
      cacheRef.current.forEach((url) => URL.revokeObjectURL(url))
      cacheRef.current.clear()
    }
  }, [])

  /**
   * 加载 / 切换 PDF：
   * 只依赖 docKey —— 也就是“文件变了才跑”
   * 切换文件时：重置 numPages / pageHeight / renderedPages，并加载新的 blob
   */
  useEffect(() => {
    console.log("[PdfPreview] load effect start, docKey:", docKey)

    // 先 reset 状态：只在 docKey 变化（文件变化）时才会跑到这里
    setNumPages(0)
    setPageHeight(null)
    setRenderedPages(new Set())
    console.log("[PdfPreview] reset numPages / pageHeight / renderedPages because docKey changed")

    if (!docKey) {
      console.log("[PdfPreview] no docKey, clear pdfUrl and stop loading")
      setPdfUrl(null)
      setError(null)
      setLoading(false)
      return
    }

    // 从 docKey 里拆回 tenderId 和 fileName
    const [currentTenderId, ...fileNameParts] = docKey.split(":")
    const fileName = fileNameParts.join(":") // 防止文件名里理论上有冒号

    if (!currentTenderId || !fileName) {
      console.error("[PdfPreview] invalid docKey parsed:", docKey)
      setPdfUrl(null)
      setError("Invalid document reference.")
      setLoading(false)
      return
    }

    const cacheKey = docKey
    const cachedUrl = cacheRef.current.get(cacheKey)
    if (cachedUrl) {
      console.log("[PdfPreview] cache hit for", cacheKey, "url:", cachedUrl)
      setPdfUrl(cachedUrl)
      setError(null)
      setLoading(false)
      return
    }

    console.log("[PdfPreview] cache miss for", cacheKey, "start fetching…")

    const controller = new AbortController()
    let cancelled = false
    setLoading(true)
    setError(null)

    ;(async () => {
      try {
        const encodedName = encodeURIComponent(fileName)
        const url = `${API_BASE_URL}/tenders/${currentTenderId}/documents/${encodedName}`
        console.log("[PdfPreview] fetching PDF from:", url)
        const response = await fetch(url, {
          credentials: "include",
          signal: controller.signal,
        })
        if (!response.ok) {
          const message = await response.text()
          console.error("[PdfPreview] fetch failed:", response.status, message)
          throw new Error(message || "Unable to fetch source document.")
        }
        const blob = await response.blob()
        const objectUrl = URL.createObjectURL(blob)
        console.log("[PdfPreview] fetch success, objectUrl created:", objectUrl)
        cacheRef.current.set(cacheKey, objectUrl)
        if (!cancelled) {
          console.log("[PdfPreview] setting pdfUrl (not cancelled)")
          setPdfUrl(objectUrl)
        } else {
          console.log("[PdfPreview] request cancelled, revoke new objectUrl")
          URL.revokeObjectURL(objectUrl)
        }
      } catch (err) {
        if (!cancelled) {
          const message =
            err instanceof Error ? err.message : "Failed to load document."
          console.error("[PdfPreview] error during fetch:", err)
          setError(message)
        } else {
          console.log("[PdfPreview] fetch aborted, ignore error:", err)
        }
      } finally {
        if (!cancelled) {
          console.log("[PdfPreview] setLoading(false)")
          setLoading(false)
        }
      }
    })()

    return () => {
      console.log("[PdfPreview] cleanup load effect for docKey:", docKey)
      cancelled = true
      controller.abort()
    }
  }, [docKey])

  useEffect(() => {
    const node = viewerRef.current
    if (!node || typeof window === "undefined") {
      console.log("[PdfPreview] viewerRef not ready or no window, skip resize observer")
      return
    }

    const updateWidth = () => {
      const width = node.clientWidth
      const clamped = Math.min(MAX_VIEWER_WIDTH, Math.max(MIN_VIEWER_WIDTH, width))
      console.log("[PdfPreview] updateWidth, clientWidth:", width, "clamped:", clamped)
      setViewerWidth(clamped)
    }

    updateWidth()

    let observer: ResizeObserver | null = null
    if ("ResizeObserver" in window) {
      console.log("[PdfPreview] use ResizeObserver for viewer width")
      observer = new ResizeObserver(updateWidth)
      observer.observe(node)
    } else {
      console.log("[PdfPreview] use window resize event for viewer width")
      window.addEventListener("resize", updateWidth)
    }

    return () => {
      if (observer) {
        console.log("[PdfPreview] disconnect ResizeObserver")
        observer.disconnect()
      } else {
        console.log("[PdfPreview] remove window resize listener")
        window.removeEventListener("resize", updateWidth)
      }
    }
  }, [])

  /**
   * 滚动到对应页：
   * - 依赖 docKey（切换文件）、activeReference?.page（切换页码）、renderedPages、numPages
   * - 注意：现在 numPages 不会因为“仅页码变化”被清零了
   */
  useEffect(() => {
    console.log(
      "[PdfPreview] scroll effect run, docKey:",
      docKey,
      "page:",
      activeReference?.page,
      "numPages:",
      numPages,
      "renderedPages:",
      Array.from(renderedPages),
    )

    if (!activeReference?.page || numPages === 0 || !viewerRef.current) {
      console.log("[PdfPreview] scroll effect early exit (no page/numPages/viewerRef)")
      return
    }

    if (scrollTimeoutRef.current) {
      console.log("[PdfPreview] clear existing scroll timeout")
      clearTimeout(scrollTimeoutRef.current)
    }

    const attemptScroll = () => {
      console.log(
        "[PdfPreview] attemptScroll to page:",
        activeReference.page,
        "current renderedPages:",
        Array.from(renderedPages),
      )
      const target = viewerRef.current?.querySelector(
        `#pdf-page-${activeReference.page}`,
      )
      if (target) {
        const rect = (target as HTMLElement).getBoundingClientRect()
        console.log(
          "[PdfPreview] scroll target found, rect.height:",
          rect.height,
        )
        setPageHeight(rect.height)
        ;(target as HTMLElement).scrollIntoView({
          behavior: "smooth",
          block: "start",
        })
      } else {
        console.log("[PdfPreview] scroll target not found, retry in 100ms")
        if (scrollTimeoutRef.current) {
          clearTimeout(scrollTimeoutRef.current)
        }
        scrollTimeoutRef.current = setTimeout(attemptScroll, 100)
      }
    }

    scrollTimeoutRef.current = setTimeout(() => {
      console.log("[PdfPreview] initial scroll timeout fired (50ms)")
      attemptScroll()
    }, 50)

    return () => {
      if (scrollTimeoutRef.current) {
        console.log("[PdfPreview] cleanup scroll timeout")
        clearTimeout(scrollTimeoutRef.current)
      }
    }
  }, [docKey, activeReference?.page, renderedPages, numPages])

  const pageNumbers = useMemo(
    () => Array.from({ length: numPages }, (_, idx) => idx + 1),
    [numPages],
  )

  const buildHighlightStyle = (pageNumber: number) => {
    if (!activeReference) {
      return null
    }
    if (activeReference.page !== pageNumber) {
      return null
    }

    const bbox = activeReference.bbox
    if (!bbox || bbox.length < 4) {
      console.warn("[PdfPreview] no bbox or invalid length for highlight:", bbox)
      return null
    }

    // bbox: 相对坐标 0~1
    const [relX0, relY0, relX1, relY1] = bbox

    if (
      relX0 < 0 ||
      relY0 < 0 ||
      relX1 > 1 ||
      relY1 > 1 ||
      relX0 >= relX1 ||
      relY0 >= relY1
    ) {
      console.warn("[PdfPreview] invalid relative bbox coordinates:", bbox)
      return null
    }

    const pageElement = pageRef.current?.querySelector("canvas")
    if (!pageElement) {
      console.warn("[PdfPreview] no canvas found on pageRef for highlight")
      return null
    }

    const displayWidth = (pageElement as HTMLCanvasElement).clientWidth
    const displayHeight = (pageElement as HTMLCanvasElement).clientHeight

    if (!displayWidth || !displayHeight) {
      console.warn(
        "[PdfPreview] canvas has no clientWidth/clientHeight, skip highlight",
      )
      return null
    }

    const marginPercent = 0.04
    const marginX = marginPercent
    const marginY = marginPercent

    const expandedRelX0 = Math.max(0, relX0 - marginX)
    const expandedRelY0 = Math.max(0, relY0 - marginY)
    const expandedRelX1 = Math.min(1, relX1 + marginX)
    const expandedRelY1 = Math.min(1, relY1 + marginY)

    const absoluteX0 = expandedRelX0 * displayWidth
    const absoluteY0 = expandedRelY0 * displayHeight
    const absoluteX1 = expandedRelX1 * displayWidth
    const absoluteY1 = expandedRelY1 * displayHeight

    const style = {
      left: absoluteX0,
      top: absoluteY0,
      width: absoluteX1 - absoluteX0,
      height: absoluteY1 - absoluteY0,
    }

    console.log(
      "[PdfPreview] highlight style for page",
      pageNumber,
      "bbox:",
      bbox,
      "style:",
      style,
    )

    return style
  }

  const handleDocumentLoadSuccess = ({ numPages: loadedPages }: { numPages: number }) => {
    console.log("[PdfPreview] Document loaded, numPages:", loadedPages)
    setNumPages(loadedPages)
  }

  const handlePageRenderSuccess = (pageProxy: PDFPageProxy) => {
    console.log("[PdfPreview] Page rendered:", pageProxy.pageNumber)
    setRenderedPages((prev) => {
      const next = new Set(prev).add(pageProxy.pageNumber)
      console.log("[PdfPreview] renderedPages updated:", Array.from(next))
      return next
    })

    if (pageProxy.pageNumber === activeReference?.page && activeReference?.orig_size?.[0]) {
      const viewport = pageProxy.getViewport({
        scale: viewerWidth / activeReference.orig_size[0],
      })
      console.log(
        "[PdfPreview] active page viewport height set to:",
        viewport.height,
      )
      setPageHeight(viewport.height)
    }
  }

  console.log(
    "[PdfPreview] render:",
    "docKey=",
    docKey,
    "pdfUrl=",
    pdfUrl,
    "loading=",
    loading,
    "error=",
    error,
    "numPages=",
    numPages,
    "viewerWidth=",
    viewerWidth,
    "pageHeight=",
    pageHeight,
  )

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
        style={{ maxHeight: pageHeight ? `${pageHeight + 80}px` : "auto" }}
      >
        {!activeReference || !tenderId ? (
          <span className="preview-placeholder">
            Select any reference badge to load the PDF page here.
          </span>
        ) : loading ? (
          <span className="preview-placeholder">Loading PDF.</span>
        ) : error ? (
          <span className="preview-placeholder error">{error}</span>
        ) : pdfUrl ? (
          <Document
            key={docKey ?? "no-doc"} // 文件变了才会重挂载
            file={pdfUrl}
            loading={null}
            onLoadSuccess={handleDocumentLoadSuccess}
          >
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
                  {highlightStyle && (
                    <div className="pdf-highlight" style={highlightStyle} />
                  )}
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
