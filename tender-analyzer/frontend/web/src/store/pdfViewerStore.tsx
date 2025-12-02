import { createContext, ReactNode, useCallback, useContext, useMemo, useState } from "react"
import type { QuestionAnswerReference } from "../api/tenders"

type PdfViewerStore = {
  isVisible: boolean
  tenderId: string | null
  activeReference: QuestionAnswerReference | null
  show: (reference: QuestionAnswerReference, tenderId: string | null) => void
  hide: () => void
  toggle: () => void
  clearActiveReference: () => void
}

// ✅ 更安全的默认值（可选）
const PdfViewerContext = createContext<PdfViewerStore | undefined>(undefined)

export const PdfViewerProvider = ({ children }: { children: ReactNode }) => {
  const [isVisible, setIsVisible] = useState(false)
  const [activeReference, setActiveReference] = useState<QuestionAnswerReference | null>(null)
  const [tenderId, setTenderId] = useState<string | null>(null)

  const show = useCallback((reference: QuestionAnswerReference, tid: string | null) => {
    setActiveReference(reference)
    setTenderId(tid)
    setIsVisible(true)
  }, [])

  const hide = useCallback(() => {
    setIsVisible(false)
    // 可选：是否同时清空？
    // setActiveReference(null)
    // setTenderId(null)
  }, [])

  const toggle = useCallback(() => {
    setIsVisible(prev => !prev)
  }, [])

  const clearActiveReference = useCallback(() => {
    setActiveReference(null)
  }, [])

  const value = useMemo<PdfViewerStore>(
    () => ({
      isVisible,
      tenderId,
      activeReference,
      show,
      hide,
      toggle,
      clearActiveReference,
    }),
    [isVisible, tenderId, activeReference]
    // 注意：useCallback 依赖为空数组，所以它们是稳定的，无需放入依赖
  )

  return <PdfViewerContext.Provider value={value}>{children}</PdfViewerContext.Provider>
}

export const usePdfViewerStore = (): PdfViewerStore => {
  const context = useContext(PdfViewerContext)
  if (!context) {
    throw new Error("usePdfViewerStore must be used inside a PdfViewerProvider")
  }
  return context
}