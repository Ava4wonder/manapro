import PdfPreviewWithHighlights from "../components/PdfPreviewWithHighlights"
import QuestionAnswersTable from "../components/QuestionAnswersTable"
import TenderList from "../components/TenderList"
import { SummaryResponse } from "../api/tenders"

type Props = {
  summary: SummaryResponse | null
  documents: string[]
}

const SummaryPage = ({ summary, documents }: Props) => (
  <section className="phase-pane">
    <h2>Phase II • Summary</h2>
    {summary?.ready ? (
      <QuestionAnswersTable
        questions={summary.questions}
        emptyMessage="Highlights are being generated."
      />
    ) : (
      <p>Highlight answers are pending. Start the analysis to see them.</p>
    )}

    <TenderList documents={documents} />

    <PdfPreviewWithHighlights documents={documents} questions={summary?.questions ?? []} />
  </section>
)

export default SummaryPage
