import type { SummaryResponse } from "../api/tenders"

type Props = {
  summary: SummaryResponse | null
  documents: string[]
  tenderId: string | null
}

const SummaryPage = (_props: Props) => (
  <section className="phase-pane summary-page" aria-label="Summary hub" />
)

export default SummaryPage
