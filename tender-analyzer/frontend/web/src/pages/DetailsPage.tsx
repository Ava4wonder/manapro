import QuestionAnswersTable from "../components/QuestionAnswersTable"
import { DetailsResponse } from "../api/tenders"

type Props = {
  details: DetailsResponse | null
}

const DetailsPage = ({ details }: Props) => (
  <section className="phase-pane">
    <h2>Phase II • Details</h2>
    {details?.ready ? (
      <QuestionAnswersTable
        questions={details.questions}
        emptyMessage="Detailed answers are not ready yet."
      />
    ) : (
      <p>Detailed question answers will arrive once the full pipeline runs.</p>
    )}
  </section>
)

export default DetailsPage
