import { EvaluationResponse } from "../api/tenders"

type Props = {
  evaluation: EvaluationResponse | null
}

const EvaluationPage = ({ evaluation }: Props) => (
  <section className="phase-pane">
    <h2>Phase III • Evaluation & suggestions</h2>
    {evaluation?.ready && evaluation.evaluation ? (
      <div className="evaluation-panel">
        <p>{evaluation.evaluation.summary}</p>
        <ul>
          <li>
            <strong>Risk level:</strong> {evaluation.evaluation.risk_level}
          </li>
          <li>
            <strong>Recommendation:</strong> {evaluation.evaluation.recommendation}
          </li>
        </ul>
      </div>
    ) : (
      <p>The evaluation step is still running. Check back once the full analysis is done.</p>
    )}
  </section>
)

export default EvaluationPage
