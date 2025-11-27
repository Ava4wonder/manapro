import { QuestionAnswer } from "../api/tenders"
import ReactMarkdown from 'react-markdown';


type Props = {
  questions?: QuestionAnswer[]
  emptyMessage?: string
}

const QuestionAnswersTable = ({ questions = [], emptyMessage = "No answers yet." }: Props) => (
  <div className="question-table">
    <h3>Question &amp; answer pairs</h3>
    {questions.length === 0 ? (
      <p>{emptyMessage}</p>
    ) : (
      <div className="question-grid">
        {questions.map((item) => (
          <article key={item.question}>
            <h4>{item.question}</h4>
            <ReactMarkdown>{item.answer}</ReactMarkdown>
          </article>
        ))}
      </div>
    )}
  </div>
)

export default QuestionAnswersTable
