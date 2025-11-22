import { QuestionAnswer } from "../api/tenders"

type Props = {
  documents: string[]
  questions: QuestionAnswer[]
}

const PdfPreviewWithHighlights = ({ documents, questions }: Props) => (
  <div className="pdf-preview">
    <header>
      <h3>PDF preview</h3>
      <p>Highlights keep bounding boxes and pages in sync with the original documents.</p>
    </header>

    <div className="preview-shell">
      {documents.length === 0 ? (
        <span className="preview-placeholder">No files have been ingested yet.</span>
      ) : (
        <ul>
          {documents.map((fileName) => (
            <li key={fileName}>{fileName}</li>
          ))}
        </ul>
      )}
    </div>

    {questions.length > 0 && (
      <div className="preview-highlights">
        <h4>Highlight answers</h4>
        <ul>
          {questions.map((item) => (
            <li key={item.question}>
              <strong>{item.question}</strong>
              <p>{item.answer}</p>
            </li>
          ))}
        </ul>
      </div>
    )}
  </div>
)

export default PdfPreviewWithHighlights
