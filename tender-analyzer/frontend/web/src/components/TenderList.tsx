type Props = {
  documents: string[]
}

const TenderList = ({ documents }: Props) => (
  <div className="tender-list">
    <h3>Uploaded documents</h3>
    {documents.length === 0 ? (
      <p>No documents are available yet.</p>
    ) : (
      <ul>
        {documents.map((doc) => (
          <li key={doc}>{doc}</li>
        ))}
      </ul>
    )}
  </div>
)

export default TenderList
