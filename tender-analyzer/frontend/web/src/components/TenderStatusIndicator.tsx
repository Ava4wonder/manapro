import { TenderStatusResponse } from "../api/tenders"

type Props = {
  status: TenderStatusResponse | null
  loading?: boolean
}

const formatState = (value: string) =>
  value.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (char) => char.toUpperCase())

const TenderStatusIndicator = ({ status, loading }: Props) => {
  if (loading) {
    return <div className="status-indicator status-blank">Waiting for a tender...</div>
  }

  if (!status) {
    return <div className="status-indicator status-idle">Upload files to get started.</div>
  }

  return (
    <div className="status-indicator status-ready">
      <div>
        State: <strong>{formatState(status.state)}</strong>
      </div>
      <div>Progress: {status.progress}%</div>
    </div>
  )
}

export default TenderStatusIndicator
