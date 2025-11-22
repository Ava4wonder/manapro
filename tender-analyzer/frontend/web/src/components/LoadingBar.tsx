type Props = {
  progress: number
}

const LoadingBar = ({ progress }: Props) => (
  <div className="loading-bar" aria-label={`Progress ${progress}%`}>
    <div className="loading-bar__fill" style={{ width: `${progress}%` }} />
  </div>
)

export default LoadingBar
