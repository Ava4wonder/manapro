import { FormEvent, useRef, useState } from "react"
import "./UploadPage.css"

type Props = {
  isUploading: boolean
  onSubmit: (name: string, files: File[]) => Promise<void>
  tenderId: string | null
  uploadComplete: boolean
}

const UploadPage = ({ isUploading, onSubmit, tenderId, uploadComplete }: Props) => {
  const [name, setName] = useState("Private tender package")
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const filesRef = useRef<File[]>([])

  const updateSelectedFiles = (files: File[]) => {
    filesRef.current = files
    setSelectedFiles(files)
  }

  const handleFileInputChange = (event: FormEvent<HTMLInputElement>) => {
    const input = event.currentTarget
    if (input.files) {
      const newFiles = Array.from(input.files)
      updateSelectedFiles([...filesRef.current, ...newFiles])
      // Reset input so the same file can be selected again
      input.value = ""
    }
    if (error) {
      // Clear "no files selected" error as soon as the user picks something
      setError(null)
    }
  }

  const handleRemoveFile = (index: number) => {
    const updated = filesRef.current.filter((_, i) => i !== index)
    updateSelectedFiles(updated)

    if (updated.length === 0 && error) {
      // Keep "no files" style error if they remove the last file
      setError("Please select at least one document to upload.")
    }
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()

    if (isUploading) {
      console.warn(`⚠️ [UPLOAD] Upload already in progress`)
      return
    }

    const filesToUpload = filesRef.current
    if (filesToUpload.length === 0) {
      const message = "Please select at least one document to upload."
      console.warn(`⚠️ [UPLOAD] ${message}`)
      setError(message)
      return
    }

    console.log(`📤 [UPLOAD] Submit button clicked`)
    console.log(`📋 Package: "${name}"`)
    console.log(`📊 Files to upload: ${filesToUpload.length}`)
    filesToUpload.forEach((file, idx) => {
      console.log(
        `  ${idx + 1}. ${file.name} (${(file.size / 1024).toFixed(2)} KB, type: ${file.type})`,
      )
    })

    try {
      setError(null)
      console.log(`⏳ [UPLOAD] Starting upload process...`)
      await onSubmit(name, filesToUpload)
      console.log(`✅ [UPLOAD] Upload completed successfully`)

      // Clear local selection after parent confirms success
      updateSelectedFiles([])
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    } catch (err) {
      console.error(`❌ [UPLOAD] Upload error:`, err)
      setError("Upload failed. Please try again or contact support if the problem persists.")
      // Parent handles the actual error; we keep files for retry (via filesRef)
    }
  }

  const handleOpenFilePicker = () => {
    fileInputRef.current?.click()
  }

  const hasSelection = selectedFiles.length > 0
  const showSuccessState = Boolean(tenderId && uploadComplete)

  return (
    <section className="upload-pane">
      <h2>Phase I • Upload documents</h2>

      <form onSubmit={handleSubmit} noValidate>
        <div className="form-group">
          <label htmlFor="package-name">Package name</label>
          <input
            id="package-name"
            type="text"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Tender 1234 — private submission"
            disabled={isUploading}
          />
        </div>

        <div className="form-group">
          <label>Documents</label>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileInputChange}
            style={{ display: "none" }}
            accept=".pdf,.docx,.txt,.md"
            disabled={isUploading}
          />

          <div className="file-input-area">
            <button
              type="button"
              className="btn-add-files"
              onClick={handleOpenFilePicker}
              disabled={isUploading}
            >
              + Add files
            </button>
            <span className="file-count">
              {hasSelection ? `${selectedFiles.length} file(s) selected` : "No files selected"}
            </span>
          </div>

          {selectedFiles.length > 0 && (
            <div className="files-list">
              {selectedFiles.map((file, index) => (
                <div key={`${file.name}-${index}`} className="file-item">
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">
                    ({(file.size / 1024).toFixed(1)} KB)
                  </span>
                  <button
                    type="button"
                    className="btn-remove"
                    onClick={() => handleRemoveFile(index)}
                    disabled={isUploading}
                    title="Remove file"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}

          {error && (
            <p className="upload-error" role="alert">
              {error}
            </p>
          )}
        </div>

        <div className="form-actions">
          <button
            type="submit"
            disabled={isUploading || selectedFiles.length === 0}
            className="btn-upload"
          >
            {isUploading ? "Uploading…" : "Upload"}
          </button>

          <div
            className={`upload-indicator ${showSuccessState ? "complete" : ""}`}
            title={showSuccessState ? "Upload complete" : ""}
          >
            {showSuccessState && <span className="indicator-dot"></span>}
          </div>
        </div>
      </form>

      <div className="upload-status-hint">
        {showSuccessState && tenderId ? (
          <p className="success">
            Uploaded. Tender ID <strong>{tenderId}</strong> has been created.
            <br />
            Next step: start the analysis for this tender from the controls at the top.
          </p>
        ) : tenderId ? (
          <p className="info">
            Tender ID <strong>{tenderId}</strong> is available. You can re-upload documents or
            re-run the analysis as needed.
          </p>
        ) : (
          <p className="info">
            Select documents and upload them to create a new tender. Analysis is started
            separately after upload.
          </p>
        )}
      </div>
    </section>
  )
}

export default UploadPage