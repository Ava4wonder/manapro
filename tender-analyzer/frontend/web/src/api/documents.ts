export async function fetchDocumentPreview(name: string) {
  return {
    name,
    preview: `Preview for ${name} will be available once the processing service runs.`,
  }
}
