# Upload Feature Implementation Guide

## Overview

The upload feature has been fully implemented to support file management, uploading, and automatic Qdrant upsert with visual feedback.

## Features Implemented

### 1. **Enhanced UploadPage Component** (`frontend/web/src/pages/UploadPage.tsx`)

**Key Features:**
- **File Selection**: Click "+ Add files" button to open file picker (supports `.pdf`, `.docx`, `.txt`, `.md`)
- **File Display Box**: Shows all selected files with:
  - File name and size in KB
  - Remove button (✕) for each file
  - Disabled state during upload
- **Upload Button**: 
  - Shows "Uploading…" during upload
  - Displays "Upload" when ready
  - Disabled when no files selected or during upload
- **Success Indicator**: Green light (`.indicator-dot`) appears after upload completes
  - Shows with glow effect and smooth transition
  - Automatically resets after navigation

### 2. **Styling** (`frontend/web/src/pages/UploadPage.css`)

Professional UI with:
- Responsive form layout
- Hover effects on buttons
- File list with scroll support (max-height: 300px)
- Color-coded elements:
  - **Blue**: Add files button
  - **Green**: Upload button
  - **Red**: Remove file buttons
  - **Green glow**: Success indicator
- Accessibility features (disabled states, transitions)

### 3. **App.tsx State Management**

**New State Variables:**
```typescript
const [uploadComplete, setUploadComplete] = useState(false)
```

**State Flow:**
1. User selects files → `uploadComplete = false`
2. User clicks upload → `isUploading = true`
3. Upload succeeds → `isUploading = false`, `uploadComplete = true`
4. Green light displays
5. Navigation resets `uploadComplete` flag after 1 second

### 4. **API Integration** (`frontend/web/src/api/tenders.ts`)

**Updated Function:**
```typescript
export async function uploadTender(name: string, files: File[]) {
  const form = new FormData()
  form.append("name", name)
  files.forEach((file) => {
    form.append("files", file)
  })
  return apiRequest<{ id: string }>("/tenders", {
    method: "POST",
    body: form,
  })
}
```

Changed from `FileList` to `File[]` for better type safety and state management.

## Backend Processing Flow

The backend automatically handles the following when files are uploaded:

### 1. **File Storage** (`ProcessingService.upload_package`)
- Saves files to `storage/{tenant_id}/{tender_id}/`
- Creates `StoredDocument` entries for tracking

### 2. **Document Chunking**
- Uses `coarse_to_fine` chunking strategy (as per requirements)
- Extracts semantic chunks from PDFs, DOCX, TXT, MD
- Preserves document structure and metadata

### 3. **Qdrant Vector Store Upsert** (`_upsert_chunks`)
- Converts chunks to embeddings
- Stores in Qdrant collection: `{tenant_id}/{tender_id}`
- Tracks insertion count and logs any failures
- Non-blocking: continues on individual file failures

### 4. **State Management**
- Initial state: `TenderState.INGESTED`
- After upsert: `TenderState.SUMMARY_READY` (optional)
- Failed uploads: `TenderState.FAILED`

## User Workflow

### Upload Flow:
1. **Select Package Name** → Enter description or use default
2. **Add Files** → Click "+ Add files" button
   - Multiple files can be added incrementally
   - Each file shows with size
   - Can remove individual files before upload
3. **Upload** → Click "Upload" button
   - Button shows "Uploading…"
   - File box greyed out
   - Cannot modify files or name
4. **Success** → Green indicator light appears
   - Confirms upload complete
   - Automatic Qdrant upsert finished
   - Ready for analysis
5. **Next Steps** → Navigate to "Project cards" tab
   - New tender appears in project list
   - Can start analysis immediately

## Technical Details

### Frontend Flow:
```
User selects files
    ↓
Files stored in component state (File[])
    ↓
User clicks Upload
    ↓
handleUpload() → uploadTender() API call
    ↓
Backend processes upload & upsert
    ↓
Response with tender ID
    ↓
setUploadComplete(true) + setUploading(false)
    ↓
Green indicator appears
    ↓
Navigate to projects (after 1 sec reset)
```

### Backend Flow:
```
POST /api/tenders
    ↓
Save files to storage
    ↓
Extract document chunks (coarse_to_fine)
    ↓
Create vector embeddings
    ↓
Upsert to Qdrant {tenant_id}/{tender_id}
    ↓
Store Tender in database
    ↓
Return tender ID
```

## File Support

| Format | Support | Processing |
|--------|---------|------------|
| `.pdf` | ✓ | Coarse-to-fine chunking |
| `.docx` | ✓ | Text extraction + chunking |
| `.txt` | ✓ | Direct chunking |
| `.md` | ✓ | Direct chunking |
| Other | ✗ | Skipped with warning |

## Error Handling

- **No files selected**: Upload button disabled
- **Individual file chunk failure**: Logged, processing continues
- **Qdrant upsert failure**: Logged as warning, doesn't block upload
- **Upload API error**: Caught in try-finally, button re-enabled

## Performance Considerations

- **File chunking**: Async with `run_in_threadpool` (non-blocking)
- **Qdrant upsert**: Asynchronous batch operations
- **Frontend feedback**: Immediate visual feedback with loading state
- **State reset**: Automatic reset after navigation (1s delay for UI feedback)

## Environment Variables (Backend)

```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=optional
QDRANT_VECTOR_DIM=1536
QDRANT_USE_GRPC=false
```

## Testing Recommendations

1. **Single file upload**: PDF, DOCX, TXT
2. **Multiple files**: 2-3 files in batch
3. **File removal**: Add and remove files before upload
4. **State transitions**: Verify button states during upload
5. **Qdrant verification**: Check collection created with correct ID format
6. **Error scenarios**: Network failure, invalid file type, oversized files

## Future Enhancements

- Progress bar showing upload percentage
- Individual file upload status
- Drag-and-drop file upload
- File type validation before upload
- Estimated processing time
- Cancellation of in-progress upload
