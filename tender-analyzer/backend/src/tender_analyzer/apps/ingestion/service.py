import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tender_analyzer.apps.qa_engine.service import AnalysisService
from tender_analyzer.common.state.enums import TenderState
from tender_analyzer.common.state.state_machine import TenderStateMachine
from tender_analyzer.common.utils.ids import generate_id
from tender_analyzer.common.utils.time import now_iso
from tender_analyzer.domain.models import StoredDocument, Tender
from tender_analyzer.domain.repositories import tender_repo
from tender_analyzer.common.config.settings import settings
from tender_analyzer.common.vectorstore.qdrant_client import QdrantVectorStore

LOG = logging.getLogger(__name__)


try:
    from fastapi import UploadFile
    from fastapi.concurrency import run_in_threadpool
except ImportError:
    UploadFile = Any

    async def run_in_threadpool(func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


class ProcessingService:
    def __init__(self, repository=None) -> None:
        self.repository = repository or tender_repo
        self.analysis_service = AnalysisService()
        self.state_machine = TenderStateMachine()
        self.storage_root = Path(__file__).resolve().parents[4] / "storage"
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.vector_store = QdrantVectorStore(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            vector_dim=max(1, settings.QDRANT_VECTOR_DIM),
            prefer_grpc=settings.QDRANT_USE_GRPC,
        )

    async def upload_package(
        self, name: str, tenant_id: str, files: List[UploadFile] | None = None
    ) -> Tender:
        files = files or []
        tender_id = generate_id("tender")
        LOG.info(f"🚀 [UPLOAD START] Processing package: '{name}' | Tender ID: {tender_id} | Tenant ID: {tenant_id}")
        LOG.info(f"📁 [FILES] Received {len(files)} file(s) for upload")
        for idx, f in enumerate(files, 1):
            LOG.info(f"  {idx}. {f.filename} (size: {f.size} bytes)")

        target_dir = self.storage_root / tenant_id / tender_id
        target_dir.mkdir(parents=True, exist_ok=True)
        LOG.info(f"📂 [STORAGE] Created storage directory: {target_dir}")

        documents: List[StoredDocument] = []
        corpus_parts: List[str] = []
        vector_targets: List[Tuple[str, List[Dict[str, Any]]]] = []

        for idx, uploaded in enumerate(files, 1):
            LOG.info(f"⏳ [FILE {idx}/{len(files)}] Processing: {uploaded.filename}")
            file_path = target_dir / uploaded.filename
            content = await uploaded.read()
            file_path.write_bytes(content)
            LOG.info(f"✅ [FILE {idx}] Saved to: {file_path}")

            doc_id = generate_id("doc")
            documents.append(
                StoredDocument(
                    id=doc_id,
                    name=uploaded.filename,
                    storage_path=str(file_path),
                    uploaded_at=now_iso(),
                )
            )
            LOG.debug(f"📄 [FILE {idx}] Document record created: {doc_id}")

            LOG.info(f"🔄 [FILE {idx}] Chunking document...")
            chunks = await self._chunk_document(file_path)
            LOG.info(f"✂️ [FILE {idx}] Generated {len(chunks)} chunks from {uploaded.filename}")

            chunk_text = self._corpus_from_chunks(chunks)
            if chunk_text:
                corpus_parts.append(chunk_text)
                LOG.debug(f"📝 [FILE {idx}] Corpus text created (length: {len(chunk_text)} chars)")
            if chunks:
                vector_targets.append((uploaded.filename, chunks))
                LOG.info(f"📦 [FILE {idx}] Added to vector targets")

        LOG.info(f"🔗 [UPSERT] Starting Qdrant upsert for {len(vector_targets)} file(s)...")
        inserted_vectors = self._upsert_chunks(tenant_id, tender_id, vector_targets)
        LOG.info(f"✨ [UPSERT] Successfully upserted {inserted_vectors} vector chunks")
        if inserted_vectors:
            LOG.debug("Stored %d vector chunks for tender %s", inserted_vectors, tender_id)

        analysis_corpus = " ".join(corpus_parts).strip()
        if not analysis_corpus:
            fallback = " ".join(document.name for document in documents)
            analysis_corpus = fallback or name or ""
        LOG.info(f"📄 [CORPUS] Analysis corpus created (length: {len(analysis_corpus)} chars)")

        tender = Tender(
            id=tender_id,
            name=name,
            tenant_id=tenant_id,
            created_at=now_iso(),
            state=TenderState.INGESTED,
            documents=documents,
            highlight_answers=[],
            full_answers=[],
            evaluation=None,
            analysis_corpus=analysis_corpus,
        )

        LOG.info(f"💾 [DB] Saving tender to database...")
        self.repository.create(tender)
        LOG.info(f"✅ [DB] Tender saved successfully | ID: {tender_id} | State: INGESTED")

        try:
            LOG.info(f"🔄 [STATE] Transitioning to SUMMARY_RUNNING...")
            self._transition_state(tender_id, TenderState.SUMMARY_RUNNING)
            LOG.info(f"⏳ [ANALYSIS] Running highlight QA analysis...")
            highlight_answers = self.analysis_service.run_highlight_qa(tender)
            LOG.info(f"✨ [ANALYSIS] Generated {len(highlight_answers)} highlight answers")
            self.repository.update_highlight_answers(tender_id, highlight_answers)
            self._transition_state(tender_id, TenderState.SUMMARY_READY)
            LOG.info(f"✅ [SUMMARY] Summary ready | Tender ID: {tender_id}")
        except Exception as exc:
            LOG.error(f"❌ [ERROR] Analysis failed: {exc}")
            self.repository.set_state(tender_id, TenderState.FAILED)
            raise

        LOG.info(f"🎉 [COMPLETE] Upload and processing complete for tender: {tender_id}")
        return self.repository.get(tender_id) or tender

    async def _chunk_document(self, file_path: Path) -> List[Dict[str, Any]]:
        try:
            from tender_analyzer.apps.ingestion.chunking.coarse_to_fine import create_chunks_coarsetofine
        except ImportError as exc:
            LOG.warning("❌ [CHUNK] Chunker unavailable during ingestion: %s", exc)
            return []

        try:
            LOG.debug(f"🔄 [CHUNK] Starting coarse-to-fine chunking for: {file_path.name}")
            chunks = await run_in_threadpool(create_chunks_coarsetofine, str(file_path))
            LOG.debug(f"✅ [CHUNK] Successfully created {len(chunks)} chunks from {file_path.name}")
        except Exception as exc:
            LOG.error(f"❌ [CHUNK] Failed to chunk {file_path.name}: {exc}")
            return []

        return chunks or []

    def _corpus_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return ""
        texts = [str(chunk.get("text") or "").strip() for chunk in chunks]
        texts = [t for t in texts if t]
        return " ".join(texts).strip()

    def _upsert_chunks(
        self,
        tenant_id: str,
        tender_id: str,
        vector_targets: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> int:
        if not vector_targets:
            LOG.info(f"⚠️ [UPSERT] No vector targets to upsert")
            return 0

        total = 0
        LOG.info(f"🔗 [UPSERT] Starting upsert to Qdrant | Tenant: {tenant_id} | Tender: {tender_id}")
        for idx, (source, chunks) in enumerate(vector_targets, 1):
            try:
                LOG.debug(f"📤 [UPSERT {idx}] Processing: {source} ({len(chunks)} chunks)")
                upserted = self.vector_store.upsert_chunks(tenant_id, tender_id, source, chunks)
                total += upserted
                LOG.info(f"✅ [UPSERT {idx}] Upserted {upserted} chunks from {source}")
            except Exception as exc:
                LOG.error(f"❌ [UPSERT] Vector store upsert failed for {source}: {exc}")
        
        LOG.info(f"🎯 [UPSERT] Total chunks upserted: {total} vectors")
        return total

    def _transition_state(self, tender_id: str, target: TenderState) -> None:
        tender = self.repository.get(tender_id)
        if not tender:
            return
        try:
            self.state_machine.transition(tender.state, target)
        except ValueError:
            self.repository.set_state(tender_id, TenderState.FAILED)
            raise
        self.repository.set_state(tender_id, target)
