# backend/src/tender_analyzer/apps/ingestion/workers/ingestion_worker.py

import os
import json
import logging
from typing import Any, Dict, List, Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from tender_analyzer.apps.ingestion.chunking.coarse_to_fine import create_chunks_coarsetofine
from tender_analyzer.apps.ingestion.embedding.embedder import Embedder
from tender_analyzer.apps.qa_analysis.field_info import build_project_card_fields
from tender_analyzer.domain.repositories import tender_repo
from tender_analyzer.common.state.enums import TenderState

# 🔧 adjust this import path to where your label code lives
from tender_analyzer.apps.ingestion.workers.generate_chunk_labels import (
    generate_chunk_label,
    SYSTEM_MSG,
)

LOG = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Qdrant config – consistent with main.py (local 6333 by default)
# -------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Default embedding model; keep in sync with AVAILABLE_EMBED_MODELS in main.py
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")

# Default collection name for tender chunks
DEFAULT_COLLECTION = os.getenv("TENDER_COLLECTION", "tender_chunks")


def get_qdrant() -> QdrantClient:
    """
    Minimal clone of main.py's get_qdrant(), pinned to local qdrant on 6333.
    """
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# -------------------------------------------------------------------
# Qdrant helpers – adapted from main.py
# -------------------------------------------------------------------

def _create_collection_compat(qc: QdrantClient, name: str, vector_dim: int) -> None:
    """
    Create a collection compatible with both newer and older qdrant-client versions.
    Mirrors main.py behavior (prefers vectors_config, falls back to vectors).
    """
    try:
        qc.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        return
    except AssertionError as e:
        if "vectors_config" not in str(e):
            raise
    # Old API
    qc.create_collection(
        collection_name=name,
        vectors=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )


def _get_collection_vector_size(qc: QdrantClient, name: str) -> int | None:
    """
    Robustly read the vector size for a collection, same idea as get_collection_vector_size()
    in main.py.
    """
    try:
        info = qc.get_collection(name)
    except Exception:
        return None

    params = getattr(info, "config", None)
    params = getattr(params, "params", params)
    if params is None:
        return None

    # Try new / old attributes
    vecs = getattr(params, "vectors_config", None)
    if vecs is None:
        vecs = getattr(params, "vectors", None)

    # Direct size
    size = getattr(vecs, "size", None)
    if isinstance(size, int):
        return size

    # Mapping of named vectors
    if isinstance(vecs, dict) and vecs:
        first = next(iter(vecs.values()))
        if hasattr(first, "size"):
            return getattr(first, "size", None)
        if isinstance(first, dict):
            return first.get("size")

    try:
        for v in vecs.values():  # type: ignore[attr-defined]
            s = getattr(v, "size", None)
            if isinstance(s, int):
                return s
    except Exception:
        pass

    return None


def ensure_collection_exact(name: str, vector_dim: int) -> None:
    """
    Ensure collection exists and has the exact vector_dim we're about to upsert.
    This matches main.py's semantics.
    """
    qc = get_qdrant()
    existing = {c.name for c in qc.get_collections().collections or []}
    if name not in existing:
        LOG.info("[qdrant] Creating collection %r with dim=%d", name, vector_dim)
        _create_collection_compat(qc, name, vector_dim)
        return

    size = _get_collection_vector_size(qc, name)
    if size is not None and size != vector_dim:
        raise RuntimeError(
            f"Collection '{name}' has vector size {size}, but embeddings are {vector_dim}. "
            f"Use a different collection or recreate this one for the selected embedding model."
        )

def _slug(value: str) -> str:
    """Normalize string for safe use as collection name or identifier."""
    if not value:
        return "unknown"
    sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    return sanitized[:32] or "unknown"

def _normalize_bbox_relative(bbox: Any, orig_size: Any) -> Any:
    """
    Convert absolute bbox [x1, y1, x2, y2] into relative coordinates
    by dividing by (width, height) from orig_size.

    If anything is missing / malformed, returns bbox unchanged.
    """
    if not bbox or orig_size is None:
        return bbox

    # Extract width / height from orig_size in a robust way
    width = height = None

    if isinstance(orig_size, (list, tuple)) and len(orig_size) >= 2:
        width, height = orig_size[0], orig_size[1]
    elif isinstance(orig_size, dict):
        width = (
            orig_size.get("width")
            or orig_size.get("w")
            or orig_size.get("page_width")
        )
        height = (
            orig_size.get("height")
            or orig_size.get("h")
            or orig_size.get("page_height")
        )

    if not width or not height:
        return bbox

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return bbox

    try:
        x1, y1, x2, y2 = bbox
        width = float(width)
        height = float(height)
        if width == 0 or height == 0:
            return bbox

        return [
            float(x1) / width,
            float(y1) / height,
            float(x2) / width,
            float(y2) / height,
        ]
    except Exception:
        # If anything blows up (non-numeric, etc.), just keep original
        return bbox



class QdrantVectorStore:
    """
    Consistent Qdrant vector store implementation mirroring qdrant_client.py
    """
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        vector_dim: int = 64,  # Default to 64 to match original
        prefer_grpc: bool = False,
    ) -> None:
        self.url = (url or "").strip()
        self.vector_dim = max(1, int(vector_dim))
        self.enabled = False
        self.client: Optional[QdrantClient] = None
        self._collections: set[str] = set()

        if not self.url:
            LOG.info("Qdrant disabled because QDRANT_URL is not configured.")
            return

        try:
            self.client = QdrantClient(
                url=self.url,
                prefer_grpc=prefer_grpc,
                api_key=api_key or None,
            )
            self.enabled = True
        except Exception as exc:  # pragma: no cover
            LOG.warning("Failed to initialize Qdrant client at %s: %s", self.url, exc)

    def _collection_name(self, tenant_id: str, tender_id: str) -> str:
        slug_tenant = _slug(tenant_id)
        slug_tender = _slug(tender_id)
        return f"tender_{slug_tenant}_{slug_tender}"

    def _ensure_collection(self, collection_name: str) -> None:
        if not self.enabled or not self.client:
            return
        if collection_name in self._collections:
            return
        try:
            self.client.get_collection(collection_name)
            self._collections.add(collection_name)
            return
        except Exception:
            pass

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            self._collections.add(collection_name)
        except Exception as exc:  # pragma: no cover
            LOG.warning("Could not create qdrant collection %s: %s", collection_name, exc)

    def upsert_chunks(
        self,
        tenant_id: str,
        tender_id: str,
        source: str,
        chunks: Iterable[Dict[str, Any]],
        embedding_model: str = DEFAULT_EMBED_MODEL,
    ) -> int:
        """Upsert the provided chunks as vectors into the tenant+tender collection."""
        if not self.enabled or not self.client:
            LOG.warning("⚠️ [QDRANT] Vector store disabled or unavailable")
            return 0

        collection = self._collection_name(tenant_id, tender_id)
        LOG.debug("📚 [QDRANT] Using collection: %s", collection)
        self._ensure_collection(collection)

        sanitized_source = _slug(os.path.basename(source or "")) or "upload"
        
        # Extract texts and prepare payloads
        texts: List[str] = []
        payloads: List[Dict[str, Any]] = []
        
        LOG.debug("🔄 [QDRANT] Preparing chunks for embedding...")
        for idx, chunk in enumerate(chunks):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            texts.append(text)

            file_name = chunk.get("file_name") or "doc"
            doc_id = int(chunk.get("doc_id") or (idx + 1))
            chunk_id = int(chunk.get("chunk_id") or (idx + 1))

            payload: Dict[str, Any] = {
                "tenant_id": tenant_id,
                "tender_id": tender_id,
                "source": sanitized_source,
                "doc_id": doc_id,
                "chunk_type": chunk.get("type"),
                "chunk_index": idx,
                "text": text,
                # "doc_chunk_key": f"{doc_id}_{chunk_id}",  # human-readable composite key
            }
            # Add other chunk fields (including chunk_id, label, etc.), but not text
            payload.update({k: v for k, v in chunk.items() if k != "text"})
            payloads.append(payload)

        if not texts:
            LOG.warning("⚠️ [QDRANT] No valid texts to upsert for %s", source)
            return 0

        # Embed the texts
        LOG.info("[qdrant][upsert] Embedding %d texts with model=%r", len(texts), embedding_model)
        embedder = Embedder(model_name=embedding_model)
        vectors = embedder.embed_texts(texts)

        if not vectors:
            raise RuntimeError("No vectors produced by the embedding backend.")

        dim = len(vectors[0])
        LOG.info("[qdrant][upsert] Inferred vector dimension: %d", dim)

        # Validate + cast vectors
        for i, v in enumerate(vectors):
            if len(v) != dim:
                raise RuntimeError(
                    f"Inconsistent vector length at index {i}: {len(v)} != {dim}"
                )
            vectors[i] = [float(x) for x in v]

        # Ensure collection exists with matching dim
        ensure_collection_exact(collection, dim)

        # Create points with embedded vectors
        points: List[PointStruct] = []
        # point_id: 1..N, across all documents in this batch
        for idx, (vec, pl) in enumerate(zip(vectors, payloads), start=1):
            point_id = idx
            pl["point_id"] = point_id  # store point_id in payload for reference
            points.append(PointStruct(id=point_id, vector=vec, payload=pl))

        try:
            LOG.info("📤 [QDRANT] Upserting %d points to collection: %s", len(points), collection)
            self.client.upsert(collection_name=collection, points=points)
            LOG.info("✅ [QDRANT] Successfully upserted %d vectors to Qdrant", len(points))
        except Exception as exc:  # pragma: no cover
            LOG.error("❌ [QDRANT] Upsert failed for %s: %s", collection, exc)
            return 0

        LOG.debug("Upserted %d vectors into Qdrant collection %s", len(points), collection)
        return len(points)


# -------------------------------------------------------------------
# Chunking helper – coarse_to_fine + labels + doc_id/chunk_id
# -------------------------------------------------------------------

def _chunk_file_with_coarse_to_fine(
    file_path: str,
    tender_id: str,
    file_name: str,
    doc_id: int,
) -> List[Dict[str, Any]]:
    """
    Thin wrapper to call your customized coarse_to_fine pipeline and normalize fields.

    doc_id:
        Integer identifier for this document (starting from 1, incremented per document).
    """
    LOG.info(
        "[chunk] Coarse-to-fine chunking file=%r (tender_id=%r, doc_id=%d)",
        file_name,
        tender_id,
        doc_id,
    )

    raw_chunks = create_chunks_coarsetofine(
        pdf_path=file_path,
        backend="pymupdf",
        verbose=True,
    )

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1  # per-document chunk_id starting at 1

    for ch in raw_chunks:
        text = (ch.get("text") or "").strip()
        if not text:
            continue

        # Generate labels for this chunk
        # try:
        #     label_raw = generate_chunk_label(text, SYSTEM_MSG)
        #     try:
        #         label = json.loads(label_raw)
        #     except Exception:
        #         label = label_raw
        # except Exception as e:
        #     LOG.warning("[chunk] Failed to generate label for chunk (doc_id=%d): %s", doc_id, e)
        #     label = None
        
        label = None  # Skip labeling for now to save time/costs

        # Page / bbox / orig_size – normalize names
        page = ch.get("page") or ch.get("page_number") or 1
        bbox = ch.get("bbox", [])
        orig_size = ch.get("orig_size") or ch.get("page_size")

        bbox_rel = _normalize_bbox_relative(bbox, orig_size)

        chunk_id = chunk_counter
        chunk_counter += 1

        chunks.append(
            {
                "doc_id": doc_id,           # int starting at 1 for first document
                "chunk_id": chunk_id,       # int starting at 1 per document
                "text": text,
                "page": page,
                "bbox": bbox_rel,
                "orig_size": orig_size,
                "tender_id": tender_id,
                "file_name": file_name,
                "label": label,             # multi-label classification from LLM
            }
        )

    LOG.info(
        "[chunk] Produced %d chunks from file=%r (tender_id=%r, doc_id=%d)",
        len(chunks),
        file_name,
        tender_id,
        doc_id,
    )
    return chunks


# -------------------------------------------------------------------
# Public API: process tender and store chunks in Qdrant
# -------------------------------------------------------------------

def process_tender_and_store_in_qdrant(
    tender_id: str,
    tender_dir: str,
    name: str,
    *,
    collection_name: str = DEFAULT_COLLECTION,  # kept for logs; actual collection derived in QdrantVectorStore
    embedding_model: str = DEFAULT_EMBED_MODEL,
    tenant_id: str = "default-tenant",
) -> int:
    """
    Walk a tender directory, run coarse-to-fine chunking on all files,
    embed chunks, and upsert into a Qdrant collection.
    """
    LOG.info(
        "[tender-ingest] Start processing tender_id=%r name=%r dir=%r → collection=%r model=%r",
        tender_id,
        name,
        tender_dir,
        collection_name,
        embedding_model,
    )

    all_chunks: List[Dict[str, Any]] = []

    # 1. Walk directory and chunk each file using coarse_to_fine
    doc_id_counter = 1
    for root, _, files in os.walk(tender_dir):
        files.sort()  # stable ordering for deterministic doc_ids
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_chunks = _chunk_file_with_coarse_to_fine(
                    file_path=file_path,
                    tender_id=tender_id,
                    file_name=file,
                    doc_id=doc_id_counter,
                )
                all_chunks.extend(file_chunks)
                doc_id_counter += 1
            except Exception as e:
                LOG.exception(
                    "[tender-ingest] Failed to chunk file %r for tender %r: %s",
                    file,
                    tender_id,
                    e,
                )

    if not all_chunks:
        LOG.warning(
            "[tender-ingest] No chunks generated for tender_id=%r in dir=%r",
            tender_id,
            tender_dir,
        )
        return 0

    # 2. Use QdrantVectorStore for consistent upsert behavior
    vector_store = QdrantVectorStore(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        vector_dim=4096,  # Default embedding dimension for most models
    )
    
    try:
        upserted = vector_store.upsert_chunks(
            tenant_id=tenant_id,
            tender_id=tender_id,
            source=tender_dir,
            chunks=all_chunks,
            embedding_model=embedding_model,
        )
    except Exception as e:
        LOG.exception(
            "[tender-ingest] Upsert failed for tender_id=%r collection=%r: %s",
            tender_id,
            collection_name,
            e,
        )
        return 0

    # 3. (optional) Cleanup temporary tender directory
    try:
        pass
        # os.system(f"rm -rf {tender_dir}")
    except Exception as e:
        LOG.warning(
            "[tender-ingest] Failed to cleanup tender_dir=%r: %s",
            tender_dir,
            e,
        )

    # 4. Build project card fields so the frontend can render metadata
    try:
        tender = tender_repo.get(tender_id)
        if tender is None:
            LOG.warning("[tender-ingest] Tender %r not found when building project card fields", tender_id)
        else:
            tenant_id = getattr(tender, "tenant_id", "") or tenant_id
            fields = build_project_card_fields(
                tender_id=tender_id,
                tenant_id=tenant_id,
                qdrant_url=QDRANT_URL,
                qdrant_api_key=QDRANT_API_KEY,
            )
            tender_repo.update_project_card_fields(tender_id, fields)
            LOG.info("[tender-ingest] Project card fields populated for %r", tender_id)
    except Exception as e:
        LOG.exception(
            "[tender-ingest] Failed to build project card fields for %r: %s",
            tender_id,
            e,
        )

    LOG.info(
        "[tender-ingest] Successfully processed tender %r: %d chunks stored in Qdrant (collection=%r)",
        tender_id,
        upserted,
        collection_name,
    )

    # 5. Update tender state to INGESTED after successful upsert
    try:
        tender_repo.set_state(tender_id, TenderState.INGESTED)
        LOG.info("[tender-ingest] Updated tender %r state to INGESTED", tender_id)
    except Exception as e:
        LOG.warning(
            "[tender-ingest] Failed to update tender state for %r: %s",
            tender_id,
            e,
        )

    return upserted
