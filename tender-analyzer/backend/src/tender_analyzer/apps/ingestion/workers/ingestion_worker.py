# backend/src/tender_analyzer/apps/ingestion/workers/ingestion_worker.py

import os
import uuid
import logging
from typing import Any, Dict, List, Tuple, Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from tender_analyzer.apps.ingestion.chunking.coarse_to_fine import create_chunks_coarsetofine
from tender_analyzer.apps.ingestion.embedding.embedder import Embedder
from tender_analyzer.domain.repositories import tender_repo
from tender_analyzer.common.state.enums import TenderState

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


def upsert_to_qdrant(
    collection: str,
    texts: List[str],
    payloads: List[Dict[str, Any]],
    embedding_model: str,
) -> int:
    """
    Upsert helper mirroring main.py:

    - embeds texts with Embedder instance
    - enforces consistent vector dimension
    - creates collection if missing (with cosine distance)
    - uses random UUIDs as point IDs
    """
    if not texts:
        LOG.warning("[qdrant][upsert] No texts provided; skipping.")
        return 0

    LOG.info("[qdrant][upsert] Embedding %d texts with model=%r", len(texts), embedding_model)
    
    # Create an embedder instance with the specified model
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

    qc = get_qdrant()
    points: List[PointStruct] = []
    for vec, pl in zip(vectors, payloads):
        pid = str(pl.get("chunk_id") or uuid.uuid4())
        points.append(PointStruct(id=pid, vector=vec, payload=pl))

    LOG.info(
        "[qdrant][upsert] Sending %d points → collection %r (Qdrant at %s)",
        len(points),
        collection,
        QDRANT_URL,
    )
    qc.upsert(collection_name=collection, points=points)
    LOG.info("[qdrant][upsert] Done.")
    return len(points)


def _slug(value: str) -> str:
    """Normalize string for safe use as collection name or identifier."""
    if not value:
        return "unknown"
    sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    return sanitized[:32] or "unknown"


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
            self.client = QdrantClient(url=self.url, prefer_grpc=prefer_grpc, api_key=api_key or None)
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
            LOG.warning(f"⚠️ [QDRANT] Vector store disabled or unavailable")
            return 0

        collection = self._collection_name(tenant_id, tender_id)
        LOG.debug(f"📚 [QDRANT] Using collection: {collection}")
        self._ensure_collection(collection)

        sanitized_source = _slug(os.path.basename(source or "")) or "upload"
        
        # Extract texts and prepare payloads
        texts: List[str] = []
        payloads: List[Dict[str, Any]] = []
        
        LOG.debug(f"🔄 [QDRANT] Preparing chunks for embedding...")
        for idx, chunk in enumerate(chunks):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            texts.append(text)
            
            payload = {
                "tenant_id": tenant_id,
                "tender_id": tender_id,
                "source": sanitized_source,
                "chunk_type": chunk.get("type"),
                "chunk_index": idx,
                "snippet": text[:512],
            }
            payload.update({k: v for k, v in chunk.items() if k != "text"})  # Add other chunk fields
            payloads.append(payload)

        if not texts:
            LOG.warning(f"⚠️ [QDRANT] No valid texts to upsert for {source}")
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
        for idx, (vec, pl) in enumerate(zip(vectors, payloads)):
            point_id = str(uuid.uuid4())
            points.append(PointStruct(id=str(point_id), vector=vec, payload=pl))

        try:
            LOG.info(f"📤 [QDRANT] Upserting {len(points)} points to collection: {collection}")
            self.client.upsert(collection_name=collection, points=points)
            LOG.info(f"✅ [QDRANT] Successfully upserted {len(points)} vectors to Qdrant")
        except Exception as exc:  # pragma: no cover
            LOG.error(f"❌ [QDRANT] Upsert failed for {collection}: {exc}")
            return 0

        LOG.debug("Upserted %d vectors into Qdrant collection %s", len(points), collection)
        return len(points)


# -------------------------------------------------------------------
# Chunking helper – force use of your coarse_to_fine strategy
# -------------------------------------------------------------------

def _chunk_file_with_coarse_to_fine(file_path: str, tender_id: str, file_name: str) -> List[Dict[str, Any]]:
    """
    Thin wrapper to call your customized coarse_to_fine pipeline and normalize fields
    to something compatible with main.py's payload expectations.
    """
    LOG.info("[chunk] Coarse-to-fine chunking file=%r (tender_id=%r)", file_name, tender_id)

    # Keep your existing call signature; adjust if your function changes.
    raw_chunks = create_chunks_coarsetofine(
        pdf_path=file_path,
        backend="pymupdf",
        verbose=True,
    )

    chunks: List[Dict[str, Any]] = []
    for ch in raw_chunks:
        text = (ch.get("text") or "").strip()
        if not text:
            continue

        # Page / bbox / orig_size – we normalize names to match main.py-style payloads
        page = ch.get("page") or ch.get("page_number") or 1
        bbox = ch.get("bbox", [])
        orig_size = ch.get("orig_size") or ch.get("page_size")

        chunk_id = ch.get("id") or str(uuid.uuid4())
        chunk_id = f"{tender_id}_{chunk_id}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "page": page,
                "bbox": bbox,
                "orig_size": orig_size,
                "tender_id": tender_id,
                "file_name": file_name,
            }
        )

    LOG.info(
        "[chunk] Produced %d chunks from file=%r (tender_id=%r)",
        len(chunks),
        file_name,
        tender_id,
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
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    tenant_id: str = "default_tenant",  # Added tenant_id parameter for consistency
) -> int:
    """
    Walk a tender directory, run coarse-to-fine chunking on all files,
    embed chunks, and upsert into a Qdrant collection.

    This mirrors main.py's ingest flow but is tailored for the backend
    tender ingestion worker.

    Args:
        tender_id: logical id for this tender
        tender_dir: directory where tender files live
        name: human-readable name (currently just logged)
        collection_name: Qdrant collection (default: 'tender_chunks')
        embedding_model: embedding model name for Ollama (or your embedder)
        tenant_id: tenant identifier for multi-tenancy (default: 'default_tenant')
    Returns:
        Number of chunks successfully upserted.
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
    for root, _, files in os.walk(tender_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_chunks = _chunk_file_with_coarse_to_fine(
                    file_path=file_path,
                    tender_id=tender_id,
                    file_name=file,
                )
                all_chunks.extend(file_chunks)
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
        # still try to clean up tender_dir
        # os.system(f"rm -rf {tender_dir}")
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
        # cleanup and signal failure
        # os.system(f"rm -rf {tender_dir}")
        return 0

    # 3. Cleanup temporary tender directory
    try:
        pass
        # os.system(f"rm -rf {tender_dir}")
    except Exception as e:
        LOG.warning(
            "[tender-ingest] Failed to cleanup tender_dir=%r: %s",
            tender_dir,
            e,
        )

    # 4. Update tender state to INGESTED after successful upsert
    try:
        tender_repo.set_state(tender_id, TenderState.INGESTED)
        LOG.info(
            "[tender-ingest] Updated tender %r state to INGESTED",
            tender_id,
        )
    except Exception as e:
        LOG.warning(
            "[tender-ingest] Failed to update tender state for %r: %s",
            tender_id,
            e,
        )

    LOG.info(
        "[tender-ingest] Successfully processed tender %r: %d chunks stored in Qdrant (collection=%r)",
        tender_id,
        upserted,
        collection_name,
    )
    return upserted

    LOG.info(
        "[tender-ingest] Successfully processed tender %r: %d chunks stored in Qdrant (collection=%r)",
        tender_id,
        upserted,
        collection_name,
    )
    return upserted