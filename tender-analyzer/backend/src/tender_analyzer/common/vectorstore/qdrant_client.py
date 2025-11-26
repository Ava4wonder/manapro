from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

LOG = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    from qdrant_client.http.models import PointStruct
    QDRANT_AVAILABLE = True
except ImportError:  # pragma: no cover
    QdrantClient = None  # type: ignore[assignment]
    rest = None  # type: ignore[assignment]
    PointStruct = None  # type: ignore[assignment]
    QDRANT_AVAILABLE = False

DEFAULT_VECTOR_DIM = 64


def _slug(value: str) -> str:
    if not value:
        return "unknown"
    sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    return sanitized[:32] or "unknown"


def build_tender_collection_name(tenant_id: str, tender_id: str) -> str:
    """Public helper for computing the canonical tender collection name."""
    slug_tenant = _slug(tenant_id)
    slug_tender = _slug(tender_id)
    return f"tender_{slug_tenant}_{slug_tender}"


def text_to_vector(text: str, dimension: int = DEFAULT_VECTOR_DIM) -> List[float]:
    """Deterministic fallback embedding that hashes the text into a float vector."""
    if not text:
        return [0.0] * max(1, dimension)
    dimension = max(1, int(dimension))
    digest = hashlib.sha512(text.encode("utf-8", errors="ignore")).digest()
    values = [byte / 255.0 for byte in digest]
    if len(values) < dimension:
        values.extend([0.0] * (dimension - len(values)))
    return values[:dimension]


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        vector_dim: int = DEFAULT_VECTOR_DIM,
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

        if not QDRANT_AVAILABLE:
            LOG.warning("qdrant-client is missing; vector store functionality disabled.")
            return

        try:
            self.client = QdrantClient(url=self.url, prefer_grpc=prefer_grpc, api_key=api_key or None)
            self.enabled = True
        except Exception as exc:  # pragma: no cover
            LOG.warning("Failed to initialize Qdrant client at %s: %s", self.url, exc)

    def _collection_name(self, tenant_id: str, tender_id: str) -> str:
        return build_tender_collection_name(tenant_id, tender_id)

    def _ensure_collection(self, collection_name: str) -> None:
        if not self.enabled or not self.client or not rest:
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
                vectors_config=rest.VectorParams(size=self.vector_dim, distance=rest.Distance.COSINE),
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
    ) -> int:
        """Upsert the provided chunks as vectors into the tenant+tender collection."""
        if not self.enabled or not self.client or not PointStruct:
            LOG.warning(f"⚠️ [QDRANT] Vector store disabled or unavailable")
            return 0

        collection = self._collection_name(tenant_id, tender_id)
        LOG.debug(f"📚 [QDRANT] Using collection: {collection}")
        self._ensure_collection(collection)

        sanitized_source = _slug(os.path.basename(source or "")) or "upload"
        points: list = []
        
        LOG.debug(f"🔄 [QDRANT] Converting chunks to vector points...")
        for idx, chunk in enumerate(chunks):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            vector = text_to_vector(text, self.vector_dim)
            payload = {
                "tenant_id": tenant_id,
                "tender_id": tender_id,
                "source": sanitized_source,
                "chunk_type": chunk.get("type"),
                "chunk_index": idx,
                "snippet": text,
            }
            point_id = f"{tenant_id}-{tender_id}-{sanitized_source}-{idx}"
            points.append(PointStruct(id=str(point_id), vector=vector, payload=payload))

        if not points:
            LOG.warning(f"⚠️ [QDRANT] No valid points to upsert for {source}")
            return 0

        try:
            LOG.info(f"📤 [QDRANT] Upserting {len(points)} points to collection: {collection}")
            self.client.upsert(collection_name=collection, points=points)
            LOG.info(f"✅ [QDRANT] Successfully upserted {len(points)} vectors to Qdrant")
        except Exception as exc:  # pragma: no cover
            LOG.error(f"❌ [QDRANT] Upsert failed for {collection}: {exc}")
            return 0

        LOG.debug("Upserted %d vectors into Qdrant collection %s", len(points), collection)
        return len(points)



__all__ = ["QdrantVectorStore", "text_to_vector", "build_tender_collection_name"]
