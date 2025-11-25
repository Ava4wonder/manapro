import json
import logging
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "qwen3:32b"
OLLAMA_EMBED_MODEL = "qwen3-embedding:8b"
COLL_PROJECT_KK = "nunagreen_noheading"

# --- Field Definitions ---
HIGHLIGHT_QUESTIONS_BY_FIELD = {
    "projectType": "What is the scope of the project (Prospect, EOI, Offer, Contract, or Addendum)?",
    "projectScope": "What is the scope of the project (e.g., new construction, rehabilitation, study, supervision)?",
    "location": "Where is the project located (continent, jurisdiction entity:The overarching legal entity under whose laws the country (and thus the project location) primarily falls, the country it belongs to, region name, city name)?",
    "deadline": "the date of the submission deadline?",
    "submission_format": "What is the required submission format (physical, electronic, or both)?",
    "budgetRange": "Is there an indication of the budget range, and does it match our expectations?",
    "evaluationMethod": "How will the proposals be evaluated (e.g., lowest price, quality-based, mixed)?",
    "weighting": "Is there a weighting system for price vs. quality?",
}

# --- Reused Components ---
class OllamaProvider:
    def __init__(self,
                 base_url: str = OLLAMA_BASE_URL,
                 chat_model: str = OLLAMA_CHAT_MODEL,
                 embed_model: str = OLLAMA_EMBED_MODEL):
        self.base = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model

    def chat(self,
             messages: List[Dict[str, str]],
             temperature: float = 0.2,
             system_prompt: Optional[str] = None) -> str:
        import requests
        payload = {
            "model": self.chat_model,
            "messages": (
                [{"role": "system", "content": system_prompt}] if system_prompt else []
            ) + messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = requests.post(f"{self.base}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()

    def embed(self, text: str) -> List[float]:
        import requests
        payload = {"model": self.embed_model, "prompt": text}
        r = requests.post(f"{self.base}/api/embeddings", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        vec = data.get("embedding") or (data.get("embeddings") and data["embeddings"][0])
        if not isinstance(vec, list) or not vec:
            raise ValueError("Invalid embedding response")
        return vec


class QdrantSearchTool:
    def __init__(self, client: QdrantClient, embedder: OllamaProvider):
        self.client = client
        self.embedder = embedder

    def search(
        self,
        collection: str,
        query: str,
        top_k: int = 5,
        must_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        vec = self.embedder.embed(query)
        qfilter = None
        if must_tags:
            qfilter = Filter(
                must=[
                    FieldCondition(key="tags", match=MatchValue(value=tag))
                    for tag in must_tags
                ]
            )

        hits = self.client.search(
            collection_name=collection,
            query_vector=vec,
            query_filter=qfilter,
            limit=top_k,
        )
        results = []
        for h in hits:
            payload = getattr(h, "payload", {}) or {}
            results.append(
                {
                    "id": str(getattr(h, "id", "")),
                    "score": float(getattr(h, "score", 0.0)),
                    "text": payload.get("text") or payload.get("content") or "",
                    "meta": payload,
                }
            )
        return {"collection": collection, "top_k": top_k, "results": results}


# --- Core Extraction Logic ---
def extract_field_value(
    provider: OllamaProvider,
    context: str,
    question: str,
    field_name: str
) -> str:
    system_prompt = (
        "You are an expert in extracting precise information from project documents. "
        "Answer ONLY with the requested value and short key word ONLY without format styles. If the information is not present, respond exactly with 'N/A'."
    )
    # print('field_nanem', field_name, '<<<', context)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        answer = provider.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            temperature=0.05  # Very low for consistency
        )
        logging.info(f"Extracted '{field_name}': {answer}")
        return answer
    except Exception as e:
        logging.error(f"Error extracting '{field_name}': {e}")
        return "ERROR"


def extract_fields_from_qdrant(
    output_path: str = "extracted_project_fields.json",
    top_k: int = 8,
    must_tags: Optional[List[str]] = None,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
) -> Dict[str, Any]:
    """
    For each field in HIGHLIGHT_QUESTIONS_BY_FIELD:
      - Use the field_description/question as the Qdrant search query
      - Build context from the top chunks returned for that query
      - Extract the field value from that context with Ollama
    """
    logging.info("Initializing Qdrant client and embedder...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    embedder = OllamaProvider()
    search_tool = QdrantSearchTool(client=client, embedder=embedder)

    provider = OllamaProvider()
    result: Dict[str, Any] = {}

    for field_name, question in HIGHLIGHT_QUESTIONS_BY_FIELD.items():
        logging.info(
            f"Searching Qdrant collection '{COLL_PROJECT_KK}' for field '{field_name}' "
            f"using description as query: {question!r}"
        )
        chunks_result = search_tool.search(
            collection=COLL_PROJECT_KK,
            query=question,      # <-- field_description used as the query
            top_k=top_k,
            must_tags=must_tags,
        )

        # Build context specific to this field
        context_chunks = [
            r["text"] for r in chunks_result["results"] if r["text"].strip()
        ]
        if not context_chunks:
            logging.warning(
                f"No relevant context retrieved from Qdrant for field '{field_name}'. "
                "Setting value to 'N/A'."
            )
            result[field_name] = "N/A"
            continue

        full_context = "\n\n".join(context_chunks)
        logging.info(
            f"[{field_name}] Retrieved {len(context_chunks)} chunks. "
            f"Total context length: {len(full_context)} chars."
        )

        # Extract value for this single field using its own context + question
        result[field_name] = extract_field_value(
            provider=provider,
            context=full_context,
            question=question,
            field_name=field_name,
        )

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logging.info(f"✅ Extraction complete. Results saved to {output_path}")
    return result


# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        json_result = extract_fields_from_qdrant(
            output_path="project_metadata.json",
            top_k=5,
            # must_tags=["rfp", "infrastructure"]  # optional filter
        )
        # Now you can access: json_result["deadline"], etc.
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        raise SystemExit(1)
