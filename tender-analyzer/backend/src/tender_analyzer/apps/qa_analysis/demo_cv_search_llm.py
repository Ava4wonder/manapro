#!/usr/bin/env python3
# demo_cv_search_llm.py (or agent_qa.py)
# Deps: pip install pydantic qdrant-client requests

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple, Union

import requests
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# ==========================
# logging setup
# ==========================

LOG_FILE = "3_process.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("agent")

# ==========================
# CONFIG
# ==========================

# OLLAMA_CHAT_MODEL_QWEN = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:30b") 
OLLAMA_CHAT_MODEL_QWEN = os.getenv("OLLAMA_CHAT_MODEL", "gpt-oss:20b")      # e.g. gpt-oss / qwen3
OLLAMA_CHAT_MODEL_QWEN_2 = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:32b") 
OLLAMA_CHAT_MODEL_GPT = os.getenv("OLLAMA_CHAT_MODEL", "gpt-oss:20b")      # e.g. gpt-oss / qwen3
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:8b")  # e.g. qwen3-embedding
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Collections
COLL_PROJECT_KK = "nunagreen_noheading"
COLL_GRUNER_CV = "cv_keyqualifications_qwen"
COLL_GRUNER_STRATEGY = "brochure_qwen"
COLL_GRUNER_PASTPROJECT = "brochure_qwen"

# RAG + tools search parameters
INITIAL_TOP_K = 12
TOOL_BASE_TOP_K = 10       # Base top_k for tool searches
TOP_K_BOOST = 2           # How much to increase K per retry
MAX_TRIES = 0
EVAL_PASS_THRESHOLD = 0.70


# Tool call logging configuration
TOOL_CALL_LOG_FILE = "tool_calls_log.json"

# Predefined questions
from tender_analyzer.apps.qa_analysis.prebid_questions_1113 import QUESTIONS
PREDEFINED_QUESTIONS = []

for key_cat, subcats in QUESTIONS.items():
    for subcat, qs in subcats.items():
        PREDEFINED_QUESTIONS.extend(qs)
print(f"Loaded {len(PREDEFINED_QUESTIONS)} predefined questions.")

# Regex used to parse inline reference tags such as "[ref_id:123]"
REF_ID_PATTERN = re.compile(r"\[ref_id:(\d+)\]")
# REF_ID_PATTERN = re.compile("[\[\u3010\u300a]ref_id:(\d+)[\]\u3011\u300b]")
REF_ID_PATTERN = re.compile(r"[\[【《(]ref_id:(\d+)[\]】》)]")


# ==========================
# Schemas / Models
# ==========================

class ToolPlanCall(BaseModel):
    tool_name: Literal["search_gruner_cv", "search_gruner_strategy", "search_gruner_pastproject"]
    queries: List[str]
    top_k: int = TOOL_BASE_TOP_K  # Use base top_k from config


class ToolPlan(BaseModel):
    goal: str
    should_call_tools: bool
    calls: List[ToolPlanCall] = Field(default_factory=list)


class EvalReport(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasons: str
    should_refine: bool


class MemoryRecord(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "assistant"
    content: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class GoalState(BaseModel):
    goal_id: str
    question: str
    created_at: float
    plan_goal: Optional[str] = None
    history: List[MemoryRecord] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """Record of a tool call request and response"""
    call_id: str
    timestamp: float
    tool_name: str
    collection: str
    query: str
    top_k_used: int
    request_meta: Dict[str, Any] = Field(default_factory=dict)
    response: Dict[str, Any] = Field(default_factory=dict)  # Contains retrieved chunks
    rag_answer: str = ""  # RAG answer generated from chunks


@dataclass
class AnswerReference:
    chunk_id: Optional[str]
    file_name: str
    page: Optional[int]
    bbox: List[float]
    snippet: str
    score: Optional[float] = None
    source_collection: Optional[str] = None
    source_tool: Optional[str] = None
    orig_size: Optional[List[float]] = None
    tender_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_name": self.file_name,
            "page": self.page,
            "bbox": self.bbox,
            "snippet": self.snippet,
            "score": self.score,
            "source_collection": self.source_collection,
            "source_tool": self.source_tool,
            "orig_size": self.orig_size,
            "tender_id": self.tender_id,
        }


@dataclass
class AnswerResult:
    answer: str
    references: List[AnswerReference]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "references": [ref.as_dict() for ref in self.references],
        }


# ==========================
# Ollama provider (chat + embeddings)
# ==========================

class OllamaProvider:
    def __init__(self,
                 base_url: str = OLLAMA_BASE_URL,
                 chat_model_gpt: str = OLLAMA_CHAT_MODEL_GPT,
                 chat_model_qwen: str = OLLAMA_CHAT_MODEL_QWEN,
                 chat_model_qwen_2: str = OLLAMA_CHAT_MODEL_QWEN_2, 
                 embed_model: str = OLLAMA_EMBED_MODEL):
        self.base = base_url.rstrip("/")
        self.chat_model_gpt = chat_model_gpt
        self.chat_model_qwen = chat_model_qwen
        self.chat_model_qwen_2 = chat_model_qwen_2
        self.embed_model = embed_model

    def chat_gpt(self,
             messages: List[Dict[str, str]],
             temperature: float = 0.05,
             system_prompt: Optional[str] = None) -> str:
        payload = {
            "model": self.chat_model_gpt,
            "messages": (
                [{"role": "system", "content": system_prompt}] if system_prompt else []
            ) + messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = requests.post(f"{self.base}/api/chat", json=payload, timeout=12000)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")
    
    def chat_qwen(self,
             messages: List[Dict[str, str]],
             temperature: float = 0.05,
             system_prompt: Optional[str] = None) -> str:
        payload = {
            "model": self.chat_model_qwen,
            "messages": (
                [{"role": "system", "content": system_prompt}] if system_prompt else []
            ) + messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = requests.post(f"{self.base}/api/chat", json=payload, timeout=12000)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")
    
    def chat_qwen_2(self,
             messages: List[Dict[str, str]],
             temperature: float = 0.05,
             system_prompt: Optional[str] = None) -> str:
        payload = {
            "model": self.chat_model_qwen_2,
            "messages": (
                [{"role": "system", "content": system_prompt}] if system_prompt else []
            ) + messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = requests.post(f"{self.base}/api/chat", json=payload, timeout=12000)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")

    def embed(self, text: str) -> List[float]:
        """Get embedding from Ollama using the embedding model"""
        payload = {
            "model": self.embed_model,
            "prompt": text,
        }
        url = f"{self.base}/api/embeddings"
        logger.info(f"[EMBED] Requesting embedding: model={self.embed_model}, url={url}")
        r = requests.post(url, json=payload, timeout=1200)
        r.raise_for_status()
        data = r.json()

        # Handle possible response shapes
        vec: Optional[List[float]] = None

        if isinstance(data, dict):
            if "embedding" in data and isinstance(data["embedding"], list):
                vec = data["embedding"]
            elif "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
                vec = data["embeddings"][0]

        if vec is None:
            logger.error(f"[EMBED] Unexpected embedding response format: {data}")
            raise ValueError(f"Unexpected embedding response from Ollama: {data}")

        if not isinstance(vec, list) or len(vec) == 0:
            logger.error(f"[EMBED] Empty or invalid embedding returned: url:{url},payload:{payload}, data:{data}, {vec}")
            raise ValueError("Empty embedding returned from Ollama; aborting search.")

        logger.info(f"[EMBED] Got embedding: dim={len(vec)}")
        return vec


# ==========================
# Qdrant search helper
# ==========================

class QdrantSearchTool:
    def __init__(self, client: QdrantClient, embedder: OllamaProvider):
        self.client = client
        self.embedder = embedder

    def get_chunks_by_point_ids(
            self,
            collection_name: str,
            point_ids: list[int],
        ) -> Dict[int, Dict[str, Any]]:
        """
        Batch retrieve chunks by point_ids.
        Returns a dict: point_id -> normalized payload dict.
        """
        if not point_ids:
            return {}

        points = self.client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )

        result: Dict[int, Dict[str, Any]] = {}
        for p in points:
            pid = int(p.id)
            payload = p.payload or {}
            result[pid] = {
                "doc_id": payload.get("doc_id"),
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text"),
                "page": payload.get("page"),
                "bbox": payload.get("bbox"),
                "orig_size": payload.get("orig_size"),
                "tender_id": payload.get("tender_id"),
                "file_name": payload.get("file_name"),
                "label": payload.get("label"),
                "source_collection": collection_name,
            }
        return result

    def search_and_stitch(
        self,
        collection: str,
        query: str,
        top_k: int = 5,
        must_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search Qdrant and stitch results into a single text blob."""
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
        stitched_texts = []
        for h in hits:
            payload = getattr(h, "payload", {}) or {}
            text = payload.get("text") 
            ref_id = str(getattr(h, "id", ""))
            if text:
                enriched = f"<c> {text} [ref_id:{ref_id}] </c>"
                stitched_texts.append(enriched)
        # print(f"[stitch] {stitched_texts}")
        return stitched_texts

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
                    "text": payload.get("text")
                    or payload.get("content")
                    or "",
                    "meta": payload,
                }
            )
        return {"collection": collection, "top_k": top_k, "results": results}


# ==========================
# Agent
# ==========================

class Agent:
    def __init__(self, provider: OllamaProvider, qdrant: QdrantClient):
        self.llm = provider
        self.qdrant = qdrant
        self.search_tool = QdrantSearchTool(qdrant, provider)
        self.memory_file = "data_memory.json"
        self.tool_call_records: List[ToolCallRecord] = []  # Track tool calls

        # Map tool -> collection name
        self.tool_to_collection: Dict[str, str] = {
            "search_gruner_cv": COLL_GRUNER_CV,
            "search_gruner_strategy": COLL_GRUNER_STRATEGY,
            "search_gruner_pastproject": COLL_GRUNER_PASTPROJECT,
        }

    def _save_tool_call_records(self):
        """Persist tool call records to JSON file"""
        try:
            records = [record.model_dump() for record in self.tool_call_records]
            with open(TOOL_CALL_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            logger.info(f"[TOOL LOG] Saved {len(records)} tool call records to {TOOL_CALL_LOG_FILE}")
        except Exception as e:
            logger.error(f"[TOOL LOG] Failed to save tool call records: {e}")

    def _extract_reference_from_chunk(self, chunk: Dict[str, Any]) -> Optional[AnswerReference]:
        meta = chunk.get("meta") or {}
        chunk_id = meta.get("chunk_id") or chunk.get("id")
        file_name = meta.get("file_name") or meta.get("source_file_name") or meta.get("source")
        bbox = meta.get("bbox") or meta.get("bounding_box") or []
        page = meta.get("page") or meta.get("page_number")
        snippet = chunk.get("text") or meta.get("snippet") or meta.get("text") or ""
        if not file_name or not bbox or len(bbox) < 4:
            return None

        bbox_values: List[float] = []
        try:
            bbox_values = [float(v) for v in bbox[:4]]
        except Exception:
            return None

        orig_size = meta.get("orig_size") or meta.get("page_size")
        if isinstance(orig_size, dict):
            width = orig_size.get("width") or orig_size.get("w")
            height = orig_size.get("height") or orig_size.get("h")
            if width and height:
                orig_size = [float(width), float(height)]
            else:
                orig_size = None
        elif isinstance(orig_size, (list, tuple)) and len(orig_size) >= 2:
            orig_size = [float(orig_size[0]), float(orig_size[1])]
        else:
            orig_size = None

        return AnswerReference(
            chunk_id=chunk_id,
            file_name=str(file_name),
            page=int(page) if page is not None else None,
            bbox=bbox_values,
            snippet=snippet,
            score=chunk.get("score"),
            source_collection=chunk.get("source_collection"),
            source_tool=chunk.get("source_tool"),
            orig_size=orig_size,
            tender_id=meta.get("tender_id"),
        )
    
    def _format_newref_answer(self, raw_answer: str, collection_name: str) -> Tuple[str, List[AnswerReference]]:
        """
        Format the initial answer by:
        1. Extracting all [ref_id:XXX] references from the answer
        2. Retrieving chunk details from Qdrant using point IDs
        3. Converting inline references to [file_name, p:page] format
        4. Returning formatted answer and reference list
        """
        # Extract all ref_ids from the answer
        ref_ids = REF_ID_PATTERN.findall(raw_answer)
        ref_ids_int = []
        try:
            ref_ids_int = [int(rid) for rid in ref_ids]
        except ValueError as e:
            logger.warning(f"[FORMAT] Failed to parse ref_ids as integers: {e}")
            return raw_answer, []

        if not ref_ids_int:
            logger.info("[FORMAT] No references found in answer")
            return raw_answer, []

        # Retrieve chunk details from Qdrant
        try:
            chunks_by_id = self.search_tool.get_chunks_by_point_ids(
                collection_name=collection_name,
                point_ids=ref_ids_int
            )
        except Exception as e:
            logger.error(f"[FORMAT] Failed to retrieve chunks from Qdrant: {e}")
            return raw_answer, []

        # Build reference list and mapping for reformatting
        references: List[AnswerReference] = []
        ref_id_to_display = {}  # Maps ref_id to display text

        for ref_id in ref_ids_int:
            chunk_info = chunks_by_id.get(ref_id)
            if not chunk_info:
                logger.warning(f"[FORMAT] Chunk {ref_id} not found in Qdrant response")
                continue

            file_name = chunk_info.get("file_name", "Unknown")
            page = chunk_info.get("page")
            text = chunk_info.get("text", "")
            bbox = chunk_info.get("bbox", [])
            orig_size = chunk_info.get("orig_size")
            tender_id = chunk_info.get("tender_id")
            chunk_id = chunk_info.get("chunk_id")

            # Create display text
            page_text = f"p:{page}" if page is not None else "p:?"
            display_text = f"[{file_name}, {page_text}](chunkref://{chunk_id})"
            ref_id_to_display[ref_id] = display_text

            # Create AnswerReference
            bbox_values = []
            if bbox and len(bbox) >= 4:
                try:
                    bbox_values = [float(v) for v in bbox[:4]]
                except (ValueError, TypeError):
                    pass

            ref = AnswerReference(
                chunk_id=chunk_id,
                file_name=file_name,
                page=int(page) if page is not None else None,
                bbox=bbox_values,
                snippet=text,
                score=None,
                source_collection=collection_name,
                source_tool=None,
                orig_size=orig_size,
                tender_id=tender_id,
            )
            references.append(ref)

        # Reformat answer: replace [ref_id:XXX] with [file_name, p:page]
        formatted_answer = raw_answer
        for ref_id in ref_ids_int:
            if ref_id in ref_id_to_display:
                old_pattern = f"[ref_id:{ref_id}]"
                new_text = ref_id_to_display[ref_id]
                formatted_answer = formatted_answer.replace(old_pattern, new_text)

        logger.info(f"[FORMAT] Extracted {len(references)} references from answer")
        return formatted_answer, references

    

    
    def _collect_references(self, docs: List[Dict[str, Any]], limit: int = 5) -> List[AnswerReference]:
        """Pick top-N unique references from retrieved docs."""
        references: List[AnswerReference] = []
        seen: set[tuple] = set()
        sorted_docs = sorted(docs, key=lambda item: float(item.get("score") or 0.0), reverse=True)
        for doc in sorted_docs:
            ref = self._extract_reference_from_chunk(doc)
            if not ref:
                continue
            key = (ref.chunk_id, ref.file_name, ref.page, tuple(ref.bbox))
            if key in seen:
                continue
            references.append(ref)
            seen.add(key)
            if len(references) >= limit:
                break
        return references

    def _generate_rag_from_chunks(self, query: str, plan_goal: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate RAG answer from retrieved chunks using Ollama"""
        if not chunks:
            return "No relevant information found for this query."

        chunks_json = json.dumps(chunks, ensure_ascii=False)
        prompt = f"""
        You are a domain expert analyzing criteria matches (e.g., experience, qualifications).
        
        CRITICAL RULE FOR CRITERIA: "Meeting the criteria" means meeting OR EXCEEDING the required standard. 
        For example:
        - If the requirement is "10+ years of experience", candidates with 10, 15, or 20 years ALL meet the criteria.
        - "At least 5 projects" includes 5, 6, or more projects.
        Never treat requirements as "exact matches only" unless explicitly stated (e.g., "exactly 10 years").

        Query: {query}
        Goal: {plan_goal}
        
        Context from retrieved documents:
        {chunks_json}
        
        Provide a concise answer that:
        1. Identifies which candidates/entries meet the criteria (using the rule above)
        2. Clearly explains how they meet or exceed the requirements
        3. Only uses information from the provided context
        """

        return self.llm.chat_qwen(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

    def _single_json_coercer(self, raw_planner_output: str) -> str:
        """Force LLM output to valid ToolPlan JSON"""
        system_prompt = """
        You are a JSON schema enforcer. Your ONLY job is to convert the input text into valid JSON
        that strictly matches this schema:
        
        {
            "goal": "string (one-sentence purpose of the tool calls)",
            "should_call_tools": "boolean (true/false)",
            "calls": [
                {
                    "tool_name": "string (must be one of: search_gruner_cv, search_gruner_strategy, search_gruner_pastproject)",
                    "queries": ["string (1-3 specific search queries)"],
                    "top_k": "integer (number of results to return)"
                }
            ]
        }
        
        Rules:
        1. If input contains JSON, fix its formatting to match the schema
        2. If input is prose/description, convert it into JSON matching the schema
        3. NEVER add explanations, comments, or text outside the JSON
        4. Ensure "tool_name" uses only the allowed values
        5. If no tool calls are needed, set "should_call_tools": false and empty "calls"
        
        Return ONLY the valid JSON object.
        """
        
        return self.llm.chat_qwen(
            messages=[{"role": "user", "content": f"Convert this to valid ToolPlan JSON: {raw_planner_output}"}],
            system_prompt=system_prompt,
            temperature=0.0
        )
    
    def _json_coercer(self, raw_planner_output: str) -> str:
        """
        Force LLM to convert any output into valid ToolPlan JSON with retry logic.
        """
        system_prompt = """
        You are a JSON schema enforcer. Your ONLY job is to convert the input text into valid JSON
        that strictly matches this schema:
        
        {
            "goal": "string (one-sentence purpose of the tool calls)",
            "should_call_tools": "boolean (true/false)",
            "calls": [
                {
                    "tool_name": "string (must be one of: search_gruner_cv, search_gruner_strategy, search_gruner_pastproject)",
                    "queries": ["string (1-3 specific search queries)"],
                    "top_k": "integer (number of results to return)"
                }
            ]
        }
        
        Rules:
        1. If input contains JSON, fix its formatting to match the schema
        2. If input is prose/description, convert it into JSON matching the schema
        3. NEVER add explanations, comments, or text outside the JSON
        4. Ensure "tool_name" uses only the allowed values
        5. If no tool calls are needed, set "should_call_tools": false and empty "calls"
        
        Return ONLY the valid JSON object.
        """
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                return self.llm.chat_qwen(
                    messages=[{"role": "user", "content": f"Convert this to valid ToolPlan JSON: {raw_planner_output}"}],
                    system_prompt=system_prompt,
                    temperature=0.0
                )
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"[COERCER] HTTP error on attempt {attempt + 1}/{max_retries}: {str(e)}. "
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            except Exception as e:
                logger.warning(
                    f"[COERCER] Unexpected error on attempt {attempt + 1}/{max_retries}: {str(e)}. "
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
        
        # Fallback: return valid default JSON if all retries fail
        logger.error(f"[COERCER] All {max_retries} retries failed. Using fallback JSON.")
        return '''{
            "goal": "Provide the best possible answer from available information.",
            "should_call_tools": false,
            "calls": []
        }'''

    def _tool_necessity_judge(self, question: str, initial_answer: str) -> bool:
        """Independently judge if tools are needed"""
        system_prompt = """
        You are a tool necessity evaluator. Analyze if additional tool calls are required
        to answer the question thoroughly, based on the initial answer provided.
        
        Tools available:
        - search_gruner_cv: Expert CVs and experience
        - search_gruner_strategy: Company strategy documents
        - search_gruner_pastproject: Similar past projects/references
        
        Decision criteria:
        - Return True if the initial answer lacks specific details that the tools could provide
        - Return True if critical information is missing that tools might supply
        - Return False only if the initial answer is comprehensive enough without tools
        
        Return ONLY "True" or "False" (boolean, no explanations).
        """
        
        judgment = self.llm.chat_qwen(
            messages=[{
                "role": "user", 
                "content": f"Question: {question}\nInitial Answer: {initial_answer}\nNeed tools? (True/False)"
            }],
            system_prompt=system_prompt,
            temperature=0.0
        ).strip().lower()
        
        return judgment == "true"

    def plan_tools(
        self, question: str, initial_answer: str, rag_docs: List[Dict[str, Any]]
    ) -> ToolPlan:
        # docs_json = json.dumps(rag_docs[:5], ensure_ascii=False)

        system_prompt = """
        You are an analyst agent. Your job:
        1) Read the question and the initial RAG-based answer.
        2) Decide whether additional evidence is needed from 3 tools:
           - search_gruner_cv: search Gruner's expert CVs and their experience.
           - search_gruner_strategy: search Gruner's strategy documents.
           - search_gruner_pastproject: search Gruner's similar past projects and references.
        4) Define the ultimate goal using bullet points: which Gruner's internal information is needed to answer the question.
        5) If tools are needed, choose which tool(s) and form sub-queries per tool targeting specificed information from the bullet points.


        Return STRICT JSON only matching the ToolPlan schema.
        """

        user_prompt = f"""
        Question: {question}
        Initial RAG answer: {initial_answer}
        Decide if you need more evidence and which tools to use.
        """

        # Get raw planner output
        raw_planner_output = self.llm.chat_qwen_2(
            [{"role": "user", "content": user_prompt}],
            temperature=0.1,
            system_prompt=system_prompt,
        )
        logger.info(f"[PLAN] Raw output: {raw_planner_output[:4000]}...")

        # Apply JSON coercer guard
        coerced_json = self._json_coercer(raw_planner_output)
        logger.info(f"[COERCER] Coerced JSON: {coerced_json[:400]}...")

        # Parse with fallback
        try:
            plan = ToolPlan(** json.loads(coerced_json))
        except Exception as e:
            logger.warning(f"[PLAN] Failed to parse coerced JSON: {e}")
            plan = ToolPlan(
                goal="Provide the best possible answer from available information.",
                should_call_tools=False,
                calls=[],
            )

        # Apply tool necessity judge guard
        judge_needs_tools = self._tool_necessity_judge(question, initial_answer)
        logger.info(f"[JUDGE] Independent tool necessity: {judge_needs_tools}")

        # Force re-plan if judge disagrees with planner
        if judge_needs_tools and not plan.should_call_tools:
            logger.info("[GUARD] Judge disagrees - forcing re-plan")
            user_prompt += "\nIMPORTANT: The question requires tool usage. Provide tool calls."
            raw_replan = self.llm.chat_qwen(
                [{"role": "user", "content": user_prompt}],
                temperature=0.1,
                system_prompt=system_prompt,
            )
            coerced_replan = self._json_coercer(raw_replan)
            try:
                plan = ToolPlan(**json.loads(coerced_replan))
            except Exception as e:
                logger.warning(f"[PLAN] Failed to parse re-plan: {e}")

        logger.info(f"[PLAN] Final plan: {plan.model_dump_json(indent=2)[:400]}...")
        return plan

    # ------------------------
    # Persistence
    # ------------------------

    def _load_persisted(self) -> Dict[str, Any]:
        if not os.path.exists(self.memory_file):
            return {"goals": {}}
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"goals": {}}

    def _save_persisted(self, store: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.memory_file) or ".", exist_ok=True)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)

    def create_goal(self, question: str) -> GoalState:
        gs = GoalState(
            goal_id=str(uuid.uuid4()),
            question=question,
            created_at=time.time(),
        )
        store = self._load_persisted()
        store["goals"][gs.goal_id] = gs.model_dump()
        self._save_persisted(store)
        return gs

    def update_goal(self, gs: GoalState):
        store = self._load_persisted()
        store["goals"][gs.goal_id] = gs.model_dump()
        self._save_persisted(store)

    # ------------------------
    # Step 1: initial RAG on project_kk
    # ------------------------

    def newref_initial_rag(
        self,
        question: str,
        collection_name: str,
        top_k: int = INITIAL_TOP_K,
    ) -> Dict[str, Any]:
        logger.info(f"[RAG] Searching collection={collection_name}, top_k={top_k}")

        stitched_context = self.search_tool.search_and_stitch(
            collection=collection_name,
            query=question,
            top_k=top_k,
        )
        
        SYSTEM_MSG = (
            "You are a professional and precise Pre-Bid Q&A assistant, very good at reading and reasoning in a surgical way. "
            "You will receive a QUESTION and a CONTEXT. The CONTEXT is a stitched text composed of multiple chunks, "
            "each wrapped in the following format: `<c> ... [ref_id:CHUNK_ID] </c>`. "
            "Each `ref_id` uniquely identifies the source chunk of evidence.\n\n"

            "(Mandatory and required) Answer format:\n"
            "1) The First line outputs a single word: either 'Mentioned' or 'No mention', indicating whether the answer "
            "can be derived from the provided CONTEXT only. If only part of the QUESTION can be answered from the CONTEXT, "
            "still output 'Mentioned' and answer all aspects that can be supported, while clearly explaining any limitations.\n"
            "2) Then provide a concise but detailed explanation.\n\n"
            "3) When explaining, you MUST attach each factual statement with evidence from the CONTEXT:\n"
            "   - At the end of each explanation paragraph or sentence, append one or more evidence references in the format `[ref_id:CHUNK_ID]`.\n"
            "   - Use the `ref_id` that appears inside the `<c> ... [ref_id:CHUNK_ID] </c>` wrapper in the CONTEXT.\n"
            "   - If a sentence is supported by multiple chunks, you may list multiple references, for example: '[ref_id:1];[ref_id:4]'.\n"
            "   - Do NOT paste or quote the original evidence text itself, only the reference(s) `[ref_id:CHUNK_ID]`.\n"
            "   - Never invent or fabricate `ref_id`s that do not appear in the CONTEXT.\n\n"

            "You must rely ONLY on the given CONTEXT for your answer:\n"
            "   - Do not introduce external knowledge or assumptions.\n"
            "   - If the CONTEXT is insufficient to fully answer the QUESTION, clearly state what is missing while still following the rules above.\n"
        )

        user_prompt = (
            f"CONTEXT:\n{stitched_context}\n\n"
            f"QUESTION:\n{question}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_prompt},
        ]
        REF_SIG = False
        num_ref_tries = 4
        while not REF_SIG and num_ref_tries > 0:
            answer = self.llm.chat_gpt(messages, temperature=0.05)
            formatted_answer, references = self._format_newref_answer(answer, collection_name)
            if len(references) > 0:
                REF_SIG = True
            else:
                num_ref_tries -= 1
                logger.info(f"[RAG] REF_SIG not found, retrying... {num_ref_tries} tries left")
        logger.info(f"[RAG] Initial answer (first 400 chars): {formatted_answer}...")
        return {
            "answer": formatted_answer,
            # "raw_answer": answer,
            # "docs": results,
            "references": references,
            # "top_k": top_k,
        }

    def initial_rag(self, question: str, collection_name: str, top_k: int = INITIAL_TOP_K) -> Dict[str, Any]:
        logger.info(f"[RAG] Searching collection={collection_name}, top_k={top_k}")
        rag_result = self.search_tool.search(
            collection=collection_name, query=question, top_k=top_k
        )
        for item in rag_result.get("results", []):
            item["source_collection"] = collection_name

        # Save topK chunk texts to retrieval_temp.json
        # try:
        #     out_payload = {
        #         "question": question,
        #         "collection": collection_name,
        #         "top_k": top_k,
        #         "chunks": [
        #             {
        #                 "rank": i + 1,
        #                 "id": str(item.get("id", "")),
        #                 "score": float(item.get("score", 0.0)),
        #                 "text": item.get("text", ""),
        #             }
        #             for i, item in enumerate(rag_result.get("results", []))
        #         ],
        #     }
        #     with open("retrieval_temp.json", "w", encoding="utf-8") as f:
        #         json.dump(out_payload, f, ensure_ascii=False, indent=2)
        #     logger.info(
        #         f"[RAG] Saved {len(out_payload['chunks'])} chunks to retrieval_temp.json"
        #     )
        # except Exception as e:
        #     logger.exception(f"[RAG] Failed to write retrieval_temp.json: {e}")

        docs_json = json.dumps(rag_result["results"], ensure_ascii=False)

        prompt = f"""
            You are a domain expert answering questions for a tender project.

            Question:
            {question}

            You are given retrieved passages from an internal project collection (project_kk):

            {docs_json}

            Write an initial answer ONLY based on the given passages. Be explicit when something is not specified.
            Do NOT mention tools or Qdrant. Keep it structured and concise.
            """

        
        answer = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.05)
        logger.info(f"[RAG] Initial answer: {answer}...")
        return {"answer": answer, "docs": rag_result["results"], "top_k": top_k}

    # ------------------------
    # Step 4: execute tools and gather results
    # ------------------------

    def execute_tool_calls(
        self,
        plan: ToolPlan,
        attempt_index: int,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Execute tool calls with:
        - Each subquery gets maximum 1 trial
        - Adjustable top_k based on attempt
        - RAG processing of results
        Returns (all_tool_data, rag_answers)
        """
        all_tool_data: List[Dict[str, Any]] = []
        rag_answers: List[str] = []
        self.tool_call_records = []  # Reset records for this execution

        if not plan.should_call_tools or not plan.calls:
            logger.info("[TOOLS] No tools to call according to plan.")
            return all_tool_data, rag_answers

        for call in plan.calls:
            coll = self.tool_to_collection.get(call.tool_name)
            if not coll:
                logger.warning(f"[TOOLS] Unknown tool_name={call.tool_name}; skipping.")
                continue

            # Calculate effective top_k with boost for retries
            effective_top_k = call.top_k + attempt_index * TOP_K_BOOST
            logger.info(
                f"[TOOLS] Executing tool={call.tool_name} on collection={coll}, "
                f"effective_top_k={effective_top_k}, queries={call.queries}"
            )

            # Process each query once (max 1 trial per subquery)
            for q in call.queries:
                call_id = str(uuid.uuid4())
                try:
                    # Execute search
                    result = self.search_tool.search(
                        collection=coll,
                        query=q,
                        top_k=effective_top_k,
                    )

                    # Generate RAG answer from chunks using Ollama
                    rag_answer = self._generate_rag_from_chunks(
                        query=q,
                        plan_goal=plan.goal,
                        chunks=result["results"]
                    )
                    rag_answers.append(rag_answer)
                    logger.info(f"[TOOLS] {len(result['results'])} chunks, RAG answer for query '{q}' (first 300 chars): {rag_answer[:300]}...")

                    # Prepare tool data for further processing
                    for r in result["results"]:
                        enriched_r = dict(r)
                        enriched_r["source_tool"] = call.tool_name
                        enriched_r["source_collection"] = coll
                        enriched_r["source_query"] = q
                        enriched_r["rag_answer"] = rag_answer
                        all_tool_data.append(enriched_r)

                    # Record the tool call
                    tool_record = ToolCallRecord(
                        call_id=call_id,
                        timestamp=time.time(),
                        tool_name=call.tool_name,
                        collection=coll,
                        query=q,
                        top_k_used=effective_top_k,
                        response=result,
                        rag_answer=rag_answer
                    )
                    self.tool_call_records.append(tool_record)

                except Exception as e:
                    logger.error(f"[TOOLS] Error executing query '{q}': {e}")
                    continue

        # Save all tool call records after execution
        self._save_tool_call_records()

        logger.info(f"[TOOLS] Total tool documents processed: {len(all_tool_data)}")
        logger.info(f"[TOOLS] Generated {len(rag_answers)} RAG answers from tool results")
        return all_tool_data, rag_answers

    # ------------------------
    # Final answer refinement
    # ------------------------

    def refine_final_answer(self, question: str, extended_answer: str) -> str:
        """Final refinement of the extended answer to ensure coherence"""
        system_prompt = """
        You are a senior editor specializing in technical documentation.
        Your task is to refine and improve the coherence of an extended answer while preserving all key information.

        "(Mandatory and required) Answer format:\n"
        "1) The First line outputs a single word: either 'Mentioned' or 'No mention', indicating whether the answer "
        "2) keep all references intact and properly placed at the same locations as in the original answer (as input)."
        
        Other Requirements:
        1. Ensure the answer flows logically and addresses the question directly
        2. Remove redundancies while keeping all important details
        3. Maintain technical accuracy
        4. Structure the answer for readability, but do not remove the first line indicating 'Mentioned' or 'No mention' of the original answer; and keep all references
        5. Do not introduce new information not present in the extended answer
        """

        prompt = f"""
        Question: {question}
        
        Extended Answer to refine:
        {extended_answer}
        
        Please provide a coherent final answer that addresses the question thoroughly.
        """

        return self.llm.chat_qwen_2(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=0.2
        )

    # ------------------------
    # MAIN per-question pipeline
    # ------------------------

    def run_once(self, question: str, collection_name: str) -> AnswerResult:
        logger.info(f"=== Start new question ===\nQ: {question}")
        gs = self.create_goal(question)

        # 1) Initial RAG from project_kk
        rag = self.newref_initial_rag(question, collection_name, top_k=INITIAL_TOP_K)
        initial_answer = rag.get("answer", "")
        print(f">>>> Initial Answer:\n{initial_answer}\n")
        rag_docs = rag.get("docs", []) or []
        inline_references = rag.get("references", [])
        extended_answer = initial_answer  # Start with initial answer

        # 2-3) Plan tools
        # plan = self.plan_tools(question, initial_answer, rag_docs)
        # gs.plan_goal = plan.goal
        # self.update_goal(gs)

        # Initialize tracking variables
        final_answer = initial_answer
        # all_docs = list(rag_docs)
        best_score = 0.0
        all_rag_answers = []

        # 4–5) Try tools + composite answer up to MAX_TRIES with increasing K
        for attempt in range(MAX_TRIES):
            logger.info(f"--- Tool attempt {attempt+1}/{MAX_TRIES} ---")
            tool_data, rag_answers = self.execute_tool_calls(plan, attempt_index=attempt)
            all_rag_answers.extend(rag_answers)

            # if tool_data:
            #     all_docs = list(rag_docs) + tool_data

            # Append RAG answers to initial answer
            if rag_answers:
                extended_answer = initial_answer + "\n\n" + "\n\n".join(rag_answers)

            # Evaluate current answer
            current_answer = extended_answer if rag_answers else initial_answer
            # report = self.evaluate(question, current_answer, all_docs, plan)
            
            # if report.score > best_score:
            #     best_score = report.score
            #     final_answer = current_answer
            final_answer = current_answer  # Always take latest answer


            # if report.score >= EVAL_PASS_THRESHOLD and not report.should_refine:
            #     logger.info("[PIPELINE] Goal satisfied; stopping retries.")
            #     break
            # else:
            #     logger.info(
            #         "[PIPELINE] Goal not fully satisfied; will try with higher top_k if attempts remain."
            #     )

        # 6) Final answer refinement
        if all_rag_answers:  # Only refine if we have tool-based answers
            final_answer = self.refine_final_answer(question, extended_answer)
            logger.info(f"[REFINEMENT] Generated final refined answer (first 400 chars): {final_answer[:400]}...")

        # # Update memory with final answer
        # gs.history.append(
        #     MemoryRecord(
        #         role="assistant",
        #         content=final_answer,
        #         meta={"best_score": best_score, "tool_calls_count": len(self.tool_call_records)},
        #     )
        # )
        # self.update_goal(gs)

        references = inline_references 
        logger.info(
            f"=== End of question === best_score={best_score:.2f}, answer_len={len(final_answer)}\n"
        )
        return AnswerResult(answer=final_answer, references=references)

    # ------------------------
    # Evaluation method
    # ------------------------

    def evaluate(
        self,
        question: str,
        answer: str,
        all_docs: List[Dict[str, Any]],
        plan: ToolPlan,
    ) -> EvalReport:
        docs_json = json.dumps(all_docs[:30], ensure_ascii=False)
        system_prompt = """
        You are a strict evaluator.

        Given:
        - a tender-related question,
        - an answer, and
        - the supporting evidence,

        You must:
        1) Score how well the answer satisfies the goal and uses the evidence (0..1).
        2) Briefly explain reasons.
        3) Indicate whether refinement is needed.

        Return STRICT JSON:
        {
          "score": 0.0-1.0,
          "reasons": "...",
          "should_refine": true/false
        }
        """

        user_prompt = f"""
        Question:
        {question}

        Goal:
        {plan.goal}

        Answer:
        {answer}

        Evidence (subset JSON):
        {docs_json}
        """
        raw = self.llm.chat_qwen(
            [{"role": "user", "content": user_prompt}],
            temperature=0.0,
            system_prompt=system_prompt,
        )
        logger.info(f"[EVAL] Raw eval output: {raw[:400]}...")
        try:
            coerced_reraw = self._json_coercer(raw)
            js = json.loads(coerced_reraw)
            report = EvalReport(**js)
        except Exception as e:
            logger.warning(f"[EVAL] Failed to parse eval JSON: {e}")
            report = EvalReport(
                score=0.5, reasons="Fallback evaluator", should_refine=False
            )

        logger.info(
            f"[EVAL] score={report.score:.2f}, refine={report.should_refine}, reasons={report.reasons}"
        )
        return report


# ==========================
# main
# ==========================
COLL_PROJECT_KK = "tender_default-tenant_e7e04df2-ca13-4312-b91b-0727d49f"
def main():
    provider = OllamaProvider()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=False)
    agent = Agent(provider, qdrant)
    collection = "project_kk"

    logger.info("Agent initialized. Starting predefined QA run.")
    print("== Predefined QA Run ==")

    for q in PREDEFINED_QUESTIONS:
        print(f"\nQ: {q}")
        try:
            ans = agent.run_once(q, COLL_PROJECT_KK)
            if isinstance(ans, AnswerResult):
                print(f"\nA:\n{ans.answer}\n")
                if ans.references:
                    print("References:")
                    for ref in ans.references:
                        print(f" - {ref.file_name} (page {ref.page})")
            else:
                print(f"\nA:\n{ans}\n")
        except Exception as e:
            logger.exception(f"Error during question '{q}': {e}")


if __name__ == "__main__":
    main()
