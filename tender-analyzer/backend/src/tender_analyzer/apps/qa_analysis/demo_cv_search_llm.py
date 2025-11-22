#!/usr/bin/env python3
# demo_cv_search_llm.py (or agent_qa.py)
# Deps: pip install pydantic qdrant-client requests

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

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

OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:32b")      # e.g. gpt-oss / qwen3
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
MAX_TRIES = 3
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


# ==========================
# Ollama provider (chat + embeddings)
# ==========================

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
        r = requests.post(url, json=payload, timeout=60)
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

        return self.llm.chat(
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
        
        return self.llm.chat(
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
                return self.llm.chat(
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
        
        judgment = self.llm.chat(
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
        docs_json = json.dumps(rag_docs[:5], ensure_ascii=False)

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
        raw_planner_output = self.llm.chat(
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
            raw_replan = self.llm.chat(
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

    def initial_rag(self, question: str, top_k: int = INITIAL_TOP_K) -> Dict[str, Any]:
        logger.info(f"[RAG] Searching collection={COLL_PROJECT_KK}, top_k={top_k}")
        rag_result = self.search_tool.search(
            collection=COLL_PROJECT_KK, query=question, top_k=top_k
        )

        # Save topK chunk texts to retrieval_temp.json
        try:
            out_payload = {
                "question": question,
                "collection": COLL_PROJECT_KK,
                "top_k": top_k,
                "chunks": [
                    {
                        "rank": i + 1,
                        "id": str(item.get("id", "")),
                        "score": float(item.get("score", 0.0)),
                        "text": item.get("text", ""),
                    }
                    for i, item in enumerate(rag_result.get("results", []))
                ],
            }
            with open("retrieval_temp.json", "w", encoding="utf-8") as f:
                json.dump(out_payload, f, ensure_ascii=False, indent=2)
            logger.info(
                f"[RAG] Saved {len(out_payload['chunks'])} chunks to retrieval_temp.json"
            )
        except Exception as e:
            logger.exception(f"[RAG] Failed to write retrieval_temp.json: {e}")

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
        
        answer = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.2)
        logger.info(f"[RAG] Initial answer (first 400 chars): {answer[:400]}...")
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
                    logger.info(f"[TOOLS] {len(result['results'])} chunks, RAG answer for query '{q}' (first 300 chars): {rag_answer[:3000]}...")

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
        
        Requirements:
        1. Ensure the answer flows logically and addresses the question directly
        2. Remove redundancies while keeping all important details
        3. Maintain technical accuracy
        4. Structure the answer for readability
        5. Do not introduce new information not present in the extended answer
        """

        prompt = f"""
        Question: {question}
        
        Extended Answer to refine:
        {extended_answer}
        
        Please provide a polished, coherent final answer that addresses the question thoroughly.
        """

        return self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=0.2
        )

    # ------------------------
    # MAIN per-question pipeline
    # ------------------------

    def run_once(self, question: str) -> str:
        logger.info(f"=== Start new question ===\nQ: {question}")
        gs = self.create_goal(question)

        # 1) Initial RAG from project_kk
        rag = self.initial_rag(question, top_k=INITIAL_TOP_K)
        initial_answer = rag["answer"]
        rag_docs = rag["docs"]
        extended_answer = initial_answer  # Start with initial answer

        # 2–3) Plan tools
        plan = self.plan_tools(question, initial_answer, rag_docs)
        gs.plan_goal = plan.goal
        self.update_goal(gs)

        # Initialize tracking variables
        final_answer = initial_answer
        all_docs = list(rag_docs)
        best_score = 0.0
        all_rag_answers = []

        # 4–5) Try tools + composite answer up to MAX_TRIES with increasing K
        for attempt in range(MAX_TRIES):
            logger.info(f"--- Tool attempt {attempt+1}/{MAX_TRIES} ---")
            tool_data, rag_answers = self.execute_tool_calls(plan, attempt_index=attempt)
            all_rag_answers.extend(rag_answers)

            # Append RAG answers to initial answer
            if rag_answers:
                extended_answer = initial_answer + "\n\n" + "\n\n".join(rag_answers)
                all_docs = list(rag_docs) + tool_data

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

        logger.info(
            f"=== End of question === best_score={best_score:.2f}, answer_len={len(final_answer)}\n"
        )
        return final_answer

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
        raw = self.llm.chat(
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

def main():
    provider = OllamaProvider()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=False)
    agent = Agent(provider, qdrant)

    logger.info("Agent initialized. Starting predefined QA run.")
    print("== Predefined QA Run ==")

    for q in PREDEFINED_QUESTIONS:
        print(f"\nQ: {q}")
        try:
            ans = agent.run_once(q)
            print(f"\nA:\n{ans}\n")
        except Exception as e:
            logger.exception(f"Error during question '{q}': {e}")


if __name__ == "__main__":
    main()