# answer_pipeline.py
from typing import List, Dict, Any, Optional
import logging
import json
from dataclasses import dataclass

# Reuse existing components from the original file
from tender_analyzer.apps.qa_analysis.demo_cv_search_llm import (
    OllamaProvider,
    QdrantClient,
    Agent,
    COLL_PROJECT_KK,
    INITIAL_TOP_K,
    TOOL_BASE_TOP_K,
    TOP_K_BOOST,
    MAX_TRIES,
    EVAL_PASS_THRESHOLD,
    ToolPlan,
    ToolCallRecord,
    GoalState,
    MemoryRecord,
)

logger = logging.getLogger("answer_pipeline")

@dataclass
class SummaryAnalysisResult:
    question: str
    final_answer: str
    used_tools: bool
    tool_calls: List[Dict[str, Any]]
    initial_rag_chunks: List[Dict[str, Any]]
    best_score: float

def run_summary_analysis(
    question: str,
    ollama_base_url: str = "http://localhost:11434",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    chat_model: str = "qwen3:32b",
    embed_model: str = "qwen3-embedding:8b",
    max_tries: int = MAX_TRIES,
    eval_threshold: float = EVAL_PASS_THRESHOLD,
) -> SummaryAnalysisResult:
    """
    Automatically choose between pure LLM answer or tool-augmented RAG based on question complexity.
    
    Returns:
        SummaryAnalysisResult with answer, metadata, and tool usage info.
    """
    # Initialize providers
    provider = OllamaProvider(
        base_url=ollama_base_url,
        chat_model=chat_model,
        embed_model=embed_model
    )
    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port, prefer_grpc=False)
    agent = Agent(provider, qdrant)
    
    logger.info(f"Starting analysis for question: {question}")
    
    # Step 1: Initial RAG from project_kk
    rag_result = agent.search_tool.search(
        collection=COLL_PROJECT_KK,
        query=question,
        top_k=INITIAL_TOP_K
    )
    initial_docs = rag_result["results"]
    
    # Generate initial answer
    docs_json = json.dumps(initial_docs, ensure_ascii=False)
    initial_prompt = f"""
You are a domain expert answering tender questions.

Question:
{question}

Retrieved context:
{docs_json}

Provide a concise, factual answer. If the context lacks key details, say so explicitly.
Do NOT mention tools or retrieval systems.
"""
    initial_answer = provider.chat([{"role": "user", "content": initial_prompt}], temperature=0.2)
    
    # Step 2: Judge if tools are necessary
    judge_prompt = f"""
Question: {question}
Initial Answer: {initial_answer}

Available tools:
- search_gruner_cv: Expert CVs and experience
- search_gruner_strategy: Company strategy
- search_gruner_pastproject: Past project references

Is the initial answer sufficient? Return ONLY "True" if NO additional tools are needed, else "False".
"""
    judge_system = "You are a strict evaluator. Return ONLY 'True' or 'False'."
    judgment = provider.chat(
        [{"role": "user", "content": judge_prompt}],
        system_prompt=judge_system,
        temperature=0.0
    ).strip().lower()
    
    used_tools = (judgment != "true")
    final_answer = initial_answer
    all_tool_calls = []
    best_score = 0.0
    
    if used_tools:
        logger.info("Tools deemed necessary. Executing full pipeline...")
        # Create minimal goal state
        gs = GoalState(
            goal_id="auto-" + question[:20].replace(" ", "_"),
            question=question,
            created_at=0.0
        )
        
        # Plan tools
        plan = agent.plan_tools(question, initial_answer, initial_docs)
        gs.plan_goal = plan.goal
        
        # Execute tool calls across retries
        all_rag_answers = []
        all_tool_data = []
        
        for attempt in range(max_tries):
            tool_data, rag_answers = agent.execute_tool_calls(plan, attempt_index=attempt)
            all_tool_data.extend(tool_data)
            all_rag_answers.extend(rag_answers)
            
            # Build extended answer
            extended = initial_answer + "\n\n" + "\n\n".join(rag_answers) if rag_answers else initial_answer
            
            # Evaluate (optional but recommended)
            # TODO: Uncomment if evaluation is desired
            # report = agent.evaluate(question, extended, initial_docs + tool_data, plan)
            # if report.score >= eval_threshold:
            #     final_answer = extended
            #     best_score = report.score
            #     break
            
            final_answer = extended  # Use latest result
        
        # Final refinement if tools were used
        if all_rag_answers:
            final_answer = agent.refine_final_answer(question, final_answer)
        
        # Capture tool call records
        all_tool_calls = [record.model_dump() for record in agent.tool_call_records]
    else:
        logger.info("Initial answer sufficient. Skipping tools.")
    
    return SummaryAnalysisResult(
        question=question,
        final_answer=final_answer,
        used_tools=used_tools,
        tool_calls=all_tool_calls,
        initial_rag_chunks=[
            {
                "rank": i + 1,
                "id": item.get("id", ""),
                "score": float(item.get("score", 0.0)),
                "text": item.get("text", "")
            }
            for i, item in enumerate(initial_docs)
        ],
        best_score=best_score
    )

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    question = "Does Gruner have at least 10 years of experience in sustainable infrastructure?"
    result = run_summary_analysis(question)
    print(f"Answer:\n{result.final_answer}\n")
    print(f"Used tools: {result.used_tools}")