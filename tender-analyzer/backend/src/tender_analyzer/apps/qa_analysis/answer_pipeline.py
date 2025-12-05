from pathlib import Path

#!/usr/bin/env python3
"""
Loop over prebid_questions, process each through agent.run_once(), 
save answers and metadata to JSONL with timing logs.
"""

import json
import logging
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Dict, List, Any

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

# 临时文件存储目录
TEMP_DIR = Path(__file__).parent.parent / "storage" / "tender_summary"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# Import agent components from existing module
from tender_analyzer.apps.qa_analysis.demo_cv_search_llm import (
    OllamaProvider,
    QdrantClient,
    Agent,
)

# Import questions directly from source module
from tender_analyzer.apps.qa_analysis.prebid_questions_1113 import QUESTIONS

# ==========================
# Configuration
# ==========================



# Agent configuration (matching File B defaults)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))



# ==========================
# Helper Functions
# ==========================

def iterate_questions(qmap: Dict[str, Dict[str, List[str]]]):
    """Generator to iterate through nested question structure."""
    for cat, subcats in qmap.items():
        for subcat, questions in subcats.items():
            for q in questions:
                yield cat, subcat, q

# ==========================
# Main Processing
# ==========================

def run_summary_analysis(tender_id: str = "", tenant_id: str = "default-tenant") -> Dict[str, Any]:
    # File paths
    OUTPUT_JSONL = TEMP_DIR / f"{tenant_id}_{tender_id}_qanswers.jsonl"
    LOG_FILE = TEMP_DIR / f"{tenant_id}_{tender_id}_pipeline_run_time.log"
    COLLECTION_NAME = build_tender_collection_name(tenant_id, tender_id)

    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("answer_pipeline")

    """
    Process all questions using agent.run_once() and save results to JSONL.
    """
    logger.info("🚀 Starting Question Processing Pipeline")
    logger.info(f"Output file: {OUTPUT_JSONL}")
    logger.info(f"Log file: {LOG_FILE}")
    
    # Initialize agent
    logger.info("Initializing Agent...")
    try:
        provider = OllamaProvider(base_url=OLLAMA_BASE_URL)
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=False)
        agent = Agent(provider, qdrant)
        logger.info("✅ Agent initialized successfully")
    except Exception as e:
        logger.exception(f"❌ Failed to initialize agent: {e}")
        sys.exit(1)
    
    query_logs = []
    start_global = time.perf_counter()
    total_questions = 0
    successful_questions = 0
    
    # Process questions and write to JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as outf:
        for cat, subcat, q in iterate_questions(QUESTIONS):
            total_questions += 1
            logger.info(f"[{total_questions}] Processing: {cat} > {subcat}")
            logger.debug(f"Full question: {q}")
            
            t0 = time.perf_counter()
            
            try:
                # Process question through agent
                answer_result = agent.run_once(q, COLLECTION_NAME)
                references_raw: List[Any] = []
                if isinstance(answer_result, dict):
                    answer_text = str(answer_result.get("answer", ""))
                    references_raw = answer_result.get("references") or []
                else:
                    answer_text = getattr(answer_result, "answer", None)
                    if answer_text is None:
                        answer_text = str(answer_result)
                    references_raw = getattr(answer_result, "references", []) or []

                # references_raw is a list of AnswerReference / dicts
                filtered_raw = []
                for ref in references_raw:
                    try:
                        ref_tid = None
                        if hasattr(ref, "tender_id"):
                            ref_tid = getattr(ref, "tender_id", None)
                        elif isinstance(ref, dict):
                            ref_tid = ref.get("tender_id")
                        # keep if tender_id matches or is missing (older data)
                        if ref_tid is None or str(ref_tid) == str(tender_id):
                            filtered_raw.append(ref)
                    except Exception:
                        # on any weird object, drop it rather than contaminate
                        continue
                    
                references_serialized: List[Dict[str, Any]] = []
                for ref in filtered_raw:
                    if ref is None:
                        continue
                    if hasattr(ref, "as_dict"):
                        references_serialized.append(ref.as_dict())  # type: ignore[attr-defined]
                    elif is_dataclass(ref):
                        references_serialized.append(asdict(ref))
                    elif isinstance(ref, dict):
                        references_serialized.append(ref)
                    else:
                        # Best-effort fallback
                        references_serialized.append({"value": str(ref)})

                elapsed = time.perf_counter() - t0
                successful_questions += 1
                
                # Create output record
                rec = {
                    "category": cat,
                    "subcategory": subcat,
                    "question": q,
                    "answer": answer_text,
                    "references": references_serialized,
                    "processing_time_sec": round(elapsed, 3),
                    "status": "success"
                }
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
                # Log timing
                query_logs.append({
                    "question": q,
                    "category": cat,
                    "subcategory": subcat,
                    "elapsed_sec": round(elapsed, 3),
                    "status": "success"
                })
                
                logger.info(f"✅ Completed in {elapsed:.2f}s")
                
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.exception(f"❌ Error processing question (after {elapsed:.2f}s)")
                
                # Write error record
                rec = {
                    "category": cat,
                    "subcategory": subcat,
                    "question": q,
                    "answer": f"ERROR: {str(e)}",
                    "processing_time_sec": round(elapsed, 3),
                    "status": "error",
                    "error_message": str(e)
                }
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
                query_logs.append({
                    "question": q,
                    "category": cat,
                    "subcategory": subcat,
                    "elapsed_sec": round(elapsed, 3),
                    "status": "error",
                    "error": str(e)
                })
    
    # Save summary timing log
    total_time = time.perf_counter() - start_global
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total_questions,
        "successful_questions": successful_questions,
        "failed_questions": total_questions - successful_questions,
        "total_sec": round(total_time, 3),
        "avg_sec_per_question": round(total_time / total_questions, 3) if total_questions > 0 else 0,
        "queries": query_logs
    }
    
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        logger.info(f"✅ Timing log saved to: {LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to save timing log: {e}")
    
    # Final summary
    logger.info("=" * 50)
    logger.info("📊 PIPELINE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total questions: {total_questions}")
    logger.info(f"Successful: {successful_questions}")
    logger.info(f"Failed: {total_questions - successful_questions}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average time: {total_time/total_questions:.2f}s/question")
    logger.info(f"Answers saved to: {os.path.abspath(OUTPUT_JSONL)}")
    logger.info("=" * 50)
    
    return {
        "output_file": os.path.abspath(OUTPUT_JSONL),
        "log_file": LOG_FILE,
        "total_questions": total_questions,
        "successful_questions": successful_questions,
        "total_time": total_time
    }

# ==========================
# Entry Point
# ==========================
# from tender_analyzer.apps.qa_analysis.prebid_questions_1113 import QUESTIONS


if __name__ == "__main__":
    print("🚀 Starting Question Processing Pipeline")
    print("=" * 60)
    
    try:
        results = run_summary_analysis()
        print("\n✨ Pipeline finished successfully!")
        print(f"📄 Output: {results['output_file']}")
        print(f"📊 Processed {results['successful_questions']}/{results['total_questions']} questions")
        print(f"⏱️  Total time: {results['total_time']:.2f}s")
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        logger.exception("Unhandled exception in main")
        sys.exit(1)

