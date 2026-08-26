"""
TrustLoop RAG Policy Service Adapter.

Provides a clean, fault-tolerant interface around existing RAG components:
- rag.policy_agent.evaluate_policy
- rag.retriever.retrieve_policy

Strict Principle:
- "No policy evidence" != "Policy violation".
- If index is unavailable, gracefully return neutral compliance status without crashing.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def analyze_policy(
    case_data: Dict[str, Any],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate policy compliance and retrieve relevant policy rules.

    Args:
        case_data: Dictionary of return case attributes.
        top_k: Number of policy chunks to retrieve.

    Returns:
        Structured policy result dictionary conforming to RAGPolicyResult schema.
    """
    try:
        from rag.policy_agent import evaluate_policy

        raw_result = evaluate_policy(case_data, top_k=top_k)

        return {
            "available": True,
            "policy_status": raw_result.get("policy_status", "POLICY_COMPLIANT"),
            "flags": raw_result.get("flags", []),
            "retrieved_policy": raw_result.get("retrieved_policy", []),
            "query": raw_result.get("query", ""),
            "reason": None,
        }

    except FileNotFoundError as exc:
        logger.warning(f"RAG policy index not found: {exc}")
        return {
            "available": False,
            "policy_status": "POLICY_COMPLIANT",
            "flags": [],
            "retrieved_policy": [],
            "query": "",
            "reason": f"Policy index unavailable: {exc}",
        }

    except Exception as exc:
        logger.warning(f"RAG policy evaluation error: {exc}")
        return {
            "available": False,
            "policy_status": "POLICY_COMPLIANT",
            "flags": [],
            "retrieved_policy": [],
            "query": "",
            "reason": f"Policy service error: {exc}",
        }
