"""
TrustLoop Vision Evidence Service Adapter.

Provides a clean, fault-tolerant interface around existing Vision component:
- vision.image_analyzer.analyze_image

Strict Design Principle:
- MISSING VISION EVIDENCE IS NOT NEGATIVE EVIDENCE.
  no image != damaged item != fraudulent claim != rejection
- If credentials or images are missing, fail gracefully without breaking the API.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


def verify_evidence(
    image_path: Optional[str] = None,
    return_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verify physical return evidence using Gemini Vision when available.

    Args:
        image_path: Optional path to local image evidence.
        return_reason: Customer's stated return reason.

    Returns:
        Structured vision result dictionary conforming to VisionEvidenceResult schema.
    """
    if not image_path or not image_path.strip():
        return {
            "available": False,
            "verified": False,
            "reason": "NO_IMAGE_EVIDENCE",
            "image_quality": None,
            "product_condition": None,
            "damage_detected": None,
            "packaging_condition": None,
            "evidence_consistent": None,
            "confidence": None,
            "explanation": "No image evidence was supplied with this return request.",
            "image_path": None,
        }

    path = Path(image_path)
    if not path.exists():
        return {
            "available": False,
            "verified": False,
            "reason": "IMAGE_NOT_FOUND",
            "image_quality": None,
            "product_condition": None,
            "damage_detected": None,
            "packaging_condition": None,
            "evidence_consistent": None,
            "confidence": None,
            "explanation": f"Specified image file not found: {image_path}",
            "image_path": image_path,
        }


    try:
        from vision.image_analyzer import analyze_image

        raw_analysis = analyze_image(str(path), return_reason=return_reason)

        return {
            "available": True,
            "verified": True,
            "reason": None,
            "image_quality": raw_analysis.get("image_quality"),
            "product_condition": raw_analysis.get("product_condition"),
            "damage_detected": raw_analysis.get("damage_detected"),
            "packaging_condition": raw_analysis.get("packaging_condition"),
            "evidence_consistent": raw_analysis.get("evidence_consistent"),
            "confidence": raw_analysis.get("confidence"),
            "explanation": raw_analysis.get("explanation"),
            "image_path": str(path),
        }

    except RuntimeError as exc:
        # e.g., GEMINI_API_KEY not configured
        logger.warning(f"Vision service credential warning: {exc}")
        return {
            "available": False,
            "verified": False,
            "reason": "VISION_CREDENTIALS_UNAVAILABLE",
            "image_quality": None,
            "product_condition": None,
            "damage_detected": None,
            "packaging_condition": None,
            "evidence_consistent": None,
            "confidence": None,
            "explanation": f"Vision analysis unavailable: {exc}",
            "image_path": str(path),
        }

    except Exception as exc:
        logger.warning(f"Vision analysis error: {exc}")
        return {
            "available": False,
            "verified": False,
            "reason": "VISION_ANALYSIS_FAILED",
            "image_quality": None,
            "product_condition": None,
            "damage_detected": None,
            "packaging_condition": None,
            "evidence_consistent": None,
            "confidence": None,
            "explanation": f"Vision processing error: {exc}",
            "image_path": str(path),
        }
