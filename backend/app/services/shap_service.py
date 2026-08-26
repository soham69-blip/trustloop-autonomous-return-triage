"""
TrustLoop Real-Time TreeSHAP Explainability Service Adapter.

Provides fast, real-time feature attribution using TreeSHAP on active LightGBM models.
Caches TreeExplainer instances per model object for sub-millisecond execution.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Explainer cache: id(model) -> shap.TreeExplainer
_EXPLAINER_CACHE: Dict[int, Any] = {}


def _get_tree_explainer(model: Any):
    """Retrieve or construct a cached TreeExplainer for the model instance."""
    import shap

    model_id = id(model)
    if model_id not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[model_id] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[model_id]


def explain_prediction(
    model: Any,
    X_df: pd.DataFrame,
    predicted_class_idx: int = 0,
    class_labels: Optional[Dict[int, str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compute local TreeSHAP feature attributions for a single case DataFrame.

    Args:
        model: Loaded LightGBM model instance.
        X_df: Single-row pandas DataFrame aligned with model features.
        predicted_class_idx: Target class index to explain (default: model's predicted class).
        class_labels: Optional mapping from class index to human-readable label.
        top_k: Number of positive and negative drivers to return.

    Returns:
        Structured explanation conforming to SHAPExplanationResult schema.
    """
    try:
        explainer = _get_tree_explainer(model)
        raw_shap = explainer.shap_values(X_df)

        feature_names = list(X_df.columns)
        first_row = X_df.iloc[0]

        # Handle different SHAP output shapes:
        # For multi-class LightGBM, raw_shap is typically ndarray of shape (1, n_features, n_classes)
        # or a list of length n_classes where each element is shape (1, n_features).
        if isinstance(raw_shap, list):
            class_idx = min(max(0, predicted_class_idx), len(raw_shap) - 1)
            class_attributions = raw_shap[class_idx][0]
        elif isinstance(raw_shap, np.ndarray):
            if raw_shap.ndim == 3:
                # Shape: (1, n_features, n_classes)
                class_idx = min(max(0, predicted_class_idx), raw_shap.shape[2] - 1)
                class_attributions = raw_shap[0, :, class_idx]
            elif raw_shap.ndim == 2:
                # Binary / single-output shape: (1, n_features)
                class_attributions = raw_shap[0]
            else:
                raise ValueError(f"Unexpected SHAP values shape: {raw_shap.shape}")

        else:
            raise ValueError(f"Unexpected SHAP return type: {type(raw_shap)}")

        feature_impacts = []
        for i, fname in enumerate(feature_names):
            val = first_row[fname]
            # Convert numpy types to native Python types for JSON serialization
            py_val = float(val) if isinstance(val, (int, float, np.number)) else str(val)
            attribution = float(class_attributions[i])
            feature_impacts.append({
                "feature": fname,
                "value": py_val,
                "attribution": round(attribution, 4),
            })

        # Top positive contributors (pushing towards the predicted class)
        top_positive = sorted(
            [f for f in feature_impacts if f["attribution"] > 0],
            key=lambda x: x["attribution"],
            reverse=True,
        )[:top_k]

        # Top negative contributors (pushing away from the predicted class)
        top_negative = sorted(
            [f for f in feature_impacts if f["attribution"] < 0],
            key=lambda x: x["attribution"],
        )[:top_k]

        target_label = (
            class_labels.get(predicted_class_idx, str(predicted_class_idx))
            if class_labels
            else str(predicted_class_idx)
        )

        pos_str = ", ".join([f"{p['feature']} (+{p['attribution']})" for p in top_positive[:3]])
        summary = (
            f"Top risk drivers for {target_label}: {pos_str}"
            if pos_str
            else f"No strong positive risk drivers identified for {target_label}."
        )

        return {
            "available": True,
            "predicted_class": target_label,
            "top_positive_drivers": top_positive,
            "top_negative_drivers": top_negative,
            "explanation_summary": summary,
        }

    except Exception as exc:
        logger.warning(f"SHAP explanation calculation failed: {exc}")
        return {
            "available": False,
            "predicted_class": str(predicted_class_idx),
            "top_positive_drivers": [],
            "top_negative_drivers": [],
            "explanation_summary": f"Real-time SHAP explanation unavailable: {exc}",
        }
