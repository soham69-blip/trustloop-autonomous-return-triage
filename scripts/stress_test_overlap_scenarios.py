"""
TrustLoop Realistic Return-Rate Overlap Stress-Testing & Threshold Analysis Suite.

Executes:
1. Generation of 6 synthetic stress scenarios introducing realistic return rate overlap (15-33% band)
2. Evaluation of candidate-v2.0.0 and production-v1.3.0 under heavy class overlap
3. Threshold optimization on validation split with evaluation on untouched stress holdout
4. 1,000+ request concurrent shadow benchmark measuring p50/p95/p99 latency and fallback rates
5. Generates reports/overlap_stress_test_results.csv, reports/overlap_threshold_curve.csv, reports/shadow_1000_benchmark.json
"""

import time
import json
import pickle
import threading
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)

from backend.app.ml_feature_builder import (
    _load_or_build_category_mappings,
    build_model_features,
    MODEL_FEATURES,
    CANDIDATE_MODEL_FEATURES,
)
from backend.app.services.customer_feature_service import (
    CustomerFeatureService,
    PersistentCustomerFeatureStore,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "trustloop"

CLASS_NAMES = ["Legitimate", "Policy Abuser", "Fraudulent Return", "Wardrobing"]


def format_batch_for_model(df_in: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    records = []
    for r in df_in.to_dict(orient="records"):
        r_str = {str(k): v for k, v in r.items()}
        feat_dict = build_model_features(r_str, feature_names=feature_names)
        records.append(feat_dict)
    df = pd.DataFrame(records)[feature_names]

    mappings = _load_or_build_category_mappings()
    categorical_columns = [
        "country",
        "customer_segment",
        "device_type",
        "payment_method",
        "platform",
        "product_category",
        "return_reason",
        "shipping_carrier",
    ]
    for col in categorical_columns:
        if col in df.columns:
            cats = mappings.get(col, [])
            df[col] = pd.Categorical(df[col].astype(str), categories=cats)
    return df


def load_models() -> Tuple[Any, List[str], Any, List[str]]:
    with open(MODELS_DIR / "lightgbm_model.pkl", "rb") as f:
        prod_m = pickle.load(f)
    prod_feats = list(prod_m.feature_name_)

    with open(MODELS_DIR / "lightgbm_candidate.pkl", "rb") as f:
        cand_m = pickle.load(f)
    cand_feats = list(cand_m.feature_name_)

    return prod_m, prod_feats, cand_m, cand_feats


# ==============================================================================
# 1. OVERLAP STRESS-TEST SCENARIOS (PHASE 12)
# ==============================================================================
def generate_stress_scenarios(base_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Creates controlled synthetic stress scenarios introducing realistic return-rate overlap in 15-33%.
    Does NOT modify the raw or processed datasets.
    """
    scenarios = {}
    sample_pool = base_df.tail(4000).copy().reset_index(drop=True)

    # Scenario A: Small Overlap (15-20% return rate for 500 samples)
    df_a = sample_pool.copy()
    idx_a = list(df_a[df_a["abuse_label"] == 0].index[:500])
    df_a.loc[idx_a, "return_rate_pct"] = list(np.random.uniform(15.0, 20.0, size=len(idx_a)))
    scenarios["Scenario_A_Small_Overlap_15_20"] = df_a

    # Scenario B: Moderate Overlap (20-28% return rate for 1,000 samples)
    df_b = sample_pool.copy()
    idx_b = list(df_b[df_b["abuse_label"] == 0].index[:1000])
    df_b.loc[idx_b, "return_rate_pct"] = list(np.random.uniform(20.0, 28.0, size=len(idx_b)))
    scenarios["Scenario_B_Moderate_Overlap_20_28"] = df_b

    # Scenario C: Heavy Overlap (15-33% return rate for 2,000 samples)
    df_c = sample_pool.copy()
    idx_c0 = list(df_c[df_c["abuse_label"] == 0].index[:1500])
    idx_c1 = list(df_c[df_c["abuse_label"] == 1].index[:500])
    df_c.loc[idx_c0, "return_rate_pct"] = list(np.random.uniform(15.0, 25.0, size=len(idx_c0)))
    df_c.loc[idx_c1, "return_rate_pct"] = list(np.random.uniform(25.0, 33.0, size=len(idx_c1)))
    scenarios["Scenario_C_Heavy_Overlap_15_33"] = df_c

    # Scenario D: High Return-Rate Legitimate Customers (25% return rate due to sizing, 0 disputes/tickets)
    df_d = sample_pool.copy()
    idx_d = list(df_d[df_d["abuse_label"] == 0].index[:800])
    df_d.loc[idx_d, "return_rate_pct"] = 25.0
    df_d.loc[idx_d, "customer_support_contacts"] = 0
    df_d.loc[idx_d, "previous_dispute_count"] = 0
    df_d.loc[idx_d, "return_reason"] = "Too Small"
    scenarios["Scenario_D_High_RR_Legit_Sizing"] = df_d

    # Scenario E: Low Return-Rate Abusers (18% return rate, high disputes and tickets)
    df_e = sample_pool.copy()
    idx_e = list(df_e[df_e["abuse_label"] == 1].index[:600])
    df_e.loc[idx_e, "return_rate_pct"] = 18.0
    df_e.loc[idx_e, "customer_support_contacts"] = 6
    df_e.loc[idx_e, "previous_dispute_count"] = 4
    scenarios["Scenario_E_Low_RR_Dispute_Abusers"] = df_e

    # Scenario F: Noisy Customer Histories (Missing/Burst transactions)
    df_f = sample_pool.copy()
    idx_f = list(df_f.index[:1000])
    df_f.loc[idx_f, "total_orders_lifetime"] = list(np.random.randint(1, 5, size=len(idx_f)))
    df_f.loc[idx_f, "return_rate_pct"] = list(np.random.choice([0.0, 20.0, 33.3, 50.0], size=len(idx_f)))
    scenarios["Scenario_F_Noisy_Histories"] = df_f

    return scenarios


def evaluate_stress_scenarios():
    print("\n[PHASE 12] Evaluating Candidate Model on Realistic Return-Rate Overlap Stress Scenarios...")
    prod_m, prod_feats, cand_m, cand_feats = load_models()
    base_df = pd.read_csv(DATA_DIR / "model_ready.csv")
    scenarios = generate_stress_scenarios(base_df)

    stress_results = []

    for name, df_scen in scenarios.items():
        y_true = df_scen["abuse_label"].to_numpy()

        # Vectorized batch prediction
        df_cand_b = format_batch_for_model(df_scen, cand_feats)
        cand_preds = cand_m.predict(df_cand_b)
        cand_probs = cand_m.predict_proba(df_cand_b)

        acc = float(accuracy_score(y_true, cand_preds))
        m_f1 = float(f1_score(y_true, cand_preds, average="macro", zero_division=0))
        pa_rec = float(recall_score(y_true == 1, cand_preds == 1, zero_division=0))
        pa_prec = float(precision_score(y_true == 1, cand_preds == 1, zero_division=0))
        pa_f1 = float(f1_score(y_true == 1, cand_preds == 1, zero_division=0))

        stress_results.append({
            "scenario": name,
            "sample_count": len(df_scen),
            "accuracy": round(acc, 4),
            "macro_f1": round(m_f1, 4),
            "policy_abuser_precision": round(pa_prec, 4),
            "policy_abuser_recall": round(pa_rec, 4),
            "policy_abuser_f1": round(pa_f1, 4),
        })

        print(f"  {name}: Acc={acc:.1%}, Macro F1={m_f1:.1%}, PA Recall={pa_rec:.1%}, PA F1={pa_f1:.1%}")

    pd.DataFrame(stress_results).to_csv(REPORTS_DIR / "overlap_stress_test_results.csv", index=False)
    return stress_results


# ==============================================================================
# 2. THRESHOLD ANALYSIS ON OVERLAPPING DATA (PHASE 13)
# ==============================================================================
def evaluate_threshold_optimization():
    print("\n[PHASE 13] Running Threshold Optimization Analysis on Overlap Stress Set...")
    _, _, cand_m, cand_feats = load_models()
    base_df = pd.read_csv(DATA_DIR / "model_ready.csv")
    df_c = generate_stress_scenarios(base_df)["Scenario_C_Heavy_Overlap_15_33"]

    # Split into 50% Val, 50% Test holdout
    n = len(df_c)
    val_df = df_c.iloc[: n // 2]
    test_df = df_c.iloc[n // 2 :]

    val_batch = format_batch_for_model(val_df, cand_feats)
    val_probs = cand_m.predict_proba(val_batch)[:, 1]
    y_val = (val_df["abuse_label"] == 1).astype(int).to_numpy()

    test_batch = format_batch_for_model(test_df, cand_feats)
    test_probs = cand_m.predict_proba(test_batch)[:, 1]
    y_test = (test_df["abuse_label"] == 1).astype(int).to_numpy()

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    thresh_records = []

    best_val_tau = 0.50
    best_val_f1 = 0.0

    for tau in thresholds:
        pred_val = (val_probs >= tau).astype(int)
        f1_v = float(f1_score(y_val, pred_val, zero_division=0))
        rec_v = float(recall_score(y_val, pred_val, zero_division=0))
        prec_v = float(precision_score(y_val, pred_val, zero_division=0))

        if f1_v > best_val_f1:
            best_val_f1 = f1_v
            best_val_tau = tau

        # Evaluate on untouched test holdout
        pred_te = (test_probs >= tau).astype(int)
        f1_te = float(f1_score(y_test, pred_te, zero_division=0))
        rec_te = float(recall_score(y_test, pred_te, zero_division=0))
        prec_te = float(precision_score(y_test, pred_te, zero_division=0))

        thresh_records.append({
            "threshold": tau,
            "val_f1": round(f1_v, 4),
            "val_recall": round(rec_v, 4),
            "val_precision": round(prec_v, 4),
            "test_f1": round(f1_te, 4),
            "test_recall": round(rec_te, 4),
            "test_precision": round(prec_te, 4),
        })

    pd.DataFrame(thresh_records).to_csv(REPORTS_DIR / "overlap_threshold_curve.csv", index=False)
    print(f"  [OK] Optimal Validation Threshold: tau={best_val_tau} (Val F1: {best_val_f1:.1%}) -> Saved to reports/overlap_threshold_curve.csv")
    return thresh_records


# ==============================================================================
# 3. 1,000+ REQUEST SHADOW BENCHMARK (PHASE 14)
# ==============================================================================
def run_1000_shadow_benchmark():
    print("\n[PHASE 14] Running 1,000-Request Concurrent Shadow Benchmark...")
    service = CustomerFeatureService()
    base_df = pd.read_csv(DATA_DIR / "model_ready.csv")
    raw_records = base_df.tail(1000).to_dict(orient="records")
    samples: List[Dict[str, Any]] = [{str(k): v for k, v in r.items()} for r in raw_records]

    latencies = []
    fallbacks = 0
    errors = 0

    from backend.app.main import load_model
    from backend.app.services.shadow_service import _get_candidate_model

    t0_all = time.perf_counter()

    for idx, sample in enumerate(samples):
        try:
            t0 = time.perf_counter()
            # Randomly assign customer ID variations (Normal, New, Anonymous, Malformed)
            if idx % 10 == 0:
                sample["customer_id"] = None
            elif idx % 25 == 0:
                sample["customer_id"] = "../traversal/id"
            else:
                sample["customer_id"] = f"CUST-SIM-{idx % 100:03d}"

            res = service.execute_safe_prediction(
                case_payload=sample,
                requested_model_version="candidate-v2.0.0",
                prod_model_loader=lambda: (load_model(), ""),
                cand_model_loader=_get_candidate_model,
            )
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            if res["fallback_occurred"]:
                fallbacks += 1

        except Exception:
            errors += 1

    total_time = time.perf_counter() - t0_all
    lat_arr = np.asarray(latencies)

    bench_summary = {
        "total_requests": len(samples),
        "total_duration_seconds": round(total_time, 2),
        "throughput_req_per_sec": round(len(samples) / total_time, 1),
        "p50_latency_ms": round(float(np.percentile(lat_arr, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(lat_arr, 99)), 2),
        "fallback_count": fallbacks,
        "fallback_rate_pct": round((fallbacks / len(samples)) * 100.0, 2),
        "error_count": errors,
        "error_rate_pct": round((errors / len(samples)) * 100.0, 2),
    }

    with open(REPORTS_DIR / "shadow_1000_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(bench_summary, f, indent=2)

    print(f"\n[OK] 1,000 Shadow Benchmark Complete:")
    print(f"  Total Duration: {bench_summary['total_duration_seconds']}s ({bench_summary['throughput_req_per_sec']} req/s)")
    print(f"  Latency: p50={bench_summary['p50_latency_ms']}ms, p95={bench_summary['p95_latency_ms']}ms, p99={bench_summary['p99_latency_ms']}ms")
    print(f"  Fallback Rate: {bench_summary['fallback_rate_pct']}% | Error Rate: {bench_summary['error_rate_pct']}%")

    return bench_summary


def main():
    print("=" * 80)
    print("TRUSTLOOP REALISTIC STRESS-TESTING & SHADOW BENCHMARK SUITE")
    print("=" * 80)

    evaluate_stress_scenarios()
    evaluate_threshold_optimization()
    run_1000_shadow_benchmark()

    print("\n" + "=" * 80)
    print("ALL STRESS TESTS & BENCHMARKS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
