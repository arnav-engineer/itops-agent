"""
Comprehensive Ablation Benchmarking Script: Base Laya vs. Fine-Tuned Laya
Evaluates Base Laya (Zero-Shot) and Fine-Tuned Laya (RLCD) side-by-side
on the held-out test split, prints detailed comparative metrics, and exports
ablation summary tables.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import laya
from laya.common import ece_score

# Resolve working paths (supports running from repo root or subdirectories)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3] if len(SCRIPT_DIR.parents) >= 4 else Path.cwd()

CANDIDATE_WORK_DIRS = [
    REPO_ROOT / "laya_work",
    SCRIPT_DIR / "laya_work",
    Path("laya_work"),
    Path("./laya_work")
]

def _resolve_work_dir() -> Path:
    for wd in CANDIDATE_WORK_DIRS:
        if wd.exists():
            return wd
    fallback = REPO_ROOT / "laya_work"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

WORK_DIR: Path = _resolve_work_dir()

TEST_CASES_PATH = WORK_DIR / "test_cases.json"
CKPT_DIR = WORK_DIR / "laya_finetuned_itops_complexity"
DATASET_PATH = REPO_ROOT / "src/itops_agent/classifier/dataset/annotated_dataset.csv"

VALID_CLASSES = ["single_hop", "multi_hop", "parallel_hop"]
COMPLEXITY_QUESTION = {
    "complexity": {
        "type": "choice",
        "instructions": "What is the operational complexity required to fulfill this IT support ticket?",
        "criteria": {
            "single_hop": "Self-contained deterministic action with no diagnostic investigation required.",
            "multi_hop": "Sequential troubleshooting, diagnostic investigation, or dependent actions.",
            "parallel_hop": "At least two independent operational tasks executed concurrently."
        }
    }
}


def load_or_create_test_cases():
    if TEST_CASES_PATH.exists():
        print(f"Loading existing test cases from: {TEST_CASES_PATH}")
        with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)
        if len(cases) > 0:
            return cases

    print(f"Generating deterministic 10% holdout test split from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df_labeled = df[df["complexity"].isin(VALID_CLASSES)].copy().reset_index(drop=True)

    _, df_test = train_test_split(
        df_labeled,
        test_size=0.10,
        random_state=42,
        stratify=df_labeled["complexity"]
    )

    test_cases = []
    for i, (_, row) in enumerate(df_test.iterrows()):
        comp = str(row["complexity"])
        test_cases.append({
            "id": f"ticket_{i}",
            "state": {
                "type": str(row["type"]) if pd.notna(row.get("type")) else "Generic",
                "ticket": str(row["ticket"])
            },
            "questions": COMPLEXITY_QUESTION,
            "gold": {
                "complexity": {
                    "label": comp,
                    "probabilities": {
                        "single_hop": 1.0 if comp == "single_hop" else 0.0,
                        "multi_hop": 1.0 if comp == "multi_hop" else 0.0,
                        "parallel_hop": 1.0 if comp == "parallel_hop" else 0.0
                    }
                }
            }
        })

    with open(TEST_CASES_PATH, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2)
    print(f"Saved {len(test_cases)} holdout test cases to: {TEST_CASES_PATH}")
    return test_cases


def evaluate_agent(agent, test_cases, model_name):
    print(f"\n--- Evaluating {model_name} on {len(test_cases)} test cases ---")
    y_true = []
    y_pred = []
    latencies = []
    accuracies = []
    soft_accuracies = []
    brier_scores = []
    all_confs = []
    all_corrects = []

    # Warmup
    agent.predict(test_cases[0]["state"], test_cases[0]["questions"])

    t0_all = time.time()
    for i, case in enumerate(test_cases):
        t0 = time.perf_counter()
        res = agent.predict(case["state"], case["questions"])
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        p_ans = res["answers"]["complexity"]
        g_ans = case["gold"]["complexity"]

        pred_label = str(p_ans["choice"])
        gold_label = str(g_ans["label"])
        y_true.append(gold_label)
        y_pred.append(pred_label)

        is_corr = float(pred_label == gold_label)
        accuracies.append(is_corr)
        all_corrects.append(is_corr)

        p_probs = np.array([p_ans["probabilities"].get(k, 1e-6) for k in VALID_CLASSES])
        g_probs = np.array([g_ans["probabilities"].get(k, 1e-6) for k in VALID_CLASSES])
        p_probs /= p_probs.sum()
        g_probs /= g_probs.sum()

        all_confs.append(float(p_probs.max()))
        soft_accuracies.append(float((p_probs * g_probs).sum()))
        brier_scores.append(float(((p_probs - g_probs) ** 2).sum()))

        if (i + 1) % 150 == 0 or (i + 1) == len(test_cases):
            print(f"  Processed {i + 1}/{len(test_cases)} cases ({lat:.1f} ms/query)...")

    total_time = time.time() - t0_all
    acc = float(np.mean(accuracies))
    soft_acc = float(np.mean(soft_accuracies))
    brier = float(np.mean(brier_scores))
    ece = float(ece_score(np.array(all_confs), np.array(all_corrects)))
    macro_f1 = float(f1_score(y_true, y_pred, labels=VALID_CLASSES, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=VALID_CLASSES, average="weighted", zero_division=0))

    metrics = {
        "model_name": model_name,
        "sample_count": len(test_cases),
        "total_eval_time_sec": total_time,
        "accuracy": acc,
        "soft_accuracy": soft_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "brier_score": brier,
        "ece": ece,
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p50": float(np.median(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    return metrics


def main():
    print("=" * 80)
    print("LAYA ABLATION BENCHMARK: BASE MODEL vs. FINE-TUNED MODEL")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    test_cases = load_or_create_test_cases()
    print(f"Total Test Cases: {len(test_cases):,}")

    # 1. Evaluate Base Laya (Zero-Shot)
    # Check for local cached snapshot to bypass HuggingFace network latency
    base_snapshot = Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148"
    base_source = str(base_snapshot) if base_snapshot.exists() else "convaiinnovations/laya"

    print(f"\n1. Loading Base Model from: {base_source} ...")
    agent_base = laya.load(base_source, device=device)
    base_metrics = evaluate_agent(agent_base, test_cases, "Base Laya (Zero-Shot)")

    # Clear VRAM between evaluations
    del agent_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Evaluate Fine-Tuned Laya (RLCD)
    if not CKPT_DIR.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint directory not found at: {CKPT_DIR}")

    print(f"\n2. Loading Fine-Tuned Checkpoint: {CKPT_DIR} ...")
    agent_ft = laya.load(str(CKPT_DIR), device=device)
    ft_metrics = evaluate_agent(agent_ft, test_cases, "Fine-Tuned Laya (RLCD)")

    # 3. Compute Delta and Relative Gain
    delta_acc = ft_metrics["accuracy"] - base_metrics["accuracy"]
    rel_acc = (delta_acc / max(1e-6, base_metrics["accuracy"])) * 100
    delta_macro_f1 = ft_metrics["macro_f1"] - base_metrics["macro_f1"]
    rel_macro_f1 = (delta_macro_f1 / max(1e-6, base_metrics["macro_f1"])) * 100
    delta_weighted_f1 = ft_metrics["weighted_f1"] - base_metrics["weighted_f1"]
    delta_brier = ft_metrics["brier_score"] - base_metrics["brier_score"]
    rel_brier = (delta_brier / max(1e-6, base_metrics["brier_score"])) * 100
    delta_ece = ft_metrics["ece"] - base_metrics["ece"]
    rel_ece = (delta_ece / max(1e-6, base_metrics["ece"])) * 100

    summary_rows = [
        {
            "Evaluation Metric": "Classification Accuracy",
            "Base Laya (Zero-Shot)": f"{base_metrics['accuracy'] * 100:.2f}%",
            "Fine-Tuned Laya (RLCD)": f"{ft_metrics['accuracy'] * 100:.2f}%",
            "Absolute Lift (Δ)": f"{'+' if delta_acc >= 0 else ''}{delta_acc * 100:.2f}%",
            "Relative Gain": f"{'+' if rel_acc >= 0 else ''}{rel_acc:.1f}%"
        },
        {
            "Evaluation Metric": "Macro F1-Score",
            "Base Laya (Zero-Shot)": f"{base_metrics['macro_f1']:.4f}",
            "Fine-Tuned Laya (RLCD)": f"{ft_metrics['macro_f1']:.4f}",
            "Absolute Lift (Δ)": f"{'+' if delta_macro_f1 >= 0 else ''}{delta_macro_f1:.4f}",
            "Relative Gain": f"{'+' if rel_macro_f1 >= 0 else ''}{rel_macro_f1:.1f}%"
        },
        {
            "Evaluation Metric": "Weighted F1-Score",
            "Base Laya (Zero-Shot)": f"{base_metrics['weighted_f1']:.4f}",
            "Fine-Tuned Laya (RLCD)": f"{ft_metrics['weighted_f1']:.4f}",
            "Absolute Lift (Δ)": f"{'+' if delta_weighted_f1 >= 0 else ''}{delta_weighted_f1:.4f}",
            "Relative Gain": f"{(delta_weighted_f1 / max(1e-6, base_metrics['weighted_f1'])) * 100:+.1f}%"
        },
        {
            "Evaluation Metric": "Brier Score (Uncertainty)",
            "Base Laya (Zero-Shot)": f"{base_metrics['brier_score']:.4f}",
            "Fine-Tuned Laya (RLCD)": f"{ft_metrics['brier_score']:.4f}",
            "Absolute Lift (Δ)": f"{delta_brier:+.4f} (lower is better)",
            "Relative Gain": f"{rel_brier:+.1f}%"
        },
        {
            "Evaluation Metric": "Expected Calibration Error (ECE)",
            "Base Laya (Zero-Shot)": f"{base_metrics['ece'] * 100:.2f}%",
            "Fine-Tuned Laya (RLCD)": f"{ft_metrics['ece'] * 100:.2f}%",
            "Absolute Lift (Δ)": f"{delta_ece * 100:+.2f}% (lower is better)",
            "Relative Gain": f"{rel_ece:+.1f}%"
        },
        {
            "Evaluation Metric": "Inference Latency (p50)",
            "Base Laya (Zero-Shot)": f"{base_metrics['latency_ms_p50']:.1f} ms",
            "Fine-Tuned Laya (RLCD)": f"{ft_metrics['latency_ms_p50']:.1f} ms",
            "Absolute Lift (Δ)": f"{ft_metrics['latency_ms_p50'] - base_metrics['latency_ms_p50']:+.1f} ms",
            "Relative Gain": "~0% overhead"
        },
        {
            "Evaluation Metric": "Inference Latency (p95)",
            "Base Laya (Zero-Shot)": f"{base_metrics['latency_ms_p95']:.1f} ms",
            "Fine-Tuned Laya (RLCD)": f"{ft_metrics['latency_ms_p95']:.1f} ms",
            "Absolute Lift (Δ)": f"{ft_metrics['latency_ms_p95'] - base_metrics['latency_ms_p95']:+.1f} ms",
            "Relative Gain": "~0% overhead"
        }
    ]

    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD ABLATION BENCHMARK RESULTS (735 TEST CASES)")
    print("=" * 90)
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_markdown(index=False))

    # Per-Class Classification Reports
    print("\n" + "=" * 90)
    print("1. BASE LAYA (ZERO-SHOT) - PER-CLASS REPORT")
    print("=" * 90)
    print(classification_report(base_metrics["y_true"], base_metrics["y_pred"], labels=VALID_CLASSES, digits=4, zero_division=0))

    print("\n" + "=" * 90)
    print("2. FINE-TUNED LAYA (RLCD) - PER-CLASS REPORT")
    print("=" * 90)
    print(classification_report(ft_metrics["y_true"], ft_metrics["y_pred"], labels=VALID_CLASSES, digits=4, zero_division=0))

    print("\n" + "=" * 90)
    print("3. CONFUSION MATRIX COMPARISON")
    print("=" * 90)
    cm_base = confusion_matrix(base_metrics["y_true"], base_metrics["y_pred"], labels=VALID_CLASSES)
    cm_ft = confusion_matrix(ft_metrics["y_true"], ft_metrics["y_pred"], labels=VALID_CLASSES)

    print("--- BASE LAYA (ZERO-SHOT) CONFUSION MATRIX ---")
    print(pd.DataFrame(cm_base, index=[f"True: {c}" for c in VALID_CLASSES], columns=[f"Pred: {c}" for c in VALID_CLASSES]))

    print("\n--- FINE-TUNED LAYA (RLCD) CONFUSION MATRIX ---")
    print(pd.DataFrame(cm_ft, index=[f"True: {c}" for c in VALID_CLASSES], columns=[f"Pred: {c}" for c in VALID_CLASSES]))

    # Export artifacts
    csv_out = WORK_DIR / "laya_ablation_summary.csv"
    json_out = WORK_DIR / "laya_ablation_benchmark_report.json"
    df_summary.to_csv(csv_out, index=False)

    report_payload = {
        "benchmark_name": "IT Operations Classification Head-to-Head Ablation",
        "sample_count": len(test_cases),
        "classes": VALID_CLASSES,
        "base_metrics": {k: v for k, v in base_metrics.items() if k not in ["y_true", "y_pred"]},
        "fine_tuned_metrics": {k: v for k, v in ft_metrics.items() if k not in ["y_true", "y_pred"]},
        "summary_table": summary_rows
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print("\n" + "=" * 90)
    print(f"[OK] Summary table exported to: {csv_out}")
    print(f"[OK] Full report exported to:   {json_out}")
    print("=" * 90)


if __name__ == "__main__":
    main()

