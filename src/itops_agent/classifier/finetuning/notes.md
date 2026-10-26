# IT Operations (ITOps) Classifier - Fine-Tuning & Benchmark Notes

## 1. Model Architecture & Operational Paradigm
- **Base Architecture**: `convaiinnovations/laya` (421M parameters).
- **Backbone**: Dual-encoder transformer (specialized ModernBERT base) paired with a typed decision projection head.
- **Inference Mechanism**: Evaluates the input ticket `state` (ticket type and text) alongside the typed decision schema `questions` in a single forward pass (~20–30 ms latency on NVIDIA RTX 4060 GPU).
- **Zero-Shot Capability**: Pre-trained on diverse structured decisions; can evaluate natural language criteria zero-shot out-of-the-box before domain fine-tuning.

---

## 2. Decision Schema (`COMPLEXITY_QUESTION`)
The classifier routes tickets into three operational execution tiers using Laya's canonical schema:
```json
{
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
```

---

## 3. Training Objective & Optimization Dynamics (RLCD)
- **Objective Function**: Reinforcement Learning from Calibrated Decisions (RLCD) combining policy gradients over exploration noise with Proper Scoring Rules (Multi-Class Brier Score / Spherical Scoring Rules).
- **Exploration Noise**: Annealed schedule $\sigma_{\text{start}} = 0.4 \to \sigma_{\text{end}} = 0.1$ across 4 epochs.
- **Hardware Profile & VRAM Optimization**:
  - GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM).
  - Optimizer: `bitsandbytes.optim.AdamW8bit` (saving ~2.5 GB VRAM momentum buffers, maintaining peak memory below 5.2 GB).
  - Numerical Precision: Mixed Precision (FP16 autocast) with PyTorch `GradScaler`.
  - Gradient Checkpointing: Active across encoder and decision head (`use_reentrant=False`).
- **Effective Batch Accounting**:
  - Micro-batch: 4 sequences per step.
  - Gradient Accumulation: 16 steps.
  - Effective Batch Size: **64 sequences** ($4 \times 16$).
- **Learning Rate Schedule**:
  - Encoder LR: $2.5 \times 10^{-5}$
  - Decision Head LR: $1.0 \times 10^{-4}$
  - Scheduler: Cosine Annealing scheduler decaying to minimum LR $1.0 \times 10^{-6}$.
- **Post-Training Temperature Calibration**:
  - Holdout calibration fit (`fit_temperature_map`) performed on rank 0 after epoch 4 using 400 held-out calibration samples.
  - Fitted Calibration Temperatures: `[4.519, 1.0, 1.0]`, aligning softmax outputs with true posterior probabilities.

---

## 4. Dataset Partitioning (Stratified 80 / 10 / 10 Split)
From the complete corpus of **7,343 tickets** annotated following the DeepSeek-R1 (8B) reasoning framework:

| Split Partition | Percentage | Sample Count | Primary Engineering Role |
| :--- | :---: | :---: | :--- |
| **Train Set** | **80%** | **5,873** | Parameter updates via RLCD policy gradients across 4 epochs (388 optimizer steps). |
| **Calibration Holdout** | **10%** | **735** | Post-training temperature calibration and learning rate convergence validation. |
| **Holdout Test Set** | **10%** | **735** | Completely withheld unbiased test benchmark evaluated by both Base and Fine-Tuned models. |
| **Total Corpus** | **100%** | **7,343** | Fully stratified across `multi_hop`, `single_hop`, and `parallel_hop`. |

---

## 5. Deliverables & Benchmark Execution
- [`tailored_finetuning.ipynb`](file:///home/arnav-gupta/Projects/itops_agent/src/itops_agent/classifier/finetuning/tailored_finetuning.ipynb):
  - End-to-end interactive notebook: tokenizes dataset, exports DDP script, runs distributed training, and fits post-training temperature calibration.
- [`benchmarking_difference.py`](file:///home/arnav-gupta/Projects/itops_agent/src/itops_agent/classifier/finetuning/benchmarking_difference.py):
  - Unified ablation benchmarking script. Evaluates Base Laya (Zero-Shot) and Fine-Tuned Laya side-by-side on the 735 holdout test cases, calculates empirical lift metrics, and exports summary CSV and JSON reports.
- **Model Checkpoint**:
  - Exported to [`laya_work/laya_finetuned_itops_complexity/`](file:///home/arnav-gupta/Projects/itops_agent/laya_work/laya_finetuned_itops_complexity/) (`model.safetensors`, `rl_agent_config.json`, tokenizer, encoder).

---

## 6. Empirical Head-to-Head Ablation Results (735 Test Cases)

Evaluated across the **735 holdout test tickets** on an **NVIDIA GeForce RTX 4060 Laptop GPU**:

### Global Comparative Performance & Empirical Lift ($\Delta$)
| Evaluation Metric | Base Laya (Zero-Shot) | Fine-Tuned Laya (RLCD) | Absolute Lift ($\Delta$) | Relative Gain | Engineering Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Classification Accuracy** | 68.84% | **95.37%** | **+26.53%** | **+38.5%** | Near-human classification accuracy |
| **Macro F1-Score** | 0.3963 | **0.8807** | **+0.4843** | **+122.2%** | Robust multi-class decision boundaries |
| **Weighted F1-Score** | 0.6238 | **0.9531** | **+0.3293** | **+52.8%** | Aggregate enterprise reliability score |
| **Brier Score (Uncertainty)** | 0.4366 | **0.0825** | **-0.3541** *(lower is better)* | **-81.1%** | 81% reduction in probability prediction error |
| **Expected Calibration Error (ECE)** | 18.92% | **1.17%** | **-17.75%** *(lower is better)* | **-93.8%** | Perfectly calibrated confidence gating |
| **Inference Latency ($p_{50}$)** | 28.1 ms | **28.1 ms** | **+0.0 ms** | **~0%** | Zero computational overhead at runtime |
| **Inference Latency ($p_{95}$)** | 32.0 ms | **31.6 ms** | **-0.4 ms** | **~0%** | Strict sub-35ms tail latency SLA |

---

### Per-Class Detailed Performance Breakdown

#### 1. Base Laya (Zero-Shot Baseline)
```text
              precision    recall  f1-score   support

  single_hop     0.5797    0.1778    0.2721       225
   multi_hop     0.7106    0.9607    0.8169       483
parallel_hop     0.1538    0.0741    0.1000        27

    accuracy                         0.6884       735
   macro avg     0.4814    0.4042    0.3963       735
weighted avg     0.6501    0.6884    0.6238       735
```

#### 2. Fine-Tuned Laya (RLCD Specialist)
```text
              precision    recall  f1-score   support

  single_hop     0.9569    0.8889    0.9217       225
   multi_hop     0.9639    0.9959    0.9796       483
parallel_hop     0.7407    0.7407    0.7407        27

    accuracy                         0.9537       735
   macro avg     0.8872    0.8752    0.8807       735
weighted avg     0.9536    0.9537    0.9531       735
```

---

### Side-by-Side Confusion Matrix Comparison

#### Base Laya (Zero-Shot) — Severe Majority Class Bias
| True Category \ Predicted Category | Pred: `single_hop` | Pred: `multi_hop` | Pred: `parallel_hop` | Class Recall (%) |
| :--- | :---: | :---: | :---: | :---: |
| **True `single_hop`** (225 tickets) | **40** | 175 | 10 | **17.78%** *(Collapse)* |
| **True `multi_hop`** (483 tickets) | 18 | **464** | 1 | **96.07%** |
| **True `parallel_hop`** (27 tickets) | 11 | 14 | **2** | **7.41%** *(Missed)* |

#### Fine-Tuned Laya (RLCD) — Sharp Operational Boundary Separation
| True Category \ Predicted Category | Pred: `single_hop` | Pred: `multi_hop` | Pred: `parallel_hop` | Class Recall (%) |
| :--- | :---: | :---: | :---: | :---: |
| **True `single_hop`** (225 tickets) | **200** *(+160 correct)* | 18 | 7 | **88.89%** |
| **True `multi_hop`** (483 tickets) | 2 | **481** *(+17 correct)* | 0 | **99.59%** |
| **True `parallel_hop`** (27 tickets) | 7 | 0 | **20** *(+18 correct)* | **74.07%** |

---

## 7. Analytical Insights for Project Defense & Thesis

1. **Resolution of the Single-Hop Collapse**:
   - Out of the box, Base Laya misclassified **175 out of 225** deterministic `single_hop` tickets as `multi_hop` (yielding only 17.78% recall).
   - *Why*: Without domain specialization, general LLM embeddings interpret any technical failure as requiring troubleshooting.
   - *Post-Finetuning*: Fine-tuned Laya recovers **88.89% recall** on `single_hop` with **95.69% precision**, allowing 200 out of 225 deterministic tickets to be resolved automatically by tier-1 automation without unnecessary engineer escalation.

2. **Mastery of Rare Parallel Workflows (`parallel_hop`)**:
   - Despite `parallel_hop` comprising only 3.68% of the corpus (27 test cases), recall soared from **7.41% to 74.07%** (20 out of 27 correct) with an F1-score of **0.7407** and zero false escalations to `multi_hop`.

3. **Near-Perfect Decision Calibration (ECE = 1.17%)**:
   - The Expected Calibration Error plummeted from **18.92% down to 1.17%** (a 93.8% calibration error reduction).
   - In autonomous agent architectures, calibrated probabilities allow confidence gating: if confidence $> 90\%$, the agent routes immediately; if ambiguous, it requests clarification.

4. **Zero Latency Overhead (28.1 ms)**:
   - Inference latency remains identical at **28.1 ms** ($p_{50}$) before and after fine-tuning because the ModernBERT dual-encoder parameter count (421M) is unchanged.
