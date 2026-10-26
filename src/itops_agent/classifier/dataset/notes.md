# IT Operations (ITOps) Classifier - Dataset & Annotation Notes

## 1. Source Dataset
- **Citation**: Tobias Bueck. (2025). *Customer IT Support - Ticket Dataset* [Dataset]. Kaggle. [https://doi.org/10.34740/KAGGLE/DSV/13959267](https://doi.org/10.34740/KAGGLE/DSV/13959267)
- **Original Size**: 28,587 rows × 10 columns

---

## 2. Preprocessing & Filtering Decisions

All preprocessing was carried out in `annotation.ipynb`:

1. **Text Field Consolidation**:
   - `subject` contained 3,838 missing (`NaN`) values.
   - Merged `subject` and `body` into a single `ticket` column using `subject + "\n\n" + body`, safely handling missing subjects and stripping whitespace.
2. **Queue Filtering**:
   - Manually inspected sample queries across all queue classes.
   - Filtered down to the three core IT operations queues:
     - `Technical Support`
     - `IT Support`
     - `Service Outages and Maintenance`
3. **Language Filtering**:
   - Retained only English queries (`language == "en"`). German queries (`de`) were excluded.
4. **Dropped Columns**:
   - `answer`: Contains agent solutions; only raw user queries are required for classification.
   - `queue`: Redundant after filtering to target queues.
   - `priority`: Not relevant to the intrinsic operational complexity of the request.
   - `language`: Redundant after filtering to English.
   - `version`: Irrelevant metadata.
   - `tag_1` & `tag_2`: Irrelevant metadata.
5. **Retained Columns**:
   - `type`: Retained (`Incident`, `Request`, `Problem`, `Change`) as supporting contextual information influencing execution hops.
   - `ticket`: Combined subject and body.
   - `complexity`: Target classification label (`single_hop`, `multi_hop`, `parallel_hop`).
6. **Deduplication**:
   - Verified duplicate ticket texts via `df["ticket"].duplicated().sum()`. Exactly 0 duplicates found.
7. **Cleaned Dataset Size**: Exactly **7,343 rows**.

---

## 3. Hardware Sizing & Model Selection

Hardware profiling was conducted using `llmfit` via `uvx`:
```bash
uvx llmfit recommend --use-case reasoning
```

### System Profile Detected
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM, 7.61 GB free)
- **CPU**: 13th Gen Intel Core i7-13700H (20 threads)
- **RAM**: 15.25 GB System RAM (5.99 GB available)
- **Backend**: CUDA

### Model Finalization
- **Model Selected**: `deepseek-r1:8b` via Ollama.
- **Base Architecture**: Based on `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (8.0B parameters).
- **Rationale**: Optimal hardware fit for 8GB VRAM (utilizing ~5.0 GB VRAM, leaving ~2.6 GB headroom for KV cache).
- **Token Budget**: Verified using `AutoTokenizer` in `token_counter.ipynb`. The rules file `annotation_rules.md` (~900 tokens) comfortably fits within the model's context window.

---

## 4. Annotation Methodology & Guidelines

All **7,343 tickets** were classified into three operational complexity classes defined in `annotation_rules.md`:

### Target Classes
1. **`single_hop`**: Exactly one self-contained, deterministic operational intent with no diagnostic investigation required (e.g., password reset, granting access, restarting a known service).
2. **`multi_hop`**: Requires sequential troubleshooting, dependent actions, or diagnostic investigation of unexplained/recurring failures (e.g., diagnosing intermittent 502s, database connection drops, crash analysis).
3. **`parallel_hop`**: At least two distinct, independent operational intents executed concurrently (e.g., checking disk usage on Server A and SSL cert on Server B).

### Execution Parameters
- **Inference Runtime**: Ollama local server (`http://localhost:11434/api/chat`).
- **Sampling Temperature**: `0.0` (fully deterministic).
- **Format**: Structured JSON output enforcing `{"reasoning": "...", "label": "..."}`.
- **Output Files**:
  - `annotated_dataset.csv`: Progress checkpoint export.
  - `final.csv`: Production dataset containing `['type', 'ticket', 'complexity']` across all 7,343 rows.

---

## 5. Dataset Splitting Strategy for Fine-Tuning

To train, tune, and evaluate the final production classifier on the 7,343 tickets, a **Stratified 80 / 10 / 10 Split** is established:

| Split | Percentage | Sample Count | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **Train Set** | **80%** | **~5,874** | Model weight updates and gradient backpropagation. Balanced class weighting applied. |
| **Validation Set** | **10%** | **~734** | Epoch-by-epoch evaluation, learning rate tuning, and early stopping checkpoint selection. |
| **Test Set** | **10%** | **~735** | Held-out unbiased final evaluation. Never touched during training or hyperparameter tuning. |

> **Note on Stratification**: Due to the natural class distribution (where `parallel_hop` is a minority class), all splits must use stratified sampling to ensure identical class proportions across train, validation, and test sets.
