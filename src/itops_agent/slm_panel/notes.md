# SLM Panel: Actor-Critic Single-Hop Resolution System

## 1. Architectural Paradigm & System Overview

The **SLM Panel** serves as the autonomous execution engine for deterministic, single-hop IT support tickets identified by the Tier-0 Laya classifier (`single_hop` route).

### The Two-Stage Actor-Critic Architecture
```
                           [ User Ticket ]
                                  │
                                  ▼
                     [ Tier-0 Laya Classifier ]
                                  │
                        (routes: single_hop)
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │       SLM PANEL GATEWAY         │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ 1. Generator Model (Actor)    │
                  │    - Runtime: Ollama Local    │
                  │    - Model: qwen2.5:3b        │
                  │    - Task: Drafts resolution  │
                  └───────────────┬───────────────┘
                                  │
                           [ Draft Response ]
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ 2. Critic Model (Gatekeeper)  │
                  │    - Runtime: Groq Cloud LPU  │
                  │    - Model: openai/gpt-oss-120b│
                  │    - Output: {"valid": bool}  │
                  └───────────────┬───────────────┘
                                  │
                       Is resolution valid?
                         /             \
                   [YES]                 [NO]
                    /                     \
                   ▼                       ▼
          [ Return to User ]      [ Log Failure & Escalate ]
          (Direct Fast-Path)      (Autonomous Circuit Breaker)
```

---

## 2. Component Specifications

### 1. Generator Model (The Actor)
- **Runtime**: Ollama local server (`http://localhost:11434`).
- **Target Model**: `qwen2.5:3b` (or `llama3.2:3b`).
- **Hardware Footprint**: ~2.2 GB VRAM on NVIDIA GeForce RTX 4060 Laptop GPU. Runs concurrently with fine-tuned Laya (1.0 GB VRAM) with zero memory contention.
- **Role**: Immediate generation of deterministic IT operational actions (e.g., password/MFA resets, access grants, service restarts, DNS cache flushes, package installs).
- **Prompt Reference**: [`generator_prompt.md`](generator_prompt.md).

### 2. Critic Model (The Gatekeeper)
- **Runtime**: Groq Cloud LPU API (`https://api.groq.com/openai/v1`).
- **Target Model**: `openai/gpt-oss-120b`.
- **Latency Profile**: Sub-100 ms generation time via Groq hardware acceleration.
- **Output Constraint**: Strict JSON output containing a binary boolean flag:
  ```json
  {
    "valid": true,
    "reason": "Accurate, deterministic, and safe resolution provided."
  }
  ```
- **Evaluation Pillars**:
  1. **Intent Fulfillment**: Confirms the response directly solves the user's specific ticket without unnecessary back-and-forth.
  2. **Technical Correctness**: Validates that CLI commands, flag parameters, config syntax, and file paths are authentic and free of hallucinations.
  3. **Enterprise Safety & Policy**: Guarantees non-destructive execution and prevents credential exposure.
- **Prompt Reference**: [`critic_prompt.md`](critic_prompt.md).

---

## 3. Configuration & Security

- **Environment File**: `.env` placed at `src/itops_agent/slm_panel/.env`.
- **Security Policy**: `.env` and `*.env` explicitly ignored in the root [`.gitignore`](../../../.gitignore) to prevent API key leaks.
- **Required Variables**:
  ```ini
  GROQ_API_KEY=your_groq_api_key_here
  OLLAMA_BASE_URL=http://localhost:11434
  GENERATOR_MODEL=qwen2.5:3b
  CRITIC_MODEL=openai/gpt-oss-120b
  ```

---

## 4. Execution & Failure Policy

1. **Success Path (`valid == true`)**:
   - The verified resolution is returned directly to the end user.
   - Total expected round-trip latency: **~1.0 to 1.5 seconds**.

2. **Failure Path (`valid == false`)**:
   - The ticket triggers the **Autonomous Safety Circuit Breaker**.
   - The invalid draft is blocked from reaching the user.
   - The failure event, critic reason, and ticket metadata are logged for auditability and routed to human Tier-2 engineers or the multi-hop diagnostic loop.

---

## 5. Development Log & Milestones

- **2026-10-26**:
  - Initialized `src/itops_agent/slm_panel/` architecture.
  - Added `.env` security protections to root `.gitignore`.
  - Created `.env` configuration file.
  - Drafted production system prompt for Generator: [`generator_prompt.md`](generator_prompt.md).
  - Drafted production validation prompt for Critic: [`critic_prompt.md`](critic_prompt.md) enforcing strict binary JSON validation with `openai/gpt-oss-120b`.
  - Created standalone Laya annotation pipeline [`annotate_with_laya.py`](annotate_with_laya.py) to classify all 7,343 filtered IT support tickets on CUDA.
  - Retained authentic IT engineer resolutions (`answer` column) alongside `type` and `ticket` for downstream response-quality evaluation across LLM direct, SLM generator, and SLM Actor-Critic.
  - Generated `annotated_eval_dataset.csv` and isolated `single_hop_eval_dataset.csv` for targeted SLM Panel benchmarking.
  - Implemented self-contained execution module [`panel.py`](panel.py) and package interface [`__init__.py`](__init__.py).
    - Query-first input model: takes the raw ticket string as the sole required input, with `ticket_type` purely optional.
    - Actor stage: Ollama integration running `qwen2.5:3b` with system prompt from [`generator_prompt.md`](generator_prompt.md).
    - Gatekeeper stage: Groq Cloud LPU integration running `openai/gpt-oss-120b` with system prompt from [`critic_prompt.md`](critic_prompt.md).
    - Autonomous circuit breaker: returns verified resolution on pass (`valid: true`), halts execution immediately on failure (`valid: false`).
    - Comprehensive step-by-step logging and dual CLI / programmatic API (`SLMPanel`, `run_panel`).
    - Automated unit test suite added in [`tests/test_panel.py`](../../../tests/test_panel.py).



