# LLM Panel: Multi-Hop Diagnostic Resolution Engine

## 1. Architectural Paradigm & System Overview

The **LLM Panel** serves as the Tier-2 expert diagnostic engine for complex, multi-hop IT operational incidents and troubleshooting workflows routed from the Tier-0 Laya classifier (`multi_hop` route).

### Execution Flow
```
                           [ User Ticket ]
                                  │
                                  ▼
                     [ Tier-0 Laya Classifier ]
                                  │
                         (routes: multi_hop)
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │       LLM PANEL GATEWAY         │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │  Tier-2 Diagnostic LLM          │
                 │  - Gateway: OpenRouter API      │
                 │  - Model: Nemotron 3 Ultra      │
                 │    (nvidia/nemotron-3-ultra-    │
                 │     550b-a55b:free)             │
                 │  - Role: Multi-hop reasoning,   │
                 │    sequential RCA, remediation  │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                    [ Diagnostic Resolution Plan ]
                                  │
                                  ▼
                          [ Return to User ]
```

---

## 2. Component Specifications

### Model: NVIDIA Nemotron 3 Ultra (550B MoE)
- **Runtime**: OpenRouter API (`https://openrouter.ai/api/v1/chat/completions`).
- **Target Model Identifier**: `nvidia/nemotron-3-ultra-550b-a55b:free`.
- **Architecture**: 550B total parameter Mixture-of-Experts (MoE) reasoning architecture.
- **Role**: Sequential diagnostic investigation, dependency isolation, hypothesis generation, and step-by-step remediation plans for complex IT failures (e.g. intermittent 502s, database deadlocks, network timeouts, Kubernetes crash loops).
- **Prompt Reference**: [`system_prompt.md`](system_prompt.md).

---

## 3. Configuration & Security

- **Environment Keys**:
  ```ini
  OPENROUTER_API_KEY=your_openrouter_api_key_here
  OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
  LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
  ```
- **Security Policy**: `.env` and `*.env` files explicitly ignored in the root [`.gitignore`](../../../.gitignore).
- A clean template is tracked at [`.env.example`](.env.example).

---

## 4. Usage

### Programmatic API
```python
from itops_agent.llm_panel import process_multi_hop

result = process_multi_hop(
    "Users report intermittent 502 Bad Gateway errors on our web application. "
    "Nginx error logs show 'upstream timed out' under peak traffic."
)

if result.success:
    print(result.response)
    print(f"Latency: {result.latency_ms:.1f} ms | Tokens: {result.total_tokens}")
```

### CLI
```bash
uv run python src/itops_agent/llm_panel/panel.py \
  "Users report intermittent 502 Bad Gateway errors" -v
```

