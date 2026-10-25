* Tobias Bueck. (2025). Customer IT Support - Ticket Dataset [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/13959267
Here, renamed to "notes.md". The following changes are done to the original dataset in annotation.ipynb.
* Subject shall be merged into body because subject has 3838 missing values. 
* Based on manual inspection of first 5 queries of class of queue, we are narrowing it down to "Technical Support","IT Support", and "Service Outages and Maintenance".
* We are filtering out german (langauge = de) queries. 
* These columns are dropped for the following reasons:
"answer" : contains response to queries, we need raw queries
"queue" : only required ones remain, the class label is redundant
"priority" : irrelvant to understand the nature of query
"language" : only english ones remain
"version" : contains irrelavant metadata
"tag_1": contains irrelavant metadata
"tag_2" " contains irrelavant metadata
* Column "type" is retained because type can influence required hops.
* New column "complexity" is created for annotation. 
* Used llmfit via uvx.
* uvx llmfit recommend --use-case reasoning
{
  "models": [
    {
      "best_quant": null,
      "capabilities": [
        "Tool Use"
      ],
      "capability_ids": [
        "tool_use"
      ],
      "category": "Reasoning",
      "context_length": 262144,
      "disk_size_gb": 6.03,
      "effective_context_length": 8192,
      "estimate_basis": {
        "assumed_context": 8192,
        "ddr_bandwidth_gbps": null,
        "efficiency": 0.55,
        "gpu_bandwidth_gbps": null,
        "local_calibration": null,
        "method": "backend_constant"
      },
      "estimate_confidence": "estimated",
      "estimate_confidence_label": "estimated",
      "estimated_tps": 30.5,
      "fit_label": "Good",
      "fit_level": "Good",
      "gguf_sources": [],
      "installed": false,
      "is_moe": false,
      "license": "apache-2.0",
      "measured_tps": null,
      "memory_available_gb": 8.0,
      "memory_required_gb": 6.78,
      "moe_offloaded_gb": null,
      "name": "Alexzander85/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-NVFP4-MLP-FP8KV",
      "notes": [
        "Context capped at 8192 tokens for estimation (model supports up to 262144; use --max-context to override)",
        "GPU: model loaded into VRAM",
        "Best quantization for hardware: Q6_K (model default: Q4_K_M)",
        "Baseline estimated speed: 30.5 tok/s",
        "best_quant cleared: 'Alexzander85/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-NVFP4-MLP-FP8KV' is a native NVFP4/MXFP4 repo, so the GGUF quant label 'Q6_K' does not apply"
      ],
      "ollama_name": null,
      "parameter_count": "7.5B",
      "params_b": 7.54,
      "prefill_tps": null,
      "provider": "alexzander85",
      "release_date": "2026-03-04",
      "run_mode": "GPU",
      "run_mode_label": "GPU",
      "runtime": "llama.cpp",
      "runtime_label": "llama.cpp",
      "score": 90.8,
      "score_components": {
        "context": 100.0,
        "fit": 76.0,
        "quality": 89.8,
        "speed": 100.0
      },
      "supports_tp": [
        1,
        2,
        4
      ],
      "total_memory_gb": 6.78,
      "ttft_ms": null,
      "usable_context": 47996,
      "use_case": "Advanced reasoning, chain-of-thought",
      "utilization_pct": 84.8,
      "verify_command": "llama-bench -m <path-to-Q6_K-gguf> -ngl 99 -p 512 -n 128"
    },
    {
      "best_quant": null,
      "capabilities": [
        "Tool Use"
      ],
      "capability_ids": [
        "tool_use"
      ],
      "category": "Reasoning",
      "context_length": 131072,
      "disk_size_gb": 5.44,
      "effective_context_length": 8192,
      "estimate_basis": {
        "assumed_context": 8192,
        "ddr_bandwidth_gbps": null,
        "efficiency": 0.55,
        "gpu_bandwidth_gbps": null,
        "local_calibration": null,
        "method": "backend_constant"
      },
      "estimate_confidence": "estimated",
      "estimate_confidence_label": "estimated",
      "estimated_tps": 30.3,
      "fit_label": "Marginal",
      "fit_level": "Marginal",
      "gguf_sources": [],
      "installed": false,
      "is_moe": false,
      "license": "mit",
      "measured_tps": null,
      "memory_available_gb": 8.0,
      "memory_required_gb": 7.07,
      "moe_offloaded_gb": null,
      "name": "PreDoctor/DeepSeek-R1-0528-Qwen3-8B-NVFP4-W4A16-PreTensor",
      "notes": [
        "Context capped at 8192 tokens for estimation (model supports up to 131072; use --max-context to override)",
        "GPU: model loaded into VRAM",
        "Best quantization for hardware: Q5_K_M (model default: Q4_K_M)",
        "Baseline estimated speed: 30.3 tok/s",
        "best_quant cleared: 'PreDoctor/DeepSeek-R1-0528-Qwen3-8B-NVFP4-W4A16-PreTensor' is a native NVFP4/MXFP4 repo, so the GGUF quant label 'Q5_K_M' does not apply"
      ],
      "ollama_name": null,
      "parameter_count": "4.7B",
      "params_b": 8.0,
      "prefill_tps": null,
      "provider": "predoctor",
      "release_date": "2026-09-13",
      "run_mode": "GPU",
      "run_mode_label": "GPU",
      "runtime": "llama.cpp",
      "runtime_label": "llama.cpp",
      "score": 90.7,
      "score_components": {
        "context": 100.0,
        "fit": 65.6,
        "quality": 92.4,
        "speed": 100.0
      },
      "supports_tp": [
        1,
        2,
        4,
        8
      ],
      "total_memory_gb": 7.07,
      "ttft_ms": null,
      "usable_context": 14972,
      "use_case": "Advanced reasoning, chain-of-thought",
      "utilization_pct": 88.4,
      "verify_command": "llama-bench -m <path-to-Q5_K_M-gguf> -ngl 99 -p 512 -n 128"
    },
    {
      "best_quant": "Q5_K_M",
      "capabilities": [
        "Tool Use"
      ],
      "capability_ids": [
        "tool_use"
      ],
      "category": "Reasoning",
      "context_length": 262144,
      "disk_size_gb": 6.09,
      "effective_context_length": 8192,
      "estimate_basis": {
        "assumed_context": 8192,
        "ddr_bandwidth_gbps": null,
        "efficiency": 0.55,
        "gpu_bandwidth_gbps": null,
        "local_calibration": null,
        "method": "backend_constant"
      },
      "estimate_confidence": "estimated",
      "estimate_confidence_label": "estimated",
      "estimated_tps": 27.0,
      "fit_label": "Marginal",
      "fit_level": "Marginal",
      "gguf_sources": [],
      "installed": false,
      "is_moe": false,
      "license": "apache-2.0",
      "measured_tps": null,
      "memory_available_gb": 8.0,
      "memory_required_gb": 6.84,
      "moe_offloaded_gb": null,
      "name": "Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-4bit",
      "notes": [
        "Context capped at 8192 tokens for estimation (model supports up to 262144; use --max-context to override)",
        "GPU: model loaded into VRAM",
        "Best quantization for hardware: Q5_K_M (model default: Q4_K_M)",
        "Baseline estimated speed: 27.0 tok/s"
      ],
      "ollama_name": null,
      "parameter_count": "9.0B",
      "params_b": 8.95,
      "prefill_tps": null,
      "provider": "jackrong",
      "release_date": "2026-03-06",
      "run_mode": "GPU",
      "run_mode_label": "GPU",
      "runtime": "llama.cpp",
      "runtime_label": "llama.cpp",
      "score": 89.9,
      "score_components": {
        "context": 100.0,
        "fit": 74.0,
        "quality": 88.8,
        "speed": 100.0
      },
      "supports_tp": [
        1,
        2,
        4
      ],
      "total_memory_gb": 6.84,
      "ttft_ms": null,
      "usable_context": 46121,
      "use_case": "Advanced reasoning, chain-of-thought",
      "utilization_pct": 85.5,
      "verify_command": "llama-bench -m <path-to-Q5_K_M-gguf> -ngl 99 -p 512 -n 128"
    },
    {
      "best_quant": "Q5_K_M",
      "capabilities": [
        "Tool Use"
      ],
      "capability_ids": [
        "tool_use"
      ],
      "category": "Reasoning",
      "context_length": 262144,
      "disk_size_gb": 6.09,
      "effective_context_length": 8192,
      "estimate_basis": {
        "assumed_context": 8192,
        "ddr_bandwidth_gbps": null,
        "efficiency": 0.55,
        "gpu_bandwidth_gbps": null,
        "local_calibration": null,
        "method": "backend_constant"
      },
      "estimate_confidence": "estimated",
      "estimate_confidence_label": "estimated",
      "estimated_tps": 27.0,
      "fit_label": "Marginal",
      "fit_level": "Marginal",
      "gguf_sources": [],
      "installed": false,
      "is_moe": false,
      "license": "apache-2.0",
      "measured_tps": null,
      "memory_available_gb": 8.0,
      "memory_required_gb": 6.84,
      "moe_offloaded_gb": null,
      "name": "Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit",
      "notes": [
        "Context capped at 8192 tokens for estimation (model supports up to 262144; use --max-context to override)",
        "GPU: model loaded into VRAM",
        "Best quantization for hardware: Q5_K_M (model default: Q4_K_M)",
        "Baseline estimated speed: 27.0 tok/s"
      ],
      "ollama_name": null,
      "parameter_count": "9.0B",
      "params_b": 8.95,
      "prefill_tps": null,
      "provider": "jackrong",
      "release_date": "2026-03-07",
      "run_mode": "GPU",
      "run_mode_label": "GPU",
      "runtime": "llama.cpp",
      "runtime_label": "llama.cpp",
      "score": 89.9,
      "score_components": {
        "context": 100.0,
        "fit": 74.0,
        "quality": 88.8,
        "speed": 100.0
      },
      "supports_tp": [
        1,
        2,
        4
      ],
      "total_memory_gb": 6.84,
      "ttft_ms": null,
      "usable_context": 46121,
      "use_case": "Advanced reasoning, chain-of-thought",
      "utilization_pct": 85.5,
      "verify_command": "llama-bench -m <path-to-Q5_K_M-gguf> -ngl 99 -p 512 -n 128"
    },
    {
      "best_quant": "Q5_K_M",
      "capabilities": [
        "Tool Use"
      ],
      "capability_ids": [
        "tool_use"
      ],
      "category": "Reasoning",
      "context_length": 262144,
      "disk_size_gb": 6.4,
      "effective_context_length": 8192,
      "estimate_basis": {
        "assumed_context": 8192,
        "ddr_bandwidth_gbps": null,
        "efficiency": 0.55,
        "gpu_bandwidth_gbps": null,
        "local_calibration": null,
        "method": "backend_constant"
      },
      "estimate_confidence": "estimated",
      "estimate_confidence_label": "estimated",
      "estimated_tps": 25.7,
      "fit_label": "Marginal",
      "fit_level": "Marginal",
      "gguf_sources": [],
      "installed": false,
      "is_moe": false,
      "license": "apache-2.0",
      "measured_tps": null,
      "memory_available_gb": 8.0,
      "memory_required_gb": 7.15,
      "moe_offloaded_gb": null,
      "name": "reasonwang/SkillGym-Qwen3.5-9B",
      "notes": [
        "Context capped at 8192 tokens for estimation (model supports up to 262144; use --max-context to override)",
        "GPU: model loaded into VRAM",
        "Best quantization for hardware: Q5_K_M (model default: Q4_K_M)",
        "Baseline estimated speed: 25.7 tok/s"
      ],
      "ollama_name": null,
      "parameter_count": "9.4B",
      "params_b": 9.41,
      "prefill_tps": null,
      "provider": "reasonwang",
      "release_date": "2026-09-13",
      "run_mode": "GPU",
      "run_mode_label": "GPU",
      "runtime": "llama.cpp",
      "runtime_label": "llama.cpp",
      "score": 89.0,
      "score_components": {
        "context": 100.0,
        "fit": 62.5,
        "quality": 90.3,
        "speed": 100.0
      },
      "supports_tp": [
        1,
        2,
        4
      ],
      "total_memory_gb": 7.15,
      "ttft_ms": null,
      "usable_context": 35960,
      "use_case": "Advanced reasoning, chain-of-thought",
      "utilization_pct": 89.4,
      "verify_command": "llama-bench -m <path-to-Q5_K_M-gguf> -ngl 99 -p 512 -n 128"
    }
  ],
  "system": {
    "available_ram_gb": 5.99,
    "backend": "CUDA",
    "cpu_cores": 20,
    "cpu_name": "13th Gen Intel(R) Core(TM) i7-13700H",
    "gpu_available_gb": 7.61,
    "gpu_count": 1,
    "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "gpu_vram_gb": 8.0,
    "gpus": [
      {
        "backend": "CUDA",
        "count": 1,
        "free_vram_gb": 7.61,
        "memory_bandwidth_gbps": null,
        "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "unified_memory": false,
        "vram_gb": 8.0
      }
    ],
    "has_gpu": true,
    "total_ram_gb": 15.25,
    "unified_memory": false
  }
}
Finalized on deepseek-r1:8b via Ollama

* token_counter.ipynb shall be used to count the tokens. 