"""
LLM Panel: Multi-Hop Diagnostic Resolution Engine

This module provides the execution engine for complex, multi-hop IT support tickets
requiring sequential diagnostic investigation, root cause analysis (RCA), and
remediation planning.

Architecture:
    - Tier-2 Expert Diagnostic Model: NVIDIA Nemotron 3 Ultra (nvidia/nemotron-3-ultra-550b-a55b:free)
    - Runtime / Gateway: OpenRouter API (`https://openrouter.ai/api/v1/chat/completions`)
    - System Prompt: `system_prompt.md` specializing in multi-hop ITOps triage

Usage:
    # Programmatic (function-based):
    from itops_agent.llm_panel import process_multi_hop
    result = process_multi_hop("Users report intermittent 502 Bad Gateway errors on web app")
    if result.success:
        print(result.response)

    # Programmatic (class-based):
    from itops_agent.llm_panel import LLMPanel
    panel = LLMPanel()
    result = panel.process("Database connection pool exhausted during peak hours")

    # CLI:
    uv run python src/itops_agent/llm_panel/panel.py "Intermittent 502 Bad Gateway errors"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("itops_agent.llm_panel")


# -----------------------------------------------------------------------------
# Environment & Configuration Loading
# -----------------------------------------------------------------------------

def load_dotenv(dotenv_path: Path) -> Dict[str, str]:
    """Parse a .env file into key-value pairs without external dependencies."""
    loaded: Dict[str, str] = {}
    if not dotenv_path.is_file():
        return loaded

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("\"'")
                    loaded[key] = val
                    if key not in os.environ:
                        os.environ[key] = val
    except Exception as e:
        logger.warning(f"Could not load .env from {dotenv_path}: {e}")
    return loaded


# Automatically discover and load .env files across likely locations
_module_dir = Path(__file__).resolve().parent
load_dotenv(_module_dir / ".env")
load_dotenv(_module_dir.parent / "slm_panel" / ".env")
load_dotenv(Path.cwd() / ".env")


@dataclass
class LLMConfig:
    """Runtime configuration for the Multi-Hop LLM Panel."""

    openrouter_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    )
    temperature: float = 0.2
    max_tokens: int = 1500
    timeout: float = 90.0

    # System prompt path
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    system_prompt_file: str = "system_prompt.md"

    def __post_init__(self):
        if self.openrouter_api_key is None:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    def get_system_prompt_path(self) -> Path:
        return self.base_dir / self.system_prompt_file


# -----------------------------------------------------------------------------
# Result Data Structures
# -----------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured response returned by the LLM Panel."""

    success: bool
    query: str
    response: Optional[str] = None
    reasoning: Optional[str] = None
    model: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# LLM Panel Implementation
# -----------------------------------------------------------------------------

class LLMPanel:
    """Self-contained Multi-Hop diagnostic panel powered by Nemotron via OpenRouter."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._system_prompt: Optional[str] = None

    def _load_system_prompt(self) -> str:
        """Load and cache the system prompt from disk."""
        path = self.config.get_system_prompt_path()
        if not path.is_file():
            raise FileNotFoundError(f"Missing LLM system prompt at: {path}")
        return path.read_text(encoding="utf-8").strip()

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self._load_system_prompt()
            logger.debug(
                f"Loaded LLM diagnostic prompt from {self.config.get_system_prompt_path()} "
                f"({len(self._system_prompt)} chars)"
            )
        return self._system_prompt

    def process(self, query: str) -> LLMResponse:
        """Process a multi-hop IT support query using the diagnostic LLM.

        Args:
            query: The multi-hop ticket query text describing the failure or issue.

        Returns:
            LLMResponse containing the diagnostic workflow, latency, and token metrics.
        """
        start_time = time.perf_counter()
        logger.info("=" * 70)
        logger.info("[LLM Panel Gateway] New Multi-Hop Query received")
        logger.info(f"Query: {query.strip()}")
        logger.info("=" * 70)

        api_key = self.config.openrouter_api_key
        if not api_key or not api_key.strip():
            error_msg = (
                "Missing OPENROUTER_API_KEY. Please provide an OpenRouter API key "
                "in your .env file or via the OPENROUTER_API_KEY environment variable."
            )
            logger.error(f"[LLM Panel] {error_msg}")
            return LLMResponse(
                success=False,
                query=query,
                model=self.config.model,
                error=error_msg,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        url = f"{self.config.openrouter_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/itops_agent",
            "X-Title": "ITOps Autonomous Agent",
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Ticket: {query.strip()}\n\n"
                        "Provide a structured multi-hop diagnostic and remediation plan."
                    ),
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        logger.info(
            f"[LLM Diagnostic Engine] Dispatching request to OpenRouter ({self.config.model}) at {url}"
        )

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout as e:
            total_time = (time.perf_counter() - start_time) * 1000.0
            error_msg = f"OpenRouter request timed out after {self.config.timeout}s: {e}"
            logger.error(f"[LLM Panel] {error_msg}")
            return LLMResponse(
                success=False,
                query=query,
                model=self.config.model,
                error=error_msg,
                latency_ms=total_time,
            )
        except requests.exceptions.RequestException as e:
            total_time = (time.perf_counter() - start_time) * 1000.0
            error_msg = f"OpenRouter HTTP request failed: {e}"
            logger.error(f"[LLM Panel] {error_msg}")
            return LLMResponse(
                success=False,
                query=query,
                model=self.config.model,
                error=error_msg,
                latency_ms=total_time,
            )

        total_time = (time.perf_counter() - start_time) * 1000.0

        # Extract message, content, reasoning, and usage
        try:
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            reasoning = message.get("reasoning")
        except (KeyError, IndexError) as e:
            error_msg = f"Malformed response payload from OpenRouter: {data}"
            logger.error(f"[LLM Panel] {error_msg}")
            return LLMResponse(
                success=False,
                query=query,
                model=self.config.model,
                error=error_msg,
                latency_ms=total_time,
            )

        # Fallback if content was left empty by a reasoning-first model
        final_answer = content.strip() if content else (reasoning.strip() if reasoning else "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        logger.info(
            f"[LLM Diagnostic Engine] Resolution generated in {total_time:.1f} ms | "
            f"Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})"
        )
        logger.info("=" * 70)
        logger.info("[LLM Panel Decision: SUCCESS] Diagnostic plan ready for delivery.")
        logger.info("=" * 70)

        return LLMResponse(
            success=True,
            query=query,
            response=final_answer,
            reasoning=reasoning,
            model=data.get("model", self.config.model),
            latency_ms=total_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


# -----------------------------------------------------------------------------
# Convenience Functions & CLI
# -----------------------------------------------------------------------------

def process_multi_hop(
    query: str,
    config: Optional[LLMConfig] = None,
) -> LLMResponse:
    """Single LLM function taking a multi-hop query and returning diagnostic output."""
    panel = LLMPanel(config=config)
    return panel.process(query=query)


# Alias for consistency with SLM panel
run_llm_panel = process_multi_hop


def configure_logging(verbose: bool = False) -> None:
    """Configure standardized logging output for the LLM Panel."""
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("itops_agent")
    root_logger.setLevel(level)

    if not root_logger.handlers:
        root_logger.addHandler(handler)
    else:
        root_logger.handlers.clear()
        root_logger.addHandler(handler)


def main() -> None:
    """Command-line interface for the Multi-Hop LLM Panel."""
    parser = argparse.ArgumentParser(
        description="LLM Panel: Multi-Hop ITOps Diagnostic Resolution Engine"
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        default=None,
        help="The multi-hop IT support query to process.",
    )
    parser.add_argument(
        "--ticket",
        "-t",
        type=str,
        dest="query_flag",
        default=None,
        help="Alternative flag to provide the query text.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Target OpenRouter model name (overrides LLM_MODEL).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (overrides OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="Maximum generation tokens (default: 1500).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )

    args = parser.parse_args()

    ticket_query = args.query or args.query_flag
    if not ticket_query:
        if not sys.stdin.isatty():
            ticket_query = sys.stdin.read().strip()

    if not ticket_query:
        parser.error("A query is required. Provide it as an argument or via --ticket / stdin.")

    configure_logging(verbose=args.verbose)

    config = LLMConfig()
    if args.model:
        config.model = args.model
    if args.api_key:
        config.openrouter_api_key = args.api_key
    if args.max_tokens:
        config.max_tokens = args.max_tokens

    result = process_multi_hop(query=ticket_query, config=config)

    print("\n" + "=" * 50)
    print("LLM PANEL EXECUTION SUMMARY")
    print("=" * 50)
    print(f"Status:            {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Model:             {result.model}")
    print(f"Total Latency:     {result.latency_ms:.1f} ms")
    print(f"Total Tokens:      {result.total_tokens} (prompt: {result.prompt_tokens}, completion: {result.completion_tokens})")
    if result.error:
        print(f"Error Encountered: {result.error}")

    if result.success and result.response:
        print("\n" + "-" * 50)
        print("DIAGNOSTIC & REMEDIATION RESOLUTION:")
        print("-" * 50)
        print(result.response)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

