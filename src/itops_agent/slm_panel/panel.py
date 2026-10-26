"""
SLM Panel: Actor-Critic Single-Hop Resolution System

This module provides the autonomous execution engine for deterministic,
single-hop IT support tickets.

Architecture:
    1. Generator (Actor): Local Ollama running `qwen2.5:3b` with `generator_prompt.md`.
    2. Critic (Gatekeeper): Cloud Groq LPU running `openai/gpt-oss-120b` with `critic_prompt.md`.
    3. Gatekeeper Decision:
       - If pass (valid=True): Resolution returned to the user.
       - If fail (valid=False): Execution halts immediately (autonomous circuit breaker).

Usage:
    # Programmatic:
    from itops_agent.slm_panel.panel import SLMPanel, run_panel
    result = run_panel("How do I flush the local DNS cache on Ubuntu 22.04?")
    if result.success:
        print(result.final_output)

    # CLI:
    python -m itops_agent.slm_panel.panel "How do I flush DNS cache on Linux?"
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
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("itops_agent.slm_panel")


# -----------------------------------------------------------------------------
# Environment & Configuration Loading
# -----------------------------------------------------------------------------

def load_dotenv(dotenv_path: Path) -> Dict[str, str]:
    """Parse a .env file into key-value pairs without external dependencies."""
    loaded = {}
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
                    # Only populate if not already set in process environment
                    if key not in os.environ:
                        os.environ[key] = val
    except Exception as e:
        logger.warning(f"Could not load .env from {dotenv_path}: {e}")
    return loaded


# Automatically load environment from default locations at import time
_current_dir = Path(__file__).resolve().parent
load_dotenv(_current_dir / ".env")
load_dotenv(Path.cwd() / ".env")


@dataclass
class PanelConfig:
    """Runtime configuration for the Actor-Critic SLM Panel."""

    # Generator (Ollama)
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    generator_model: str = field(
        default_factory=lambda: os.getenv("GENERATOR_MODEL", "qwen2.5:3b")
    )
    generator_temperature: float = 0.2
    generator_timeout: float = 60.0

    # Critic (Groq)
    groq_base_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )
    critic_model: str = field(
        default_factory=lambda: os.getenv("CRITIC_MODEL", "openai/gpt-oss-120b")
    )
    critic_temperature: float = 0.0
    critic_timeout: float = 30.0

    # Prompt paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    generator_prompt_file: str = "generator_prompt.md"
    critic_prompt_file: str = "critic_prompt.md"

    def __post_init__(self):
        # Fallback if environment was loaded or modified dynamically
        if self.groq_api_key is None:
            self.groq_api_key = os.getenv("GROQ_API_KEY")

    def get_generator_prompt_path(self) -> Path:
        return self.base_dir / self.generator_prompt_file

    def get_critic_prompt_path(self) -> Path:
        return self.base_dir / self.critic_prompt_file


# -----------------------------------------------------------------------------
# Result Data Structures
# -----------------------------------------------------------------------------

@dataclass
class CriticEvaluation:
    """Structured decision returned by the Critic (Gatekeeper)."""

    valid: bool
    reason: str
    raw_response: str
    latency_ms: float


@dataclass
class PanelResponse:
    """End-to-end outcome of an SLM Panel processing request."""

    success: bool
    query: str
    ticket_type: Optional[str] = None
    draft_resolution: Optional[str] = None
    critic: Optional[CriticEvaluation] = None
    final_output: Optional[str] = None
    total_latency_ms: float = 0.0
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# SLM Panel Implementation
# -----------------------------------------------------------------------------

class SLMPanel:
    """Self-contained Actor-Critic execution panel for single-hop IT tickets."""

    def __init__(self, config: Optional[PanelConfig] = None):
        # Automatically load module-local .env if present
        current_dir = Path(__file__).resolve().parent
        load_dotenv(current_dir / ".env")

        self.config = config or PanelConfig()
        self._generator_prompt: Optional[str] = None
        self._critic_prompt: Optional[str] = None

    def _load_prompt(self, file_path: Path, prompt_name: str) -> str:
        """Load and cache prompt file from disk."""
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Missing {prompt_name} system prompt at: {file_path}"
            )
        return file_path.read_text(encoding="utf-8").strip()

    @property
    def generator_prompt(self) -> str:
        if self._generator_prompt is None:
            path = self.config.get_generator_prompt_path()
            self._generator_prompt = self._load_prompt(path, "Generator")
            logger.debug(f"Loaded generator prompt from {path} ({len(self._generator_prompt)} chars)")
        return self._generator_prompt

    @property
    def critic_prompt(self) -> str:
        if self._critic_prompt is None:
            path = self.config.get_critic_prompt_path()
            self._critic_prompt = self._load_prompt(path, "Critic")
            logger.debug(f"Loaded critic prompt from {path} ({len(self._critic_prompt)} chars)")
        return self._critic_prompt

    def generate(self, query: str, ticket_type: Optional[str] = None) -> Tuple[str, float]:
        """Invoke the Generator model (Actor) via Ollama.

        Returns:
            Tuple[str, float]: (draft_resolution_text, latency_ms)
        """
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"
        if ticket_type:
            formatted_user_prompt = f"Type: {ticket_type}\nTicket: {query.strip()}"
        else:
            formatted_user_prompt = f"Ticket: {query.strip()}"

        payload = {
            "model": self.config.generator_model,
            "messages": [
                {"role": "system", "content": self.generator_prompt},
                {"role": "user", "content": formatted_user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.config.generator_temperature,
            },
        }

        logger.info(
            f"[Actor: Generator] Dispatching query to Ollama ({self.config.generator_model}) at {url}"
        )
        logger.debug(f"[Actor: Generator] Formatted input:\n{formatted_user_prompt}")

        start_time = time.perf_counter()
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.generator_timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError as e:
            msg = (
                f"Failed to connect to Ollama at {url}. "
                "Ensure the Ollama service is active (`ollama serve`)."
            )
            logger.error(f"[Actor: Generator] Connection error: {msg} (Details: {e})")
            raise ConnectionError(msg) from e
        except requests.exceptions.RequestException as e:
            logger.error(f"[Actor: Generator] HTTP request failed: {e}")
            raise RuntimeError(f"Ollama request error: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError(f"Ollama returned empty response payload: {data}")

        logger.info(
            f"[Actor: Generator] Draft resolution generated in {latency_ms:.1f} ms "
            f"({len(content)} chars)"
        )
        logger.debug(f"[Actor: Generator] Draft Resolution:\n{content}")
        return content, latency_ms

    def critique(
        self, query: str, draft: str, ticket_type: Optional[str] = None
    ) -> CriticEvaluation:
        """Invoke the Critic model (Gatekeeper) via Groq Cloud API.

        Returns:
            CriticEvaluation: Parsed binary decision and diagnostic reason.
        """
        api_key = self.config.groq_api_key
        if not api_key or api_key.strip() == "":
            raise ValueError(
                "Missing GROQ_API_KEY. Please provide a valid Groq API key in your "
                ".env file or via the GROQ_API_KEY environment variable."
            )

        url = f"{self.config.groq_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if ticket_type:
            critic_user_content = (
                f"Ticket Type: {ticket_type}\n"
                f"Ticket Text: {query.strip()}\n\n"
                f"Draft Response:\n{draft.strip()}"
            )
        else:
            critic_user_content = (
                f"Ticket: {query.strip()}\n\n"
                f"Draft Response:\n{draft.strip()}"
            )

        payload = {
            "model": self.config.critic_model,
            "messages": [
                {"role": "system", "content": self.critic_prompt},
                {"role": "user", "content": critic_user_content},
            ],
            "temperature": self.config.critic_temperature,
            "response_format": {"type": "json_object"},
        }

        logger.info(
            f"[Gatekeeper: Critic] Dispatching evaluation to Groq ({self.config.critic_model}) at {url}"
        )
        logger.debug(f"[Gatekeeper: Critic] Formatted input:\n{critic_user_content}")

        start_time = time.perf_counter()
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.critic_timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"[Gatekeeper: Critic] HTTP request failed: {e}")
            raise RuntimeError(f"Groq API request error: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        raw_content = ""
        try:
            raw_content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise ValueError(f"Malformed response structure from Groq API: {data}") from e

        # Strip markdown code fences if model enclosed JSON in ```json ... ```
        cleaned_content = raw_content
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
        if match:
            cleaned_content = match.group(1).strip()

        try:
            parsed = json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            logger.error(f"[Gatekeeper: Critic] Failed to parse JSON: '{raw_content}'")
            raise ValueError(f"Critic did not return valid JSON: {raw_content}") from e

        is_valid = bool(parsed.get("valid", False))
        reason = str(parsed.get("reason", "No reason provided by Critic.")).strip()

        logger.info(
            f"[Gatekeeper: Critic] Evaluation finished in {latency_ms:.1f} ms -> "
            f"Decision: {'[PASS]' if is_valid else '[FAIL]'} | Reason: {reason}"
        )

        return CriticEvaluation(
            valid=is_valid,
            reason=reason,
            raw_response=raw_content,
            latency_ms=latency_ms,
        )

    def process(self, query: str, ticket_type: Optional[str] = None) -> PanelResponse:
        """Process a single-hop IT support query through the Actor-Critic pipeline.

        Execution flow:
            1. Receive single-hop query.
            2. Pass query + generator prompt to Generator (`qwen2.5:3b` via Ollama).
            3. Pass query + draft response + critic prompt to Critic (`openai/gpt-oss-120b` via Groq).
            4. Binary gatekeeper:
               - If pass (valid == True): Return resolution directly to user.
               - If fail (valid == False): Halt execution immediately (stop there).

        Args:
            query: The single-hop ticket text.
            ticket_type: Optional ticket classification (Incident, Request, Problem, Change).

        Returns:
            PanelResponse with detailed step-by-step metadata and decision outcome.
        """
        overall_start = time.perf_counter()
        logger.info("=" * 70)
        type_str = f" [Type: {ticket_type}]" if ticket_type else ""
        logger.info(f"[SLM Panel Gateway] New Single-Hop Query received{type_str}")
        logger.info(f"Query: {query.strip()}")
        logger.info("=" * 70)

        # ---------------------------------------------------------------------
        # Step 1: Generator Model (Actor)
        # ---------------------------------------------------------------------
        draft_resolution: Optional[str] = None
        try:
            draft_resolution, gen_latency = self.generate(
                query=query, ticket_type=ticket_type
            )
        except Exception as e:
            total_time = (time.perf_counter() - overall_start) * 1000.0
            logger.error(f"[SLM Panel] Generation stage aborted due to error: {e}")
            return PanelResponse(
                success=False,
                query=query,
                ticket_type=ticket_type,
                error=f"Generator Error: {e}",
                total_latency_ms=total_time,
            )

        # ---------------------------------------------------------------------
        # Step 2: Critic Model (Gatekeeper)
        # ---------------------------------------------------------------------
        critic_result: Optional[CriticEvaluation] = None
        try:
            critic_result = self.critique(
                query=query, draft=draft_resolution, ticket_type=ticket_type
            )
        except Exception as e:
            total_time = (time.perf_counter() - overall_start) * 1000.0
            logger.error(f"[SLM Panel] Gatekeeper critique aborted due to error: {e}")
            return PanelResponse(
                success=False,
                query=query,
                ticket_type=ticket_type,
                draft_resolution=draft_resolution,
                error=f"Critic Error: {e}",
                total_latency_ms=total_time,
            )

        total_time = (time.perf_counter() - overall_start) * 1000.0

        # ---------------------------------------------------------------------
        # Step 3: Decision Routing
        # ---------------------------------------------------------------------
        if critic_result.valid:
            logger.info("=" * 70)
            logger.info(
                f"[SLM Panel Decision: PASS] Verified by Critic in {critic_result.latency_ms:.1f} ms. "
                "Returning verified resolution to user."
            )
            logger.info(f"Total pipeline latency: {total_time:.1f} ms")
            logger.info("=" * 70)

            return PanelResponse(
                success=True,
                query=query,
                ticket_type=ticket_type,
                draft_resolution=draft_resolution,
                critic=critic_result,
                final_output=draft_resolution,
                total_latency_ms=total_time,
            )
        else:
            logger.warning("=" * 70)
            logger.warning(
                f"[SLM Panel Decision: FAIL] Critic rejected draft resolution! "
                f"Reason: {critic_result.reason}"
            )
            logger.warning(
                "[Autonomous Safety Circuit Breaker] Execution halted. Response withheld from user."
            )
            logger.warning(f"Total pipeline latency: {total_time:.1f} ms")
            logger.warning("=" * 70)

            return PanelResponse(
                success=False,
                query=query,
                ticket_type=ticket_type,
                draft_resolution=draft_resolution,
                critic=critic_result,
                final_output=None,
                total_latency_ms=total_time,
            )


# -----------------------------------------------------------------------------
# Convenience Functions & CLI
# -----------------------------------------------------------------------------

def run_panel(
    query: str,
    ticket_type: Optional[str] = None,
    config: Optional[PanelConfig] = None,
) -> PanelResponse:
    """Convenience wrapper to run a single query through the SLM Panel."""
    panel = SLMPanel(config=config)
    return panel.process(query=query, ticket_type=ticket_type)


def configure_logging(verbose: bool = False) -> None:
    """Configure standardized logging output for the SLM Panel."""
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("itops_agent")
    root_logger.setLevel(level)

    # Avoid duplicate handlers if invoked multiple times
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    else:
        root_logger.handlers.clear()
        root_logger.addHandler(handler)


def main() -> None:
    """Command-line interface for testing and interacting with panel.py."""
    parser = argparse.ArgumentParser(
        description="SLM Panel: Actor-Critic Single-Hop Resolution System"
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        default=None,
        help="The single-hop IT support query to process.",
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
        "--type",
        type=str,
        default=None,
        choices=["Incident", "Request", "Problem", "Change"],
        help="Optional IT ticket classification type (e.g. Incident, Request).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Base URL of the Ollama server (overrides OLLAMA_BASE_URL).",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default=None,
        help="Generator model name in Ollama (overrides GENERATOR_MODEL).",
    )
    parser.add_argument(
        "--groq-key",
        type=str,
        default=None,
        help="Groq Cloud API key (overrides GROQ_API_KEY).",
    )
    parser.add_argument(
        "--critic-model",
        type=str,
        default=None,
        help="Critic model name in Groq (overrides CRITIC_MODEL).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )

    args = parser.parse_args()

    # Determine query string
    ticket_query = args.query or args.query_flag
    if not ticket_query:
        # Check standard input if piped
        if not sys.stdin.isatty():
            ticket_query = sys.stdin.read().strip()

    if not ticket_query:
        parser.error(
            "A query is required. Provide it as an argument or via --ticket / stdin."
        )

    configure_logging(verbose=args.verbose)

    # Build config overrides
    config = PanelConfig()
    if args.ollama_url:
        config.ollama_base_url = args.ollama_url
    if args.generator_model:
        config.generator_model = args.generator_model
    if args.groq_key:
        config.groq_api_key = args.groq_key
    if args.critic_model:
        config.critic_model = args.critic_model

    result = run_panel(query=ticket_query, ticket_type=args.type, config=config)

    print("\n" + "=" * 50)
    print("SLM PANEL EXECUTION SUMMARY")
    print("=" * 50)
    print(f"Status:            {'SUCCESS (Passed Critic)' if result.success else 'FAILED / HALTED'}")
    print(f"Total Latency:     {result.total_latency_ms:.1f} ms")
    if result.critic:
        print(f"Critic Gatekeeper: {'PASS' if result.critic.valid else 'FAIL'}")
        print(f"Critic Reason:     {result.critic.reason}")
    if result.error:
        print(f"Error Encountered: {result.error}")

    if result.success and result.final_output:
        print("\n" + "-" * 50)
        print("FINAL USER-FACING RESOLUTION:")
        print("-" * 50)
        print(result.final_output)
        sys.exit(0)
    else:
        print("\n[Circuit Breaker Active] Draft resolution withheld.")
        sys.exit(1)


if __name__ == "__main__":
    main()

