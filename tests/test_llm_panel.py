"""
Unit tests for Multi-Hop LLM Panel
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from itops_agent.llm_panel.panel import (
    LLMConfig,
    LLMPanel,
    process_multi_hop,
)


class TestLLMPanel(unittest.TestCase):
    def setUp(self):
        self.panel = LLMPanel()

    def test_system_prompt_loaded(self):
        self.assertIn("Tier-2 IT Operations", self.panel.system_prompt)
        self.assertIn("Multi-Hop Sequential Diagnostic Investigation", self.panel.system_prompt)

    @patch("itops_agent.llm_panel.panel.requests.post")
    def test_successful_multi_hop_resolution(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Step 1: Check nginx error logs.\nStep 2: Inspect upstream application pool.",
                    "reasoning": "Analyze 502 error causes."
                }
            }],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 85,
                "total_tokens": 205
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = LLMConfig(openrouter_api_key="mock_key")
        result = process_multi_hop("Intermittent 502 Bad Gateway errors on web application", config=cfg)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.response)
        self.assertIn("Check nginx error logs", result.response)
        self.assertEqual(result.total_tokens, 205)
        self.assertEqual(result.model, "nvidia/nemotron-3-ultra-550b-a55b:free")

    def test_missing_api_key(self):
        cfg = LLMConfig(openrouter_api_key="")
        result = process_multi_hop("Database deadlock issue", config=cfg)

        self.assertFalse(result.success)
        self.assertIn("Missing OPENROUTER_API_KEY", result.error or "")
        self.assertIsNone(result.response)

    @patch("itops_agent.llm_panel.panel.requests.post")
    def test_request_timeout(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        cfg = LLMConfig(openrouter_api_key="mock_key")
        result = process_multi_hop("Investigate Kubernetes crashloop", config=cfg)

        self.assertFalse(result.success)
        self.assertIn("timed out", result.error or "")
        self.assertIsNone(result.response)


if __name__ == "__main__":
    unittest.main()

