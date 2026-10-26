"""
Unit tests for SLMPanel
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is in sys.path
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from itops_agent.slm_panel.panel import (
    PanelConfig,
    SLMPanel,
    run_panel,
)


class TestSLMPanel(unittest.TestCase):
    def setUp(self):
        self.panel = SLMPanel()

    def test_prompts_loaded(self):
        self.assertIn("Generator System Prompt", self.panel.generator_prompt)
        self.assertIn("Critic System Prompt", self.panel.critic_prompt)

    @patch("itops_agent.slm_panel.panel.requests.post")
    def test_successful_pass_flow(self, mock_post):
        # 1. Mock Generator (Ollama)
        mock_ollama_resp = MagicMock()
        mock_ollama_resp.json.return_value = {
            "message": {"content": "Run `sudo systemctl restart nginx` to restart the service."}
        }
        mock_ollama_resp.raise_for_status = MagicMock()

        # 2. Mock Critic (Groq)
        mock_groq_resp = MagicMock()
        mock_groq_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"valid": true, "reason": "Accurate and safe service restart."}'
                }
            }]
        }
        mock_groq_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_ollama_resp, mock_groq_resp]

        cfg = PanelConfig(groq_api_key="mock_key")
        result = run_panel("Restart the nginx web server", ticket_type="Request", config=cfg)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.final_output)
        self.assertIn("sudo systemctl restart nginx", result.final_output)
        self.assertIsNotNone(result.critic)
        self.assertTrue(result.critic.valid)
        self.assertEqual(result.critic.reason, "Accurate and safe service restart.")

    @patch("itops_agent.slm_panel.panel.requests.post")
    def test_rejection_fail_flow(self, mock_post):
        # 1. Mock Generator (Ollama) returns bad/destructive command
        mock_ollama_resp = MagicMock()
        mock_ollama_resp.json.return_value = {
            "message": {"content": "Run rm -rf /"}
        }
        mock_ollama_resp.raise_for_status = MagicMock()

        # 2. Mock Critic (Groq) rejects it
        mock_groq_resp = MagicMock()
        mock_groq_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"valid": false, "reason": "Destructive command violates safety policy."}'
                }
            }]
        }
        mock_groq_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_ollama_resp, mock_groq_resp]

        cfg = PanelConfig(groq_api_key="mock_key")
        result = run_panel("Free disk space immediately", ticket_type="Request", config=cfg)

        self.assertFalse(result.success)
        self.assertIsNone(result.final_output)
        self.assertIsNotNone(result.critic)
        self.assertFalse(result.critic.valid)
        self.assertEqual(result.critic.reason, "Destructive command violates safety policy.")

    @patch("itops_agent.slm_panel.panel.requests.post")
    def test_critic_markdown_fenced_json(self, mock_post):
        mock_ollama_resp = MagicMock()
        mock_ollama_resp.json.return_value = {
            "message": {"content": "Run `systemctl status sshd`"}
        }
        mock_ollama_resp.raise_for_status = MagicMock()

        mock_groq_resp = MagicMock()
        mock_groq_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": "```json\n{\n  \"valid\": true,\n  \"reason\": \"Safe status check.\"\n}\n```"
                }
            }]
        }
        mock_groq_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_ollama_resp, mock_groq_resp]

        cfg = PanelConfig(groq_api_key="mock_key")
        result = run_panel("Check ssh status", config=cfg)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.critic)
        self.assertTrue(result.critic.valid)
        self.assertEqual(result.critic.reason, "Safe status check.")

    @patch("itops_agent.slm_panel.panel.requests.post")
    def test_run_without_ticket_type(self, mock_post):
        # Verify query-only input works without ticket_type
        mock_ollama_resp = MagicMock()
        mock_ollama_resp.json.return_value = {
            "message": {"content": "Run `systemctl restart NetworkManager`"}
        }
        mock_ollama_resp.raise_for_status = MagicMock()

        mock_groq_resp = MagicMock()
        mock_groq_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"valid": true, "reason": "Standard network restart command."}'
                }
            }]
        }
        mock_groq_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_ollama_resp, mock_groq_resp]

        cfg = PanelConfig(groq_api_key="mock_key")
        result = run_panel("My wifi is not reconnecting", config=cfg)

        self.assertTrue(result.success)
        self.assertIsNone(result.ticket_type)
        self.assertEqual(result.final_output, "Run `systemctl restart NetworkManager`")

        # Check call args to verify formatted prompt does not have Type:
        gen_payload = mock_post.call_args_list[0][1]["json"]
        self.assertEqual(gen_payload["messages"][1]["content"], "Ticket: My wifi is not reconnecting")

    @patch("itops_agent.slm_panel.panel.requests.post")
    def test_generator_connection_failure(self, mock_post):
        mock_post.side_effect = ConnectionError("Connection refused by Ollama")

        result = run_panel("Check ssh status")
        self.assertFalse(result.success)
        self.assertIn("Generator Error", result.error or "")
        self.assertIsNone(result.final_output)


if __name__ == "__main__":
    unittest.main()

