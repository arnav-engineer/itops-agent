"""
SLM Panel Package
Actor-Critic single-hop resolution system for IT operations.
"""

from itops_agent.slm_panel.panel import (
    CriticEvaluation,
    PanelConfig,
    PanelResponse,
    SLMPanel,
    run_panel,
)

__all__ = [
    "CriticEvaluation",
    "PanelConfig",
    "PanelResponse",
    "SLMPanel",
    "run_panel",
]

