"""Prompt generation layer for the PDA Engineering College AI project.

This package extracts legacy prompt construction from `old-pda-backend` and
adapts it to the `ai_core.prompts` foundation.
"""

from projects.pda_ai.prompts.loader import load_pda_system_template
from projects.pda_ai.prompts.system_prompt import (
    build_system_prompt,
    get_system_prompt_template_text,
    get_time_based_greeting,
)
from projects.pda_ai.prompts.templates import PDA_SYSTEM_TEMPLATE_PATH

__all__ = [
    "PDA_SYSTEM_TEMPLATE_PATH",
    "build_system_prompt",
    "get_system_prompt_template_text",
    "get_time_based_greeting",
    "load_pda_system_template",
]
