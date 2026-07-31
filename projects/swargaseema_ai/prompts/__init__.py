"""Prompts module for Swargaseema AI Sandalwood Farms inbound receptionist."""

from projects.swargaseema_ai.prompts.loader import (
    SWARGASEEMA_SYSTEM_TEMPLATE_ID,
    SWARGASEEMA_SYSTEM_TEMPLATE_PATH,
    load_swargaseema_system_template,
)
from projects.swargaseema_ai.prompts.system_prompt import (
    build_system_prompt,
    get_system_prompt_template_text,
    get_time_based_greeting,
)

__all__ = [
    "SWARGASEEMA_SYSTEM_TEMPLATE_ID",
    "SWARGASEEMA_SYSTEM_TEMPLATE_PATH",
    "build_system_prompt",
    "get_system_prompt_template_text",
    "get_time_based_greeting",
    "load_swargaseema_system_template",
]
