"""Swargaseema AI Sandalwood Farms inbound receptionist project package.

This package provides the foundation layer for Swargaseema AI, reusing AI Core
and exposing provider-agnostic bridge and assistant components.
"""

from projects.swargaseema_ai.assistant import SwargaseemaAssistant
from projects.swargaseema_ai.bridge import SwargaseemaBridge
from projects.swargaseema_ai.config import SwargaseemaSettings
from projects.swargaseema_ai.prompts import (
    SWARGASEEMA_SYSTEM_TEMPLATE_ID,
    SWARGASEEMA_SYSTEM_TEMPLATE_PATH,
    build_system_prompt,
    get_system_prompt_template_text,
    get_time_based_greeting,
    load_swargaseema_system_template,
)

__all__ = [
    "SwargaseemaAssistant",
    "SwargaseemaBridge",
    "SwargaseemaSettings",
    "SWARGASEEMA_SYSTEM_TEMPLATE_ID",
    "SWARGASEEMA_SYSTEM_TEMPLATE_PATH",
    "build_system_prompt",
    "get_system_prompt_template_text",
    "get_time_based_greeting",
    "load_swargaseema_system_template",
]
