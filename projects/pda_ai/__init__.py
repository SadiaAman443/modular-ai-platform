"""PDA Engineering College AI Student Support Assistant project package.

This package integrates the legacy PDA AI student support business logic and
Twilio audio framing with the domain-agnostic `ai_core` foundation.
"""

from projects.pda_ai.assistant import PDAAssistant
from projects.pda_ai.bridge import PDAGeminiServiceBridge, PDATwilioBridge
from projects.pda_ai.prompts import (
    PDA_SYSTEM_TEMPLATE_PATH,
    build_system_prompt,
    get_time_based_greeting,
    load_pda_system_template,
)

__all__ = [
    "PDAAssistant",
    "PDAGeminiServiceBridge",
    "PDATwilioBridge",
    "PDA_SYSTEM_TEMPLATE_PATH",
    "build_system_prompt",
    "get_time_based_greeting",
    "load_pda_system_template",
]
