"""System prompt construction and rendering service for Swargaseema AI.

This module provides helper functions to render Swargaseema Sandalwood Farms
inbound receptionist prompts using the domain-agnostic `ai_core.prompts` engine.
"""

from datetime import datetime
from typing import Any, Optional

from ai_core.prompts import PromptEngine
from projects.swargaseema_ai.prompts.loader import (
    SWARGASEEMA_SYSTEM_TEMPLATE_ID,
    load_swargaseema_system_template,
)


def get_time_based_greeting(current_hour: Optional[int] = None) -> str:
    """Returns a time-of-day appropriate greeting string.

    Args:
        current_hour: Optional hour override (0-23) for testing.

    Returns:
        A greeting such as 'Good Morning', 'Good Afternoon', or 'Good Evening'.
    """
    hour = current_hour if current_hour is not None else datetime.now().hour
    if hour < 12:
        return "Good Morning"
    elif hour < 17:
        return "Good Afternoon"
    else:
        return "Good Evening"


def get_system_prompt_template_text() -> str:
    """Returns the raw un-rendered text of the Swargaseema system prompt template.

    Returns:
        The template string with variable placeholders.
    """
    template = load_swargaseema_system_template()
    return template.template_text


def build_system_prompt(
    customer: Optional[dict[str, Any]] = None,
    project: Optional[dict[str, Any]] = None,
) -> str:
    """Renders the Swargaseema inbound AI receptionist system prompt.

    Args:
        customer: Optional dictionary containing customer details such as
            `customer_name` and `preferred_language`.
        project: Optional dictionary containing project details such as
            `project_name` and `project_location`.

    Returns:
        The rendered system prompt as a plain Python string.
    """
    customer = customer or {}
    project = project or {}

    engine = PromptEngine()
    template = load_swargaseema_system_template()
    engine.register_template(template)

    context = {
        "customer_name": customer.get("customer_name")
        or customer.get("name")
        or "Valued Caller",
        "preferred_language": customer.get("preferred_language")
        or customer.get("language")
        or "English",
        "project_name": project.get("project_name")
        or project.get("name")
        or "Swargaseema Sandalwood Farms",
        "project_location": project.get("project_location")
        or project.get("location")
        or "Hyderabad, Telangana",
    }

    rendered = engine.render(SWARGASEEMA_SYSTEM_TEMPLATE_ID, **context)
    return rendered.text
