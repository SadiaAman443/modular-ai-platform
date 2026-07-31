"""System prompt construction module for PDA Engineering College AI assistant.

This module extracts the business-specific prompt generation logic from the legacy
`gemini_service.py` and migrates it to use the domain-agnostic `ai_core.prompts`
foundation. It produces plain system prompt strings ready for execution.
"""

from datetime import datetime
from typing import Any, Optional

from ai_core.prompts import PromptEngine
from projects.pda_ai.prompts.loader import load_pda_system_template


def get_time_based_greeting(current_hour: Optional[int] = None) -> str:
    """Returns a contextual greeting based on local server time.

    Extracted from `GeminiService.get_time_based_greeting()` in the legacy project.

    Args:
        current_hour: Optional hour of the day (0-23) for testing or override.
            If None, uses `datetime.now().hour`.

    Returns:
        One of 'Good Morning', 'Good Afternoon', or 'Good Evening'.
    """
    hour = current_hour if current_hour is not None else datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning"
    elif 12 <= hour < 17:
        return "Good Afternoon"
    else:
        return "Good Evening"


def get_system_prompt_template_text() -> str:
    """Returns the raw PDA Engineering College system prompt template text.

    Returns:
        The un-rendered template string containing `{variable}` placeholders.
    """
    tmpl = load_pda_system_template()
    return tmpl.template_text


def build_system_prompt(
    student: dict[str, Any],
    campaign: Optional[dict[str, Any]] = None,
    greeting: Optional[str] = None,
) -> str:
    """Constructs a plain system prompt string for a student and campaign call.

    This function replaces legacy `GeminiService.generate_system_instruction()`.
    It uses `ai_core.prompts.PromptEngine` to render the PDA prompt template with
    the provided student and campaign parameters.

    Args:
        student: Dictionary containing student details (`parent_name`,
            `student_name`, `attendance_percentage`).
        campaign: Optional dictionary containing campaign metadata (`type`).
        greeting: Optional override for the time-based greeting.

    Returns:
        A plain rendered system prompt string that can be passed directly to
        an LLM adapter or ConversationEngine.
    """
    campaign_type = campaign.get("type", "Attendance Alert") if campaign else "Attendance Alert"
    resolved_greeting = greeting or get_time_based_greeting()

    parent_name = student.get("parent_name", "Parent")
    student_name = student.get("student_name", "Student")
    attendance_percentage = student.get("attendance_percentage", "N/A")

    template = load_pda_system_template()
    rendered = PromptEngine.render_template(
        template,
        parent_name=parent_name,
        student_name=student_name,
        campaign_type=campaign_type,
        attendance_percentage=attendance_percentage,
        greeting=resolved_greeting,
    )

    return rendered.text
