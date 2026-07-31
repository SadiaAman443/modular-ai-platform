"""Prompt template loading and discovery for Swargaseema AI.

This module adapts the domain-agnostic `ai_core.prompts.PromptLoader` to load
and validate the Swargaseema Sandalwood Farms inbound receptionist system template.
"""

from pathlib import Path

from ai_core.prompts import PromptLoader, PromptTemplate, PromptVariable

SWARGASEEMA_SYSTEM_TEMPLATE_ID = "swargaseema_inbound_receptionist"
SWARGASEEMA_SYSTEM_TEMPLATE_PATH = (
    Path(__file__).parent / "templates" / "swargaseema_system_template.txt"
)


def load_swargaseema_system_template() -> PromptTemplate:
    """Loads the Swargaseema inbound receptionist system prompt template from disk.

    Returns:
        A validated PromptTemplate configured with default variable fallbacks.

    Raises:
        PromptLoadError: If the template file cannot be read.
    """
    template = PromptLoader.load_from_text_file(
        SWARGASEEMA_SYSTEM_TEMPLATE_PATH,
        template_id=SWARGASEEMA_SYSTEM_TEMPLATE_ID,
        is_system_prompt=True,
    )

    template.variables = {
        "customer_name": PromptVariable(
            name="customer_name",
            required=False,
            default_value="Valued Caller",
            description="Name of the calling customer or lead.",
        ),
        "preferred_language": PromptVariable(
            name="preferred_language",
            required=False,
            default_value="English",
            description="Preferred spoken language for communication.",
        ),
        "project_name": PromptVariable(
            name="project_name",
            required=False,
            default_value="Swargaseema Sandalwood Farms",
            description="Name of the farmland development project.",
        ),
        "project_location": PromptVariable(
            name="project_location",
            required=False,
            default_value="Hyderabad, Telangana",
            description="Geographic location of the project.",
        ),
    }

    return template
