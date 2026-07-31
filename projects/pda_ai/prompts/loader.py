"""Prompt template loader for the PDA Engineering College AI project.

This module adapts the domain-agnostic `ai_core.prompts.PromptLoader` to load
PDA-specific prompt templates from filesystem storage.
"""

from typing import Optional

from ai_core.prompts import PromptLoader, PromptTemplate
from projects.pda_ai.prompts.templates import PDA_SYSTEM_TEMPLATE_PATH


def load_pda_system_template(
    template_id: str = "pda_student_support_system",
) -> PromptTemplate:
    """Loads the PDA Engineering College system prompt template from disk.

    Args:
        template_id: Optional custom ID to assign to the loaded template.

    Returns:
        A validated PromptTemplate instance ready for registration in PromptEngine.

    Raises:
        PromptLoadError: If the template file is missing or cannot be read.
    """
    return PromptLoader.load_from_text_file(
        PDA_SYSTEM_TEMPLATE_PATH,
        template_id=template_id,
        is_system_prompt=True,
    )
