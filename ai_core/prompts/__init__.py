"""Prompt foundation module for the AI Core Platform.

This package provides domain-agnostic prompt templates, loaders, and rendering
services without coupling to any LLM provider.
"""

from ai_core.prompts.engine import PromptEngine
from ai_core.prompts.exceptions import (
    PromptException,
    PromptLoadError,
    PromptRenderError,
    PromptTemplateError,
)
from ai_core.prompts.loader import PromptLoader
from ai_core.prompts.models import PromptTemplate, PromptVariable, RenderedPrompt

__all__ = [
    "PromptEngine",
    "PromptException",
    "PromptLoadError",
    "PromptLoader",
    "PromptRenderError",
    "PromptTemplate",
    "PromptTemplateError",
    "PromptVariable",
    "RenderedPrompt",
]
