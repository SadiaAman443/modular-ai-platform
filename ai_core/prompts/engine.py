"""Prompt Engine for managing, retrieving, and rendering prompt templates.

This module implements the central service for handling prompt templates and
rendering parameterized prompts without coupling to any LLM provider.
"""

from typing import Any, Optional

from ai_core.prompts.exceptions import PromptException, PromptRenderError
from ai_core.prompts.models import PromptTemplate, RenderedPrompt


class PromptEngine:
    """Service for registering, retrieving, and rendering prompt templates.

    The PromptEngine acts as an in-memory registry and rendering engine for all
    application prompt templates. It never invokes any LLM APIs directly.

    Attributes:
        _templates: Internal mapping of template IDs to PromptTemplate instances.
    """

    def __init__(self, templates: Optional[dict[str, PromptTemplate]] = None) -> None:
        """Initializes the PromptEngine.

        Args:
            templates: Optional dictionary of pre-loaded prompt templates.
        """
        self._templates: dict[str, PromptTemplate] = {}
        if templates:
            for tmpl in templates.values():
                self.register_template(tmpl)

    def register_template(self, template: PromptTemplate) -> None:
        """Registers a PromptTemplate in the engine.

        Args:
            template: The PromptTemplate instance to store.

        Raises:
            PromptException: If template is not an instance of PromptTemplate.
        """
        if not isinstance(template, PromptTemplate):
            raise PromptException(
                f"Expected PromptTemplate instance, got {type(template).__name__}."
            )
        self._templates[template.template_id] = template

    def get_template(self, template_id: str) -> PromptTemplate:
        """Retrieves a registered PromptTemplate by ID.

        Args:
            template_id: Unique identifier of the template.

        Returns:
            The matching PromptTemplate instance.

        Raises:
            PromptException: If the requested template_id is not registered.
        """
        tmpl = self._templates.get(template_id)
        if not tmpl:
            raise PromptException(
                f"Prompt template '{template_id}' is not registered in PromptEngine."
            )
        return tmpl

    def list_templates(self) -> list[str]:
        """Returns a sorted list of all registered template IDs."""
        return sorted(self._templates.keys())

    def render(self, template_id: str, **kwargs: Any) -> RenderedPrompt:
        """Retrieves a registered template by ID and renders it with kwargs.

        Args:
            template_id: Unique identifier of the template to render.
            **kwargs: Variable substitutions for template placeholders.

        Returns:
            A RenderedPrompt instance containing the substituted text.

        Raises:
            PromptException: If template is not registered.
            PromptRenderError: If required variables are missing or rendering fails.
        """
        template = self.get_template(template_id)
        return template.render(**kwargs)

    @classmethod
    def render_template(cls, template: PromptTemplate, **kwargs: Any) -> RenderedPrompt:
        """Renders an arbitrary PromptTemplate instance without registration.

        Args:
            template: The PromptTemplate instance to render.
            **kwargs: Variable substitutions for template placeholders.

        Returns:
            A RenderedPrompt instance containing the substituted text.

        Raises:
            PromptRenderError: If required variables are missing or rendering fails.
        """
        return template.render(**kwargs)

    @classmethod
    def render_string(
        cls,
        template_text: str,
        template_id: str = "ad_hoc_template",
        is_system_prompt: bool = False,
        **kwargs: Any,
    ) -> RenderedPrompt:
        """Creates an ad-hoc PromptTemplate from a string and renders it immediately.

        Args:
            template_text: Raw template string with `{var}` placeholders.
            template_id: Identifier to tag the rendered prompt.
            is_system_prompt: Whether this is a system instruction template.
            **kwargs: Variable substitutions for placeholders.

        Returns:
            A RenderedPrompt instance containing the substituted text.

        Raises:
            PromptRenderError: If required variables are missing or rendering fails.
        """
        template = PromptTemplate(
            template_id=template_id,
            template_text=template_text,
            is_system_prompt=is_system_prompt,
        )
        return template.render(**kwargs)
