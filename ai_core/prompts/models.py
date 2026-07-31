"""Data models for prompt templates and rendered prompts.

This module defines domain-agnostic structures for declaring, validating, and
rendering reusable prompt templates.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ai_core.prompts.exceptions import PromptRenderError, PromptTemplateError


@dataclass
class PromptVariable:
    """Metadata describing a variable within a prompt template.

    Attributes:
        name: Name of the variable placeholder in the template.
        required: Whether a value must be supplied during rendering.
        default_value: Default value if not provided during rendering.
        description: Documentation describing the purpose of the variable.
    """

    name: str
    required: bool = True
    default_value: Optional[Any] = None
    description: Optional[str] = None


@dataclass
class RenderedPrompt:
    """The result of rendering a prompt template with variables.

    Attributes:
        text: The final rendered prompt string.
        template_id: Identifier of the source template that generated this prompt.
        variables_used: Dictionary of variable names and values used during rendering.
    """

    text: str
    template_id: str
    variables_used: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTemplate:
    """A reusable, parameterized template for system or user prompts.

    Attributes:
        template_id: Unique identifier for this prompt template.
        template_text: String containing `{variable}` placeholders.
        description: Optional documentation describing template intent.
        is_system_prompt: Whether this template is intended for system instructions.
        variables: Optional mapping of variable names to `PromptVariable` metadata.
    """

    template_id: str
    template_text: str
    description: Optional[str] = None
    is_system_prompt: bool = False
    variables: dict[str, PromptVariable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates template text and discovers placeholders after initialization."""
        if not self.template_id or not isinstance(self.template_id, str):
            raise PromptTemplateError("PromptTemplate must have a non-empty string template_id.")
        if not isinstance(self.template_text, str):
            raise PromptTemplateError("PromptTemplate template_text must be a string.")

        # Discover {variable} placeholders in template_text
        placeholders = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", self.template_text))
        for name in placeholders:
            if name not in self.variables:
                self.variables[name] = PromptVariable(name=name, required=True)

    def render(self, **kwargs: Any) -> RenderedPrompt:
        """Renders the template by substituting variable values.

        Args:
            **kwargs: Values for template placeholders.

        Returns:
            A RenderedPrompt instance containing the substituted text.

        Raises:
            PromptRenderError: If required variables are missing or formatting fails.
        """
        values: dict[str, Any] = {}
        missing: list[str] = []

        for var_name, var_meta in self.variables.items():
            if var_name in kwargs:
                values[var_name] = kwargs[var_name]
            elif var_meta.default_value is not None:
                values[var_name] = var_meta.default_value
            elif var_meta.required:
                missing.append(var_name)

        if missing:
            raise PromptRenderError(
                f"Missing required variables for template '{self.template_id}': {sorted(missing)}"
            )

        try:
            rendered_text = self.template_text.format(**values)
        except KeyError as exc:
            raise PromptRenderError(
                f"Unresolved placeholder {exc} in template '{self.template_id}'."
            ) from exc
        except Exception as exc:
            raise PromptRenderError(
                f"Error rendering template '{self.template_id}': {exc}"
            ) from exc

        return RenderedPrompt(
            text=rendered_text,
            template_id=self.template_id,
            variables_used=values,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes template properties to a dictionary.

        Returns:
            A dictionary representation of the prompt template.
        """
        return {
            "template_id": self.template_id,
            "template_text": self.template_text,
            "description": self.description,
            "is_system_prompt": self.is_system_prompt,
            "variables": {
                name: {
                    "required": var.required,
                    "default_value": var.default_value,
                    "description": var.description,
                }
                for name, var in self.variables.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        """Creates a PromptTemplate from a dictionary representation.

        Args:
            data: Mapping containing template properties.

        Returns:
            A populated PromptTemplate instance.

        Raises:
            PromptTemplateError: If required fields are missing or malformed.
        """
        if not isinstance(data, dict):
            raise PromptTemplateError("Expected a dictionary to construct PromptTemplate.")

        template_id = data.get("template_id")
        template_text = data.get("template_text")
        if not template_id or template_text is None:
            raise PromptTemplateError(
                "PromptTemplate dictionary must contain 'template_id' and 'template_text'."
            )

        variables_raw = data.get("variables", {})
        variables: dict[str, PromptVariable] = {}
        if isinstance(variables_raw, dict):
            for var_name, meta in variables_raw.items():
                if isinstance(meta, dict):
                    variables[var_name] = PromptVariable(
                        name=var_name,
                        required=meta.get("required", True),
                        default_value=meta.get("default_value"),
                        description=meta.get("description"),
                    )
                else:
                    variables[var_name] = PromptVariable(name=var_name, required=True)

        return cls(
            template_id=str(template_id),
            template_text=str(template_text),
            description=data.get("description"),
            is_system_prompt=bool(data.get("is_system_prompt", False)),
            variables=variables,
        )
