"""Prompt loader utilities for the AI Core Prompt module.

This module provides static and class methods to load PromptTemplate instances
from dictionaries, text files, JSON files, YAML files, or directories.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

from ai_core.prompts.exceptions import PromptLoadError, PromptTemplateError
from ai_core.prompts.models import PromptTemplate


class PromptLoader:
    """Loads and constructs PromptTemplate instances from external sources."""

    @classmethod
    def load_from_dict(cls, data: dict[str, Any]) -> PromptTemplate:
        """Loads a PromptTemplate from a Python dictionary.

        Args:
            data: Raw template dictionary.

        Returns:
            A validated PromptTemplate instance.

        Raises:
            PromptLoadError: If dictionary validation fails.
        """
        try:
            return PromptTemplate.from_dict(data)
        except PromptTemplateError as exc:
            raise PromptLoadError(f"Failed to load PromptTemplate from dict: {exc}") from exc

    @classmethod
    def load_from_text_file(
        cls,
        file_path: Union[str, Path],
        template_id: Optional[str] = None,
        is_system_prompt: bool = False,
    ) -> PromptTemplate:
        """Loads a PromptTemplate from a raw text file.

        Args:
            file_path: Path to the `.txt` file containing the prompt template text.
            template_id: Optional unique identifier. Defaults to file stem if None.
            is_system_prompt: Whether this template represents a system instruction.

        Returns:
            A constructed PromptTemplate instance.

        Raises:
            PromptLoadError: If the file does not exist or cannot be read.
        """
        path = Path(file_path)
        if not path.exists():
            raise PromptLoadError(f"Prompt text file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            tid = template_id or path.stem
            return PromptTemplate(
                template_id=tid,
                template_text=content,
                is_system_prompt=is_system_prompt,
            )
        except Exception as exc:
            raise PromptLoadError(f"Error reading prompt text file '{path}': {exc}") from exc

    @classmethod
    def load_from_json_file(cls, file_path: Union[str, Path]) -> PromptTemplate:
        """Loads a PromptTemplate from a JSON file.

        Args:
            file_path: Path to the `.json` prompt configuration file.

        Returns:
            A constructed PromptTemplate instance.

        Raises:
            PromptLoadError: If JSON decoding or file reading fails.
        """
        path = Path(file_path)
        if not path.exists():
            raise PromptLoadError(f"Prompt JSON file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.load_from_dict(data)
        except json.JSONDecodeError as exc:
            raise PromptLoadError(f"Invalid JSON in prompt file '{path}': {exc}") from exc
        except Exception as exc:
            raise PromptLoadError(f"Error reading prompt JSON file '{path}': {exc}") from exc

    @classmethod
    def load_from_yaml_file(cls, file_path: Union[str, Path]) -> PromptTemplate:
        """Loads a PromptTemplate from a YAML file.

        Args:
            file_path: Path to the `.yaml` or `.yml` prompt configuration file.

        Returns:
            A constructed PromptTemplate instance.

        Raises:
            PromptLoadError: If YAML parsing fails or PyYAML is missing.
        """
        try:
            import yaml
        except ImportError as exc:
            raise PromptLoadError(
                "PyYAML is required to load YAML prompt files. Please install 'pyyaml'."
            ) from exc

        path = Path(file_path)
        if not path.exists():
            raise PromptLoadError(f"Prompt YAML file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise PromptLoadError("Top-level YAML prompt content must be a dictionary mapping.")
            return cls.load_from_dict(data)
        except Exception as exc:
            raise PromptLoadError(f"Error reading prompt YAML file '{path}': {exc}") from exc

    @classmethod
    def load_from_directory(cls, directory_path: Union[str, Path]) -> dict[str, PromptTemplate]:
        """Loads all prompt templates from JSON, YAML, and text files in a directory.

        Args:
            directory_path: Path to the directory containing prompt files.

        Returns:
            A dictionary mapping `template_id` to `PromptTemplate` instances.

        Raises:
            PromptLoadError: If the directory does not exist or files are malformed.
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise PromptLoadError(f"Prompt directory not found or is not a directory: {dir_path}")

        templates: dict[str, PromptTemplate] = {}
        for p in sorted(dir_path.iterdir()):
            if not p.is_file():
                continue
            try:
                suffix = p.suffix.lower()
                if suffix == ".json":
                    tmpl = cls.load_from_json_file(p)
                elif suffix in (".yaml", ".yml"):
                    tmpl = cls.load_from_yaml_file(p)
                elif suffix == ".txt":
                    tmpl = cls.load_from_text_file(p)
                else:
                    continue
                templates[tmpl.template_id] = tmpl
            except Exception as exc:
                raise PromptLoadError(f"Failed to load template from file '{p}': {exc}") from exc

        return templates
