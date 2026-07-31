"""Prompt template files for PDA Engineering College AI assistant."""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent
PDA_SYSTEM_TEMPLATE_PATH = TEMPLATES_DIR / "pda_system_template.txt"

__all__ = ["PDA_SYSTEM_TEMPLATE_PATH", "TEMPLATES_DIR"]
