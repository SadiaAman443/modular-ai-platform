"""Unit tests for the AI Core Prompt module."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ai_core.prompts.engine import PromptEngine
from ai_core.prompts.exceptions import (
    PromptException,
    PromptLoadError,
    PromptRenderError,
    PromptTemplateError,
)
from ai_core.prompts.loader import PromptLoader
from ai_core.prompts.models import PromptTemplate, PromptVariable, RenderedPrompt


def test_prompt_template_discovery_and_rendering():
    tmpl = PromptTemplate(
        template_id="greeting",
        template_text="Hello, {name}! Welcome to {service}.",
    )
    assert "name" in tmpl.variables
    assert "service" in tmpl.variables

    rendered = tmpl.render(name="Alice", service="AI Platform")
    assert isinstance(rendered, RenderedPrompt)
    assert rendered.text == "Hello, Alice! Welcome to AI Platform."
    assert rendered.template_id == "greeting"
    assert rendered.variables_used == {"name": "Alice", "service": "AI Platform"}


def test_prompt_template_missing_required_var():
    tmpl = PromptTemplate(
        template_id="test",
        template_text="Hello, {user}!",
    )
    with pytest.raises(PromptRenderError):
        tmpl.render()


def test_prompt_template_default_value():
    tmpl = PromptTemplate(
        template_id="defaults",
        template_text="Welcome, {user}! Role: {role}",
        variables={
            "user": PromptVariable(name="user", required=True),
            "role": PromptVariable(name="role", required=False, default_value="Guest"),
        },
    )
    rendered = tmpl.render(user="Bob")
    assert rendered.text == "Welcome, Bob! Role: Guest"


def test_prompt_template_serialization():
    tmpl = PromptTemplate(
        template_id="serialize_test",
        template_text="Hi {name}",
        is_system_prompt=True,
    )
    data = tmpl.to_dict()
    assert data["template_id"] == "serialize_test"
    assert data["is_system_prompt"] is True

    tmpl2 = PromptTemplate.from_dict(data)
    assert tmpl2.template_id == "serialize_test"
    assert tmpl2.is_system_prompt is True


def test_prompt_loader_dict():
    data = {
        "template_id": "dict_test",
        "template_text": "Hello {city}",
    }
    tmpl = PromptLoader.load_from_dict(data)
    assert tmpl.template_id == "dict_test"
    assert "city" in tmpl.variables


def test_prompt_loader_files_and_directory():
    with TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir)

        # JSON file
        json_path = dir_path / "greeting.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"template_id": "json_greet", "template_text": "Hi {name}"}, f)

        # Text file
        txt_path = dir_path / "simple_sys.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("You are an AI assistant.")

        # Load individual JSON
        tmpl1 = PromptLoader.load_from_json_file(json_path)
        assert tmpl1.template_id == "json_greet"

        # Load individual text
        tmpl2 = PromptLoader.load_from_text_file(txt_path, is_system_prompt=True)
        assert tmpl2.is_system_prompt is True
        assert tmpl2.template_text == "You are an AI assistant."

        # Load directory
        all_tmpls = PromptLoader.load_from_directory(dir_path)
        assert "json_greet" in all_tmpls
        assert "simple_sys" in all_tmpls


def test_prompt_loader_missing_file():
    with pytest.raises(PromptLoadError):
        PromptLoader.load_from_json_file("/non/existent/path/prompt.json")


def test_prompt_engine_registration_and_rendering():
    engine = PromptEngine()
    tmpl = PromptTemplate(
        template_id="engine_test",
        template_text="Score: {score}",
    )
    engine.register_template(tmpl)

    assert "engine_test" in engine.list_templates()
    rendered = engine.render("engine_test", score=100)
    assert rendered.text == "Score: 100"


def test_prompt_engine_render_string():
    rendered = PromptEngine.render_string(
        "Ad hoc {value}",
        template_id="ad_hoc",
        value=42,
    )
    assert rendered.text == "Ad hoc 42"
    assert rendered.template_id == "ad_hoc"


def test_prompt_engine_unregistered_template():
    engine = PromptEngine()
    with pytest.raises(PromptException):
        engine.get_template("unknown_template")
