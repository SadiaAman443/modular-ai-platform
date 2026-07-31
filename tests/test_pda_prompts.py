"""Unit tests for the PDA Engineering College AI prompt generation layer."""

import pytest

from projects.pda_ai.prompts import (
    build_system_prompt,
    get_system_prompt_template_text,
    get_time_based_greeting,
    load_pda_system_template,
)


def test_time_based_greeting_hours():
    assert get_time_based_greeting(current_hour=8) == "Good Morning"
    assert get_time_based_greeting(current_hour=14) == "Good Afternoon"
    assert get_time_based_greeting(current_hour=20) == "Good Evening"
    assert get_time_based_greeting(current_hour=2) == "Good Evening"


def test_load_pda_system_template():
    tmpl = load_pda_system_template()
    assert tmpl.template_id == "pda_student_support_system"
    assert tmpl.is_system_prompt is True

    expected_vars = {
        "parent_name",
        "student_name",
        "campaign_type",
        "attendance_percentage",
        "greeting",
    }
    assert expected_vars.issubset(set(tmpl.variables.keys()))


def test_build_system_prompt_complete():
    student = {
        "parent_name": "Rajesh Kumar",
        "student_name": "Aarav Kumar",
        "attendance_percentage": "68%",
    }
    campaign = {"type": "Attendance Warning"}

    prompt_str = build_system_prompt(
        student=student,
        campaign=campaign,
        greeting="Good Morning",
    )

    assert isinstance(prompt_str, str)
    assert "You are calling Rajesh Kumar regarding Aarav Kumar's Attendance Warning" in prompt_str
    assert "(Attendance: 68%)" in prompt_str
    assert "Good Morning! I am the AI Student Support Assistant calling from PDA College." in prompt_str
    assert "Shri Rajesh Kumar ji se baat kar raha hoon" in prompt_str


def test_build_system_prompt_defaults():
    student = {
        "parent_name": "Suresh",
        "student_name": "Ramesh",
    }

    prompt_str = build_system_prompt(student=student, greeting="Good Afternoon")
    assert isinstance(prompt_str, str)
    assert "Attendance Alert" in prompt_str
    assert "(Attendance: N/A)" in prompt_str
    assert "Good Afternoon" in prompt_str


def test_get_system_prompt_template_text():
    raw_text = get_system_prompt_template_text()
    assert "{parent_name}" in raw_text
    assert "{student_name}" in raw_text
    assert "{attendance_percentage}" in raw_text
