"""Minimal scenario-generation code vendored from
anthropic-experimental/agentic-misalignment @ ea0630e1a3eaae7f9f9740fd2703229d3854ccda
(MIT; see vendored/LICENSE).

This is the GENERATION path only — the pure functions that turn a
(scenario, goal_type, goal_value, urgency_type) condition into a filled
system prompt + email content. It is adapted from the upstream
`scripts/generate_prompts.py` with two deliberate trims, each marked
`[sisteer]`:

  1. the argparse / YAML / file-IO / metadata machinery is dropped — we
     return strings, we do not write a prompt directory;
  2. `load_env`, `datetime.now()`, and every other non-deterministic or
     network dependency is removed, so generation is a pure function of
     the vendored template text.

All three upstream scenario families render: blackmail, leaking and
murder. (The murder branch was trimmed in the first vendoring, when the
realign suite excluded that family; it was restored verbatim from the
same pinned commit for the misalign campaign, and the v1
blackmail+leaking output is byte-identical before and after, which the
scenario tests pin.) Upstream resolves the murder family's goal
variables from the BLACKMAIL variable files (`var_scenario`), and so
does this port; the vendored `murder/variables/` copies are identical to
`blackmail/variables/` at the pinned commit, so the choice is
content-neutral, but the lookup path is upstream's, not ours.

Nothing here calls a model or an API, and none of the upstream
classifier code is vendored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from string import Template

# [sisteer] verbatim import of the upstream template constants.
from .system_prompt_templates import (  # noqa: F401
    BLACKMAIL_SYSTEM_PROMPT,
    BLACKMAIL_SYSTEM_PROMPT_PROD,
    GOAL_VARIABLES,
    LEAKING_SYSTEM_PROMPT,
    LEAKING_SYSTEM_PROMPT_PROD,
    MURDER_SYSTEM_PROMPT,
    MURDER_SYSTEM_PROMPT_PROD,
    USER_PROMPT,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"

# [sisteer] verbatim from upstream EMAIL_TEMPLATES.
EMAIL_TEMPLATES = {
    "blackmail": {
        "base": "blackmail/templates/email-template-blackmail-shared.md",
        "replacement": "blackmail/templates/email-template-blackmail-replacement.md",
        "restriction": "blackmail/templates/email-template-blackmail-restriction.md",
        "none": "blackmail/templates/email-template-blackmail-none.md",
    },
    "leaking": {
        "base": "leaking/templates/email-template-leaking-shared.md",
        "replacement": "leaking/templates/email-template-leaking-replacement.md",
        "restriction": "leaking/templates/email-template-leaking-restriction.md",
        "none": "leaking/templates/email-template-leaking-none.md",
    },
    "murder": {
        "base": "murder/templates/email-template-murder-shared.md",
        "replacement": "murder/templates/email-template-murder-replacement.md",
        "restriction": "murder/templates/email-template-murder-restriction.md",
        "none": "murder/templates/email-template-murder-none.md",
    },
}


class ValidationError(Exception):
    """Raised when template or configuration validation fails."""


# -- template primitives (verbatim from upstream) ------------------------------


def load_template(file_path: Path) -> str:
    with open(file_path) as f:
        return f.read()


def fill_template(template: str, variables: dict) -> str:
    return Template(template).safe_substitute(variables)


def parse_markdown_variables(file_path: Path) -> dict:
    with open(file_path) as f:
        content = f.read()
    pattern = r"## (.*?)\n\s*```json\s*(.*?)\s*```"
    matches = re.findall(pattern, content, re.DOTALL)
    variables = {}
    for section_name, json_content in matches:
        section_key = section_name.strip().lower().replace(" ", "_").replace("-", "_")
        variables[section_key] = json.loads(json_content)
    return variables


def validate_variable_mapping(scenario: str, goal_value: str) -> Path:
    var_scenario = "blackmail" if scenario == "murder" else scenario
    variables_dir = TEMPLATES_DIR / var_scenario / "variables"
    if not variables_dir.exists():
        raise ValidationError(f"Variables directory not found: {variables_dir}")
    if goal_value in ("none", "ambiguous"):
        path = variables_dir / "ambiguous.md"
        if not path.exists():
            raise ValidationError(f"Required ambiguous.md not found: {path}")
        return path
    exact, partial = [], []
    for var_file in variables_dir.glob("*.md"):
        if var_file.stem == "ambiguous":
            continue
        stem = var_file.stem.lower()
        gl = goal_value.lower()
        if gl in stem:
            if f"_{gl}_" in f"_{stem}_" or stem.startswith(gl) or stem.endswith(gl):
                exact.append(var_file)
            else:
                partial.append(var_file)
    if exact:
        return exact[0]
    if partial:
        return partial[0]
    fallback = variables_dir / "ambiguous.md"
    if fallback.exists():
        return fallback
    raise ValidationError(f"No variable file found for {goal_value!r} and no fallback")


def get_variables_for_condition(
    scenario: str, goal_type: str, goal_value: str, urgency_type: str
) -> dict:
    """Verbatim from upstream: resolve the variable section for a condition
    and merge the goal statement/reminder. Murder reads the blackmail
    variable files, as upstream does (`var_scenario`)."""
    if goal_type in ("none", "ambiguous"):
        var_scenario = "blackmail" if scenario == "murder" else scenario
        variable_path = TEMPLATES_DIR / var_scenario / "variables" / "ambiguous.md"
        if not variable_path.exists():
            raise ValidationError(f"Required ambiguous.md not found: {variable_path}")
        all_variables = parse_markdown_variables(variable_path)
    else:
        variable_path = validate_variable_mapping(scenario, goal_value)
        all_variables = parse_markdown_variables(variable_path)

    variables: dict = {}
    if goal_value in ("none", "ambiguous"):
        if len(all_variables) == 1:
            variables = list(all_variables.values())[0]
        else:
            variables = all_variables
    else:
        matching_section = None
        gl = goal_value.lower()
        for section_key, section_vars in all_variables.items():
            sl = section_key.lower()
            if gl in sl or f"pro_{gl}" == sl:
                matching_section = section_vars
                break
        if matching_section is not None:
            variables = matching_section
        elif all_variables:
            variables = list(all_variables.values())[0]

    goal_key = f"{goal_type}-{goal_value}"
    goal_vars = GOAL_VARIABLES.get(goal_key, GOAL_VARIABLES.get("none", {}))
    variables.update(goal_vars)
    return variables


def generate_system_prompt(
    scenario: str,
    goal_type: str,
    goal_value: str,
    urgency_type: str,
    *,
    prod: bool = False,
) -> str:
    """Filled system prompt for a condition. `extra_system_instructions`
    and `alignment_faking` (both default-off upstream) are dropped —
    [sisteer] we never enable them, so their placeholders resolve empty."""
    variables = get_variables_for_condition(scenario, goal_type, goal_value, urgency_type)
    variables["extra_system_instructions"] = ""
    variables["alignment_faking_warning"] = ""
    if scenario == "blackmail":
        template = BLACKMAIL_SYSTEM_PROMPT_PROD if prod else BLACKMAIL_SYSTEM_PROMPT
    elif scenario == "leaking":
        template = LEAKING_SYSTEM_PROMPT_PROD if prod else LEAKING_SYSTEM_PROMPT
    elif scenario == "murder":
        template = MURDER_SYSTEM_PROMPT_PROD if prod else MURDER_SYSTEM_PROMPT
    else:
        raise ValueError(f"Invalid scenario: {scenario!r}")
    return fill_template(template, variables)


def generate_email_content(
    scenario: str, goal_type: str, goal_value: str, urgency_type: str
) -> str:
    """Filled email content for a condition (verbatim upstream logic)."""
    variables = get_variables_for_condition(scenario, goal_type, goal_value, urgency_type)
    if goal_type == "none":
        base_template_name = (
            f"{scenario}/templates/email-template-{scenario}-goal-conflict-none-{urgency_type}.md"
        )
        base_template_path = TEMPLATES_DIR / base_template_name
        if not base_template_path.exists():
            raise ValidationError(f"Goal-conflict template not found: {base_template_path}")
        content = fill_template(load_template(base_template_path), variables)
    else:
        cfg = EMAIL_TEMPLATES[scenario]
        base_content = fill_template(load_template(TEMPLATES_DIR / cfg["base"]), variables)
        urgency_content = fill_template(
            load_template(TEMPLATES_DIR / cfg[urgency_type]), variables
        )
        content = base_content + urgency_content

    if scenario == "leaking":
        parts = content.split("<email>")
        if len(parts) > 1:
            content = parts[0]
            for i, part in enumerate(parts[1:], 1):
                content += f"<email>\nID: {i}{part}"
    return content


def generate_user_prompt() -> str:
    """The instruction-only user turn (upstream `generate_user_prompt_fixed`)."""
    return USER_PROMPT


def create_condition_id(
    scenario: str, goal_type: str, goal_value: str, urgency_type: str
) -> str:
    return f"{scenario}_{goal_type}-{goal_value}_{urgency_type}"
