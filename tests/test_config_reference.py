"""The generated EnvConfig reference must stay in step with the dataclass."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from chassis_sim.config.schema import EnvConfig

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "gen_config_docs.py"


def _load():
    spec = importlib.util.spec_from_file_location("gen_config_docs", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load()
FIELD_NAMES = [field.name for field in dataclasses.fields(EnvConfig)]


def test_every_field_is_grouped_exactly_once():
    grouped = [name for _, _, names in GEN.GROUPS for name in names]
    assert sorted(grouped) == sorted(FIELD_NAMES)
    assert len(grouped) == len(set(grouped))


@pytest.mark.parametrize("table", [GEN.EN, GEN.ZH], ids=["en", "zh"])
def test_every_field_has_a_description(table):
    assert sorted(table) == sorted(FIELD_NAMES)


@pytest.mark.parametrize(("name", "lang"), GEN.TARGETS)
def test_readme_tables_are_up_to_date(name, lang):
    path = REPO / name
    current = path.read_text(encoding="utf-8")
    assert GEN.BEGIN in current and GEN.END in current, name
    assert GEN.splice(path, lang) == current, (
        f"{name} is stale; run python tools/gen_config_docs.py"
    )


@pytest.mark.parametrize(("name", "lang"), GEN.TARGETS)
def test_readme_lists_every_field(name, lang):
    body = (REPO / name).read_text(encoding="utf-8")
    block = body[body.index(GEN.BEGIN) : body.index(GEN.END)]
    rows = [line for line in block.splitlines() if line.startswith("| `")]
    assert len(rows) == len(FIELD_NAMES)
    for field in FIELD_NAMES:
        assert f"| `{field}` |" in block, (name, field)


def test_prose_field_count_matches():
    count = len(FIELD_NAMES)
    en = (REPO / "README.md").read_text(encoding="utf-8")
    zh = (REPO / "README.zh-CN.md").read_text(encoding="utf-8")
    assert f"({count} fields)" in en
    assert f"（{count} 个字段）" in zh
