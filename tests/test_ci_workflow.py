"""CI 工作流的形状守卫（2026-11-04 事故的回归用例）。

PyYAML 只能证明文件是合法 YAML，证明不了它是 GitHub 认的 workflow：把
`matrix.python-version` 写成 `"3.12"` 标量时，YAML 依然合法，但 GitHub 判整份
workflow 无效 —— run 直接以 0 job 失败，没有任何 job 日志可看（实测 run
41843571545/41843563813 两次都是 jobs=[]）。本文件把这条规则与跨平台 job 的关键
形状钉住，省掉一次"推上去才发现"。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
_EXPR = re.compile(r"^\$\{\{.*\}\}$", re.DOTALL)


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _jobs(workflow: dict) -> dict:
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict) and jobs, "ci.yml 必须声明 jobs"
    return jobs


def test_matrix_values_are_arrays_or_expressions(workflow):
    """矩阵值只能是数组或单个 ${{ }} 表达式 —— 标量会让整份 workflow 失效。"""
    for job_name, job in _jobs(workflow).items():
        matrix = (job.get("strategy") or {}).get("matrix")
        if matrix is None:
            continue
        if _EXPR.match(str(matrix)):
            continue
        assert isinstance(matrix, dict), f"{job_name}: matrix 既不是映射也不是表达式"
        for key, value in matrix.items():
            ok = isinstance(value, list) or (
                isinstance(value, str) and _EXPR.match(value)
            )
            assert ok, (
                f"{job_name}.strategy.matrix.{key} = {value!r} 不是数组也不是 "
                "表达式；GitHub 会判定整份 workflow 无效（run 0 job 失败）"
            )


def test_every_job_has_runs_on_and_steps(workflow):
    for job_name, job in _jobs(workflow).items():
        assert job.get("runs-on"), f"{job_name}: 缺少 runs-on"
        assert job.get("steps"), f"{job_name}: 缺少 steps"


def test_cross_platform_job_exists(workflow):
    """macOS/Windows 必须真跑，否则 OS Independent 只是声明。"""
    jobs = _jobs(workflow)
    assert "test-os" in jobs, "缺少跨平台 job test-os"
    job = jobs["test-os"]
    assert job["runs-on"] == "${{ matrix.os }}"
    matrix = job["strategy"]["matrix"]
    assert {"macos-latest", "windows-latest"} <= set(matrix["os"])
    assert job["strategy"]["fail-fast"] is False


def test_cross_platform_job_runs_same_command_in_bash(workflow):
    """两个 run 步骤都必须钉 shell: bash（Windows 默认是 pwsh）。"""
    job = _jobs(workflow)["test-os"]
    runs = [step for step in job["steps"] if "run" in step]
    assert len(runs) >= 2, "test-os 应同时有安装与测试两个 run 步骤"
    for step in runs:
        assert step.get("shell") == "bash", f"步骤 {step.get('name')} 未钉 shell: bash"
    assert any("pytest" in step["run"] for step in runs)


def test_ubuntu_matrix_still_covers_every_declared_python(workflow):
    """ubuntu 那档要覆盖 pyproject 声明支持的全部 Python 小版本。"""
    matrix = _jobs(workflow)["test"]["strategy"]["matrix"]
    versions = matrix["python-version"]
    assert isinstance(versions, list)
    assert {"3.10", "3.11", "3.12", "3.13"} <= set(versions)
