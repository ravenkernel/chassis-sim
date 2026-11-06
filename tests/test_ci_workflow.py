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


def _pip_commands(step: dict) -> list[str]:
    """一步 run 里的 pip install 命令行（去掉缩进，忽略注释）。"""
    return [
        line.strip()
        for line in (step.get("run") or "").splitlines()
        if line.strip().startswith("pip install")
    ]


def test_test_job_runs_both_numpy_arms(workflow):
    """pyproject 只声明 numpy>=1.24（没有上界）：1.x 与 2.x 两档都得真跑。"""
    matrix = _jobs(workflow)["test"]["strategy"]["matrix"]
    assert matrix["numpy"] == ["numpy<2", "numpy>=2"]


def test_numpy_arm_is_pinned_in_the_same_pip_resolution(workflow):
    """档位必须和 extras 在同一条 pip install 里解析。

    分两步装的话，第二条命令只按自己的约束挑版本，会把 numba 声明的 numpy
    上界（例如 0.60 的 numpy<2.1）踩掉 —— 装出来的组合根本不可运行。
    """
    steps = _jobs(workflow)["test"]["steps"]
    install = [step for step in steps if step.get("name") == "Install"]
    assert install, "test job 缺少 Install 步骤"
    commands = _pip_commands(install[0])
    assert len(commands) == 1, commands
    assert "matrix.numpy" in commands[0], commands[0]
    assert ".[dev,viz,planning]" in commands[0], commands[0]


def test_numpy_major_version_is_verified_after_install(workflow):
    """装成功不等于装到了档位：矩阵值写松了（numpy<3）也要让 job 红。"""
    steps = _jobs(workflow)["test"]["steps"]
    verify = [
        step
        for step in steps
        if (step.get("env") or {}).get("NUMPY_ARM") == "${{ matrix.numpy }}"
    ]
    assert verify, "test job 缺少 env: NUMPY_ARM 把档位传给断言步骤"
    run = verify[0]["run"]
    assert "numpy.__version__" in run, run
    assert "sys.exit" in run, run
    # 档位不符必须非零退出：只 print 一行不算断言。
    assert "与档位" in run, run


def test_only_the_cell_without_a_wheel_is_excluded(workflow):
    """3.13 × numpy<2 是唯一装不出来的格：cp313 的 numpy 从 2.1.0 才开始。"""
    matrix = _jobs(workflow)["test"]["strategy"]["matrix"]
    assert matrix["exclude"] == [{"python-version": "3.13", "numpy": "numpy<2"}]
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "cp313" in text, "排除理由必须写在 ci.yml 注释里（cp313 没有 numpy<2 轮子）"
    assert "numba 0.61.0" in text or "numba 0.61" in text, "3.13 保留的理由也要写清"


def test_readme_python_blocks_have_a_docs_job(workflow):
    """README 的 python 块必须由 CI 真跑，否则文档腐烂没人知道。"""
    jobs = _jobs(workflow)
    assert "docs" in jobs, "缺少跑 README 代码块的 docs job"
    runs = [step for step in jobs["docs"]["steps"] if "run" in step]
    assert any("tools/check_readme_blocks.py" in step["run"] for step in runs)
    checker = WORKFLOW.parents[2] / "tools" / "check_readme_blocks.py"
    assert checker.is_file(), f"docs job 调用的脚本不存在：{checker}"


def test_config_reference_is_checked_in_ci(workflow):
    """README 的字段表由生成器产出，就必须由 CI 验证它没过期。"""
    jobs = _jobs(workflow)
    assert "docs" in jobs, "缺少 docs job"
    runs = [step for step in jobs["docs"]["steps"] if "run" in step]
    assert any("tools/gen_config_docs.py --check" in step["run"] for step in runs)
    generator = WORKFLOW.parents[2] / "tools" / "gen_config_docs.py"
    assert generator.is_file(), f"docs job 调用的生成器不存在：{generator}"


def test_gate_tools_are_not_installed_behind_the_dev_extra(workflow):
    """mypy / pytest-cov / types-PyYAML 只由 dev extra 提供。

    3.0.0 之前 CI 在 `pip install -e .[dev]` 之后再补一条 pip install 装这三样，
    于是照 README 装 .[dev] 的人得到的不是 CI 那一套 —— 这条断言防它回来。
    """
    offenders = []
    for job_name, job in _jobs(workflow).items():
        for step in job.get("steps") or []:
            for command in _pip_commands(step):
                for tool in ("mypy", "pytest-cov", "types-PyYAML"):
                    if tool in command:
                        offenders.append(f"{job_name} / {step.get('name')}: {command}")
    assert not offenders, offenders
