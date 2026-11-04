#!/usr/bin/env python3
"""README 的 python 围栏块守卫：抽块、逐个在子进程里执行、报真实计数。

用法（仓库根目录或任意目录都可以）：

    python tools/check_readme_blocks.py
    python tools/check_readme_blocks.py README.md --timeout 120
    python tools/check_readme_blocks.py --list

约定：python 块的第一行写 `# no-exec`（可带理由，如 `# no-exec: 需要 GPU`）即显式
跳过；跳过的块计入 skipped 并逐条打印，绝不会被算成 passed。

退出码：0 = 没有失败（允许显式跳过）；1 = 至少一个块跑挂；2 = 结构问题（文件缺失、
围栏未闭合、一个 python 块都没有 —— 那种情况不能算通过）。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = ("README.md", "README.zh-CN.md")
DEFAULT_TIMEOUT = 300.0
PYTHON_TAGS = frozenset({"python", "py", "python3"})
SKIP_RE = re.compile(r"^#\s*no-exec\b")
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")


class BlockFormatError(Exception):
    """README 的围栏本身不合法（未闭合），没法可靠地切块。"""


@dataclass(frozen=True)
class Block:
    """一个 fenced code block；line 是开围栏那一行的行号（1-based）。"""

    path: Path
    line: int
    tag: str
    code: str

    @property
    def location(self) -> str:
        return f"{self.path.name}:{self.line}"


@dataclass(frozen=True)
class Result:
    block: Block
    status: str
    seconds: float
    reason: str = ""

    @property
    def location(self) -> str:
        return self.block.location


def _closes(match: re.Match[str], marker: str) -> bool:
    """闭围栏：同一种字符、不短于开围栏、且没有额外信息串。"""
    fence = match.group(2)
    return (
        fence[0] == marker[0]
        and len(fence) >= len(marker)
        and not match.group(3).strip()
    )


def extract_blocks(path: Path) -> list[Block]:
    """按行扫出全部围栏块。容忍缩进、```` 与 `~~~`、带属性的信息串。"""
    blocks: list[Block] = []
    marker: str | None = None
    tag = ""
    start = 0
    body: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        match = FENCE_RE.match(line)
        if marker is None:
            if match is not None:
                marker = match.group(2)
                info = match.group(3).strip().split(maxsplit=1)
                tag = info[0].lower() if info else ""
                start = number
                body = []
            continue
        if match is not None and _closes(match, marker):
            blocks.append(Block(path, start, tag, "\n".join(body)))
            marker = None
            continue
        body.append(line)
    if marker is not None:
        raise BlockFormatError(f"{path.name}:{start} 起的围栏没有闭合")
    return blocks


def skip_reason(block: Block) -> str | None:
    """块首行（跳过前导空行）是 `# no-exec` 就返回理由，否则 None。"""
    for line in block.code.splitlines():
        if not line.strip():
            continue
        if SKIP_RE.match(line.strip()):
            return line.strip()
        return None
    return None


def run_block(block: Block, timeout: float) -> Result:
    """在独立临时目录、独立子进程里执行一块；相对路径产物落在临时目录里。"""
    if not block.code.strip():
        return Result(block, "fail", 0.0, "空块：python 围栏里什么都没有")
    reason = skip_reason(block)
    if reason is not None:
        return Result(block, "skip", 0.0, reason)
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="readme-block-") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", block.code],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Result(
                block, "fail", time.monotonic() - started, f"超时（{timeout:.0f}s）"
            )
    seconds = time.monotonic() - started
    if proc.returncode == 0:
        return Result(block, "pass", seconds)
    output = (proc.stderr or proc.stdout).strip().splitlines()
    tail = "\n".join(output[-8:]) if output else "（没有输出）"
    return Result(block, "fail", seconds, f"exit {proc.returncode}\n{tail}")


def check_file(
    path: Path, timeout: float, list_only: bool, totals: dict[str, int]
) -> int:
    """检查一个文件，返回 0 / 1 / 2（含义同进程退出码）。"""
    if not path.is_file():
        print(f"FAIL {path.name}: 文件不存在")
        return 2
    try:
        blocks = extract_blocks(path)
    except BlockFormatError as exc:
        print(f"FAIL {exc}")
        return 2
    python_blocks = [block for block in blocks if block.tag in PYTHON_TAGS]
    print(f"{path.name}: {len(python_blocks)} 个 python 块 / 共 {len(blocks)} 个围栏块")
    if not python_blocks:
        print(f"FAIL {path.name}: 一个 python 块都没有 —— 这不算通过")
        return 2
    status = 0
    for block in python_blocks:
        if list_only:
            lines = len(block.code.splitlines())
            print(f"  {block.location:24s} [{block.tag}] {lines} 行")
            continue
        result = run_block(block, timeout)
        totals[result.status] += 1
        print(
            f"  {result.location:24s} {result.status.upper():4s} {result.seconds:6.1f}s"
        )
        if result.status == "pass":
            continue
        if result.status == "skip":
            print(f"      {result.reason}")
        else:
            status = max(status, 1)
            for line in result.reason.splitlines():
                print(f"      {line}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="抽取 README 里的 python 围栏块并逐个执行（# no-exec 显式跳过）"
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=list(DEFAULT_FILES),
        help=f"相对 --root 的文件名，默认：{' '.join(DEFAULT_FILES)}",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="仓库根目录")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"单块超时秒数（默认 {DEFAULT_TIMEOUT:.0f}）",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_only", help="只列块，不执行"
    )
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    names = args.files or list(DEFAULT_FILES)
    totals = {"pass": 0, "skip": 0, "fail": 0}
    status = 0
    for name in names:
        status = max(
            status, check_file(root / name, args.timeout, args.list_only, totals)
        )
    if args.list_only:
        return status
    print(
        f"summary: {totals['pass']} passed, {totals['skip']} skipped, {totals['fail']} failed"
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
