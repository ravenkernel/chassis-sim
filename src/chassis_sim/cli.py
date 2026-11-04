"""命令行入口：`chassis-sim <subcommand>` 与 `python -m chassis_sim` 同一实现。

四个子命令（argparse 子解析器，不手搓 argv）：

    chassis-sim list-scenarios                 打印全部场景名，一行一个
    chassis-sim record --out demo.gif [--scenario S] [--controller pp|dwa]
                       [--steps N] [--every K] [--fps F] [--seed S]
    chassis-sim eval [--scenario S] [--controller pp|dwa] [--episodes N] [--seed S]
    chassis-sim render --out frame.png [--scenario S] [--seed S]

延迟导入纪律
------------
模块顶层只 import 标准库。env 本体（`chassis_sim.api` -> `envs` -> `bridges`）、
`record`/`render`（matplotlib + imageio）与 `eval` 全部在**函数体内**导入，于是
`import chassis_sim.cli` 与 `python -m chassis_sim --help` 都不拉重模块；缺可视化
依赖时只打印一条中文提示（含 `pip install "chassis-sim[viz]"`）并 exit 2，不吐
traceback。缺 planning extra（numba）时同样只有一条提示，但**只**提 planning
extra：两条提示各管各的，不互相串味。

默认场景
--------
`--scenario` 的默认值必须是 planning-free 的（不需要 numba）：只装 core
（`pip install chassis-sim`）的用户开箱跑 `chassis-sim eval` 得能成功。
`obstacles_dense` 这类 `requires: planning` 的场景只能显式 `--scenario` 指定。

退出码
------
0 = 成功；2 = 用法错误 / 缺依赖 / 未知场景或控制器（与 argparse 自己的用法错误
同一个码，脚本里 `chassis-sim ... || echo 失败` 一刀切即可）。
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 只为注解：不要为了类型提示在运行时拉起 env 本体
    from chassis_sim.envs import ChassisSimEnv

__all__ = ["main"]

#: 缺可视化依赖时给用户的那一句（`[project.optional-dependencies] viz`）
_VIZ_HINT = 'pip install "chassis-sim[viz]"'
_VIZ_MODULES: tuple[str, ...] = ("matplotlib", "imageio")
_CONTROLLERS: tuple[str, ...] = ("pp", "dwa")
_EXIT_OK = 0
_EXIT_USAGE = 2
#: 默认场景必须 planning-free（不需要 numba）：core 安装里 `eval` 要开箱可跑，
#: 而 `obstacles_dense` 这类场景 requires: planning，只能显式 --scenario 指定。
_DEFAULT_SCENARIO = "corridor"
#: record 的安全默认：省略 --steps/--every 时最多约 50 帧（env.max_steps 是 1000，
#: 每步一帧会渲染 1000 张 600px PNG —— 几分钟、几十 MB，不是能给的默认值）。
_DEFAULT_STEPS = 200
_DEFAULT_EVERY = 4


class _CliError(Exception):
    """可预期的用户错误：main() 打印一条消息后 exit 2，不吐 traceback。"""


# ── 前置检查（全部失败在动手之前，不留半截产物）─────────────────
def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _require_viz(who: str) -> None:
    """确认可视化依赖可用；缺了就抛 _CliError（消息带 pip 提示）。"""
    missing = [name for name in _VIZ_MODULES if not _module_available(name)]
    if missing:
        raise _CliError(
            f"{who}：需要 {' / '.join(missing)}；安装可视化依赖：{_VIZ_HINT}"
        )


def _check_scenario(name: str | None) -> str | None:
    """场景名必须是已知名字或已有的 YAML 路径，否则列出全部可用值。"""
    if name is None:
        return None
    from .scenarios import list_scenarios

    available = list_scenarios()
    if name in available or Path(name).is_file():
        return name
    raise _CliError(
        f"未知场景 {name!r}；可用场景：{', '.join(available)}"
        "（也可以直接给场景 YAML 的路径）"
    )


def _make_env(scenario: str | None, seed: int | None) -> ChassisSimEnv:
    """按场景/种子造一台 ackermann env（CLI 只开这一个车型）。"""
    from .api import make

    try:
        return make("ackermann", scenario=scenario, seed=seed)
    except LookupError as exc:
        raise _CliError(f"场景加载失败：{exc}") from None


def _make_controller(kind: str, env: ChassisSimEnv) -> Any:
    """用 env 自己的底盘造控制器（两侧同标定，evaluate 的一致性护栏才不炸）。"""
    from .controllers import make_controller

    return make_controller(kind, env=env)


def _positive_int(text: str) -> int:
    """argparse 的 type=：正整数，非法值由 argparse 自己 exit 2。"""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"需要整数，收到 {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"需要正整数，收到 {value}")
    return value


# ── 子命令 ────────────────────────────────────────────────────────
def _cmd_list_scenarios(args: argparse.Namespace) -> int:
    from .scenarios import list_scenarios

    for name in list_scenarios():
        print(name)
    return _EXIT_OK


def _cmd_record(args: argparse.Namespace) -> int:
    scenario = _check_scenario(args.scenario)
    _require_viz("record 子命令")
    env = _make_env(scenario, args.seed)
    try:
        controller = _make_controller(args.controller, env)
        from .recording import record

        result = record(
            env,
            controller,
            out=args.out,
            every=args.every,
            fps=args.fps,
            max_steps=args.steps,
            seed=args.seed,
        )
    finally:
        env.close()
    print(f"path: {result.path}")
    print(f"frames: {result.frames}")
    print(f"steps: {result.steps}")
    print(f"total_reward: {result.total_reward:.4f}")
    return _EXIT_OK


def _cmd_eval(args: argparse.Namespace) -> int:
    # 刻意**不**调 _require_viz：eval 只跑仿真 + 打印 Markdown，与 matplotlib /
    # imageio 无关（viz 只属于 record / render）。
    scenario = _check_scenario(args.scenario)
    env = _make_env(scenario, args.seed)
    try:
        controller = _make_controller(args.controller, env)
        from .eval import evaluate
        from .scenarios import scenario_reset_options_factory

        # 给了场景就每回合重掷一张独立地图（n 个回合才真是 n 个独立样本）；
        # 没给场景就不覆盖，地图由 env 自己按 seed 决定。
        reset_options = (
            scenario_reset_options_factory(scenario, base_seed=args.seed)
            if scenario is not None
            else None
        )
        result = evaluate(
            env,
            controller,
            episodes=args.episodes,
            seed=args.seed,
            reset_options=reset_options,
        )
    finally:
        env.close()
    print(result.markdown())
    return _EXIT_OK


def _cmd_render(args: argparse.Namespace) -> int:
    scenario = _check_scenario(args.scenario)
    _require_viz("render 子命令")
    env = _make_env(scenario, args.seed)
    try:
        env.reset(seed=args.seed, options=getattr(env, "default_reset_options", None))
        # 子命令沿用 render（给人敲的名字），函数叫 snapshot：chassis_sim.render
        # 是子包，同名顶层函数会被子模块导入重绑掉（见 recording 模块 docstring）。
        from .recording import snapshot

        path = snapshot(env, args.out)
    finally:
        env.close()
    print(path)
    return _EXIT_OK


# ── 解析器 ────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chassis-sim",
        description="chassis-sim 命令行：列场景 / 录 GIF / 批量评估 / 存 PNG",
        epilog=f"缺可视化依赖时按提示安装：{_VIZ_HINT}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scoped = sub.add_parser("list-scenarios", help="列出全部场景名（一行一个）")
    scoped.set_defaults(handler=_cmd_list_scenarios)

    recorded = sub.add_parser("record", help="跑一个回合并合成 GIF")
    recorded.add_argument(
        "--out", default="demo.gif", help="GIF 输出路径（默认 demo.gif）"
    )
    recorded.add_argument(
        "--scenario",
        default=_DEFAULT_SCENARIO,
        help=f"场景名（默认 {_DEFAULT_SCENARIO}）",
    )
    recorded.add_argument(
        "--controller",
        choices=_CONTROLLERS,
        default="pp",
        help="控制器（默认 pp；可选 pp / dwa）",
    )
    recorded.add_argument(
        "--steps",
        type=_positive_int,
        default=_DEFAULT_STEPS,
        help=(
            f"回合步数上限（默认 {_DEFAULT_STEPS} 步；正整数，0/负数由 argparse "
            "直接 exit 2，不会退化成 env.max_steps）"
        ),
    )
    recorded.add_argument(
        "--every",
        type=_positive_int,
        default=_DEFAULT_EVERY,
        help=f"每几步渲染一帧（默认每 {_DEFAULT_EVERY} 步一帧，正整数）",
    )
    recorded.add_argument(
        "--fps", type=_positive_int, default=12, help="GIF 帧率（默认 12）"
    )
    recorded.add_argument("--seed", type=int, default=0, help="复位种子（默认 0）")
    recorded.set_defaults(handler=_cmd_record)

    evaluated = sub.add_parser("eval", help="跑 N 个回合并打印 Markdown 表")
    evaluated.add_argument(
        "--scenario",
        default=_DEFAULT_SCENARIO,
        help=f"场景名（默认 {_DEFAULT_SCENARIO}）",
    )
    evaluated.add_argument(
        "--controller",
        choices=_CONTROLLERS,
        default="pp",
        help="控制器（默认 pp；可选 pp / dwa）",
    )
    evaluated.add_argument(
        "--episodes", type=_positive_int, default=5, help="回合数（默认 5）"
    )
    evaluated.add_argument("--seed", type=int, default=0, help="起始种子（默认 0）")
    evaluated.set_defaults(handler=_cmd_eval)

    rendered = sub.add_parser("render", help="把当前画面存成 PNG")
    rendered.add_argument(
        "--out", default="frame.png", help="PNG 输出路径（默认 frame.png）"
    )
    rendered.add_argument(
        "--scenario",
        default=_DEFAULT_SCENARIO,
        help=f"场景名（默认 {_DEFAULT_SCENARIO}）",
    )
    rendered.add_argument("--seed", type=int, default=0, help="复位种子（默认 0）")
    rendered.set_defaults(handler=_cmd_render)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：返回退出码（0 成功 / 2 用法错误或缺依赖、未知场景/控制器）。

    Args:
        argv: 参数序列，省略时用 `sys.argv[1:]`（argparse 的默认行为）。

    Returns:
        进程退出码。`--help` 与用法错误由 argparse 直接抛 SystemExit
        （0 / 2），与 console script 的约定一致。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    # required=True 时走不到这里：兜底给一条用法错误，而不是 AttributeError
    if handler is None:
        parser.print_usage(sys.stderr)
        return _EXIT_USAGE
    try:
        return handler(args)
    except _CliError as exc:
        print(f"[chassis-sim] 错误：{exc}", file=sys.stderr)
        return _EXIT_USAGE
    except ImportError as exc:  # 兜底：任何延迟导入缺依赖都只给一行提示
        # 自带安装提示的 ImportError（例如 planning 缺 numba）原样透出：再补一行
        # viz 提示会把"去装 matplotlib"写进一条 numba 的错里，方向就错了。
        hint = "" if "pip install" in str(exc) else f"\n  安装可视化依赖：{_VIZ_HINT}"
        print(f"[chassis-sim] 缺少依赖：{exc}{hint}", file=sys.stderr)
        return _EXIT_USAGE
