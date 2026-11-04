"""README 演示素材：三车型 GIF、两种控制器 GIF、六张场景静图。

用法（仓库根目录）::

    python examples/gallery.py --out docs/media              # 全部 11 个素材
    python examples/gallery.py --out docs/media --only gif   # 只出 5 段 GIF
    python examples/gallery.py --out docs/media --only png   # 只出 6 张场景静图
    python examples/gallery.py --out docs/media --dpi 120    # 静图降分辨率（下限 100）

产物规格：GIF 每段 TARGET_FRAMES(28) 帧、fps FPS(12)、画布 GIF_FIGSIZE x GIF_DPI；
6 张静图仍是 900x900（6x6 @ dpi 150）。单文件 <= 1.5 MB、11 个文件合计 <= 6.5 MB。

体积预算怎么花的（2026-11-04 实测，600x600 画布下扫帧数 x 调色板；括号内为
"5 段 GIF 合计 + 6 张静图 948,628 B"，两道上限都要过）—— 总计按 out_dir 中全部 11 个
素材统计，`--only gif` / `--only png` 也把目录里既有的另一类算进去（缺的跳过并按 0 计，
打印缺几个）：
600x600 + 36 帧 + 128 色 = 5 段共 7,230,286 B（不含静图）且 3 段破单文件上限 ⇒ 帧数是唯一
有效的旋钮。20/24/28/32 帧逐档实测（每档先 128 色，超预算降到 96 色）：
20/128 总计 5,079,888 B、24/128 5,871,522 B、28/128 6,640,367 B（超总计）、
32/128 7,411,693 B（超总计）、28/96 6,386,633 B（过）、32/96 7,123,266 B（超总计）。
最终取 GIF_FIGSIZE=(6.0,6.0)（600x600）+ 28 帧 + 96 色：5 段 5,438,005 B、
11 个文件 6,386,633 B、最大单段 1,190,791 B。相比 450x450/36 帧是拿帧数换分辨率
（画布 +78% 像素、帧数 −22%），帧数仍远高于"看得出幻灯片感"的 20 帧下限。
imageio 的 save_gif 不认 palettesize/subrectangles（2.37 实测静默忽略），拿不到这个量级，
所以 GIF 编码在本脚本内用 PIL 直写（含"与上一帧逐像素相同处留透明"的帧间差分）。

确定性：场景只由 scenario_to_reset_options(seed=0) 决定，控制器无随机，
PIL 量化无随机；"先跑一遍录动作、再 reset 一遍回放"两步的初态逐位相同。
渲染层写 PNG 时清掉 Software 元数据与时间戳、GIF 不含时间戳 ⇒ 同一条命令连跑两次，
11 个文件逐字节相同（_report 会把每个文件的 sha256 打出来）。临时帧目录用
tempfile.mkdtemp()，函数结束即删，不落在 docs/media 里。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

DPI = 150.0
FIGSIZE = (6.0, 6.0)
#: GIF 画布（英寸 x GIF_DPI）：600x600 是 README 门面尺寸，帧数/色数按它实测标定
GIF_FIGSIZE = (6.0, 6.0)
GIF_DPI = 100.0
#: 右下角缩放内嵌窗边长（米）：10x10 m 世界里车体只有 ~30 px，靠它看清四轮与车头
ZOOM = 2.0
FPS = 12
#: 每段 GIF 的帧数：28 帧 @ 12 fps ≈ 2.33 s 动画；低于 20 帧看得出幻灯片感，
#: 而 600x600 下 32 帧连 96 色都超总计预算（见模块 docstring 的实测表）
TARGET_FRAMES = 28
#: GIF 调色板颜色数（含 1 个透明索引）：600x600 + 28 帧必须降到 96 色才进总计 6.5 MB
GIF_COLORS = 96
SEED = 0
#: 单段回合的步数上限；到点/撞车的回合提前结束，剩下的步数不用凑
CLIP_STEPS = 120
MAX_GIF_BYTES = 1_500_000
MAX_TOTAL_BYTES = 6_500_000
MIN_DPI = 100.0
#: 展示素材用的两个场景：三车型共跑稀疏障碍（都能开到终点），控制器对照跑静态密障
CHASSIS_SCENARIO = "obstacles_sparse"
CONTROLLER_SCENARIO = "dense_static"

#: 控制器名 -> 写进 title 的显示名（title 由控制器/场景字段拼出来，不会与文件名漂开）
CONTROLLER_LABEL = {"pure_pursuit": "pure pursuit", "dwa": "dwa"}


class GalleryError(Exception):
    """素材生成失败（帧数不足、体积超限等），消息直接给用户看。"""


@dataclass(frozen=True)
class Clip:
    """一段 GIF 的全部参数：车型 x 控制器 x 场景 x seed 固定，别的都不允许漂。"""

    stem: str
    chassis: str
    controller: str
    scenario: str

    @property
    def title(self) -> str:
        """右上角附注：只写"控制器 · 场景"。

        车型名由渲染层自己打在上一行（figure.py 里 telemetry["chassis"]），
        这里再带一遍就会在右上角看到两行 ackermann。
        """
        return f"{CONTROLLER_LABEL[self.controller]} · {self.scenario}"


@dataclass(frozen=True)
class Made:
    """一个产物的实测参数，供末尾报告与体积检查使用。"""

    path: Path
    scenario: str
    chassis: str
    controller: str
    steps: int
    outcome: str
    frames: int
    every: float
    dpi: float


#: 五段 GIF：三车型同图同控制器同 seed，只换底盘；两种控制器同车型同 seed，只换控制器
CLIPS = (
    # 阿克曼：前轮转角受限，弯道必须外扩、贴着最小转弯半径走，车身姿态最能说明转向机构
    Clip("chassis_ackermann", "ackermann", "pure_pursuit", CHASSIS_SCENARIO),
    # 差速：没有横向自由度，靠内外轮速差转向；同一控制器下轨迹与麦轮几乎重合，正好对照
    Clip("chassis_diff_drive", "diff_drive", "pure_pursuit", CHASSIS_SCENARIO),
    # 麦轮：多一个正交速度分量，纯追踪不给 vy 时退化成差速 —— 图上用来标出"同一控制器"的边界
    Clip("chassis_mecanum", "mecanum", "pure_pursuit", CHASSIS_SCENARIO),
    # 纯追踪：单一前瞻圆弧，静态密障里一路压着路径开到终点星形
    Clip("controller_pure_pursuit", "ackermann", "pure_pursuit", CONTROLLER_SCENARIO),
    # DWA：在速度窗口里采样并避障，绕障碍更保守，同一张图与纯追踪逐帧对照
    Clip("controller_dwa", "ackermann", "dwa", CONTROLLER_SCENARIO),
)

#: 六张俯视静图：ackermann 停在起点（reset 后不 step），title 就是场景名
SCENES = (
    # 直道：最干净的基线，看没有障碍时参考路径、车体与终点标记各自长什么样
    "straight",
    # 走廊：两侧墙把可行域压成一条带子，能读出走廊宽度与车体两侧的余量
    "corridor",
    # 静态密障：14 个障碍挤在窄通道里，纯静态，与下面由地图生成器产出的密障对照
    "dense_static",
    # 规划器密障：obstacles_dense 现场生成，障碍更贴近参考路径
    "obstacles_dense",
    # 动态交叉：横穿路径的移动障碍用红虚线圆标出 t=0 位置，解释"再过几帧就会撞"
    "dynamic_crossing",
    # 稀疏障碍：5 个障碍散在大空间里，是控制器对比的低难度基线
    "obstacles_sparse",
)

#: 11 个固定素材文件名：_report 的“总计”与 --only all 的完整性检查都按它逐个取，
#: 与本次写了哪一类无关（--only gif 时 6 张静图只要在目录里就计入总计）
ASSET_NAMES = tuple(f"{clip.stem}.gif" for clip in CLIPS) + tuple(
    f"scene_{name}.png" for name in SCENES
)


def _import_viz() -> ModuleType | None:
    """导入 matplotlib(agg) 与 imageio；缺依赖时打印一句中文提示，返回 None。"""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import imageio  # noqa: F401  只检查在位
        from matplotlib import pyplot as plt
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        print(
            f"缺少可视化依赖 {missing}：本脚本需要 matplotlib 与 imageio，"
            "装好后重试（pip install -e '.[viz]'）。",
            file=sys.stderr,
        )
        return None
    return plt


def _make_controller(name: str, env) -> object:
    """按名字造控制器；必须用 env 的底盘构造，否则 limits 标定不同源。"""
    from chassis_sim.controllers import DwaController, PurePursuitController

    if name == "pure_pursuit":
        return PurePursuitController(chassis=env.chassis)
    if name == "dwa":
        return DwaController(chassis=env.chassis, dt=env.dt)
    raise GalleryError(f"未知控制器 {name!r}（支持 pure_pursuit / dwa）")


def _outcome(env, terminated: bool, truncated: bool, steps: int) -> str:
    """回合结果的一句话：到点 / 撞了 / 超时，三种都不藏。"""
    if terminated:
        if env.bridge.check_goal_reached(env.goal_threshold):
            return f"到点（第 {steps} 步）"
        return f"撞了（第 {steps} 步）"
    if truncated:
        return f"超时（第 {steps} 步，上限 {CLIP_STEPS}）"
    return f"跑满 {steps} 步"


def _pick_frames(frames: list[Path], want: int) -> list[Path]:
    """从逐帧 PNG 里等间隔抽 want 帧，首帧=第 1 步、末帧=回合最后一步。"""
    if len(frames) <= want:
        return list(frames)
    last = len(frames) - 1
    picked = sorted({round(index * last / (want - 1)) for index in range(want)})
    return [frames[index] for index in picked]


def _write_gif(frames: list[Path], target: Path) -> Path:
    """把 PNG 序列写成 GIF：GIF_COLORS 色、关 dither、与上一帧相同处留透明。

    渲染层 save_gif 走的 imageio 不认 palettesize（2.37 实测静默忽略），
    拿不到体积预算；这里用 PIL 直写：逐帧中位切分量化到 GIF_COLORS-1 色，
    再把与上一帧逐像素相同的像素置成透明索引，GIF 就不必为静止像素重复付费
    （实测同一批帧 1.76 MB -> 1.58 MB，600x600 下 2.08 MB -> 1.58 MB）。
    """
    from PIL import Image

    arrays = [np.asarray(Image.open(path).convert("RGB")) for path in frames]
    transparent = GIF_COLORS - 1
    images: list[Image.Image] = []
    for index, array in enumerate(arrays):
        quantized = Image.fromarray(array).quantize(
            colors=GIF_COLORS - 1,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        palette = quantized.getpalette()
        pixels = np.asarray(quantized).copy()
        if index:
            pixels[np.all(array == arrays[index - 1], axis=-1)] = transparent
        palette[3 * transparent : 3 * transparent + 3] = [0, 0, 0]
        image = Image.fromarray(pixels)
        image.putpalette(palette)
        images.append(image)
    # GIF 延时以 1/100 s 为单位（PIL 的 duration 收毫秒，内部只存 cs），12 fps 的 8.33 cs
    # 只能凑：按累计时间四舍五入到整数 cs 再乘 10。整段总时长 = round(100 x 帧数 / FPS) x 10 ms
    # （28 帧 = 2330 ms ⇒ 实际 12.02 fps，是 10 ms 粒度下最接近 28/12 s 的可表示值）
    delays = [
        10 * (round(100 * (index + 1) / FPS) - round(100 * index / FPS))
        for index in range(len(images))
    ]
    images[0].save(
        target,
        save_all=True,
        append_images=images[1:],
        duration=delays,
        loop=0,
        optimize=True,
        transparency=transparent,
        disposal=1,
    )
    return target


def _make_clip(clip: Clip, out_dir: Path, plt: ModuleType) -> Made:
    """先跑一遍录下控制器动作，再 reset 回放渲染抽帧，最后合成 GIF。"""
    import chassis_sim as cs
    from chassis_sim.eval import controller_observation
    from chassis_sim.render import render_episode_frames
    from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

    options = scenario_to_reset_options(load_scenario(clip.scenario), seed=SEED)
    env = cs.make(clip.chassis, max_steps=CLIP_STEPS)
    frame_dir = Path(tempfile.mkdtemp(prefix=f"chassis_sim_{clip.stem}_"))
    try:
        env.reset(seed=SEED, options=options)
        controller = _make_controller(clip.controller, env)
        controller.set_path(np.asarray(env.bridge.get_path_global(), dtype=np.float64))
        actions: list[np.ndarray] = []
        terminated = truncated = False
        for _ in range(CLIP_STEPS):
            action = controller.act(controller_observation(env))
            _, _, terminated, truncated, _ = env.step(action)
            actions.append(np.asarray(action, dtype=np.float64))
            if terminated or truncated:
                break
        steps = len(actions)
        outcome = _outcome(env, terminated, truncated, steps)
        # 同一 seed 再 reset ⇒ 初态逐位相同，回放的动作序列就是刚才那段真实控制
        env.reset(seed=SEED, options=options)
        # every=1 ⇒ 每步一帧，文件名自带步号（frame_00011.png），末帧必是回合末步
        rendered = render_episode_frames(
            env.bridge,
            actions,
            frame_dir,
            every=1,
            show_lidar=True,
            show_trail=True,
            dpi=GIF_DPI,
            figsize=GIF_FIGSIZE,
            zoom=ZOOM,
            title=clip.title,
        )
        frames = _pick_frames(rendered, TARGET_FRAMES)
        if len(frames) < 2:
            raise GalleryError(f"{clip.stem}: 只渲染出 {len(frames)} 帧，凑不出动画")
        print(
            f"       {len(actions)} 步 -> {len(frames)} 帧 "
            f"{frames[0].stem} … {frames[-1].stem}（末帧步号 {frames[-1].stem[-5:]}）"
        )
        gif = _write_gif(frames, out_dir / f"{clip.stem}.gif")
        every = (steps - 1) / (len(frames) - 1) if len(frames) > 1 else 1.0
        return Made(
            path=gif,
            scenario=clip.scenario,
            chassis=clip.chassis,
            controller=clip.controller,
            steps=steps,
            outcome=outcome,
            frames=len(frames),
            every=every,
            dpi=GIF_DPI,
        )
    finally:
        plt.close("all")
        env.close()
        shutil.rmtree(frame_dir, ignore_errors=True)


def _draw_dynamic(bridge, ax) -> None:
    """把移动障碍的 t=0 圆画成红虚线（渲染层只画静态障碍，这里补一笔）。"""
    from matplotlib.patches import Circle

    getter = getattr(bridge, "dynamic_positions", None)
    if getter is None:
        return
    moved = np.asarray(getter(), dtype=np.float64)
    if moved.size == 0:
        return
    for x, y, radius in moved.reshape(-1, 3):
        ax.add_patch(
            Circle(
                (float(x), float(y)),
                float(radius),
                fill=False,
                linestyle=(0, (3.0, 2.0)),
                linewidth=1.4,
                edgecolor="#d1495b",
                zorder=6,
            )
        )


def _make_scene_png(name: str, out_dir: Path, plt: ModuleType, dpi: float) -> Made:
    """场景静图：ackermann 停在起点（reset 后不 step），带激光与缩放内嵌窗。"""
    import chassis_sim as cs
    from chassis_sim.render import render_frame, save_figure
    from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

    options = scenario_to_reset_options(load_scenario(name), seed=SEED)
    env = cs.make("ackermann")
    try:
        env.reset(seed=SEED, options=options)
        fig = render_frame(
            env.bridge,
            show_lidar=True,
            dpi=dpi,
            figsize=FIGSIZE,
            title=name,  # 只写场景名：渲染层会自己把车型名放到上一行
            zoom=ZOOM,
        )
        _draw_dynamic(env.bridge, fig.axes[0])
        target = save_figure(fig, out_dir / f"scene_{name}.png", dpi=dpi)
        return Made(
            path=target,
            scenario=name,
            chassis="ackermann",
            controller="—",
            steps=0,
            outcome="停在起点（reset 后未 step）",
            frames=1,
            every=1.0,
            dpi=dpi,
        )
    finally:
        plt.close("all")
        env.close()


def _report(made: list[Made], out_dir: Path, only: str) -> None:
    """打印每个产物的实测参数与 sha256，并检查单文件与总计的体积上限。

    体积口径：两个上限都按 out_dir 里**最终存在**的 11 个固定素材逐个取 st_size 判定，
    缺的跳过并按 0 计、打印缺几个。不用本次写入的清单——否则 `--only gif` 会把目录里
    既有的 6 张静图漏掉，11 文件口径的总计上限被误判成 PASS。
    """
    import hashlib

    print()
    print(f"=== 产物（{out_dir}） ===")
    oversize: list[str] = []
    for item in sorted(made, key=lambda m: m.path.name):
        size = item.path.stat().st_size
        print(
            f"{item.path.name:30s} {size:9d} B  "
            f"{item.frames:2d} 帧 fps {FPS} every {item.every:.2f} dpi {item.dpi:.0f}  "
            f"{item.scenario}/{item.chassis}/{item.controller}  "
            f"{item.steps:3d} 步  {item.outcome}"
        )
        print(f"{'':30s} sha256 {hashlib.sha256(item.path.read_bytes()).hexdigest()}")

    total = 0
    missing: list[str] = []
    for name in ASSET_NAMES:
        path = out_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        size = path.stat().st_size
        total += size
        if path.suffix == ".gif" and size > MAX_GIF_BYTES:
            oversize.append(f"{name} {size} B > {MAX_GIF_BYTES} B")
    print(
        f"{'合计':30s} {total:9d} B / 上限 {MAX_TOTAL_BYTES} B"
        f"（按 {len(ASSET_NAMES)} 个固定素材统计，缺 {len(missing)} 个）"
    )
    if missing:
        print(f"{'':30s} 缺：{'、'.join(missing)}")
    print(
        f"判定：单 GIF {'全部 <= ' + str(MAX_GIF_BYTES) + ' B' if not oversize else '超限'}；"
        f"总计 {total} B {'<= ' if total <= MAX_TOTAL_BYTES else '> '}"
        f"{MAX_TOTAL_BYTES} B"
    )
    if oversize:
        raise GalleryError(
            "以下 GIF 超过单文件上限，请先降 GIF_FIGSIZE 再考虑加色数：\n  "
            + "\n  ".join(oversize)
        )
    if total > MAX_TOTAL_BYTES:
        raise GalleryError(
            f"合计 {total} B 超过上限 {MAX_TOTAL_BYTES} B，请降 GIF_FIGSIZE"
        )
    if only == "all":
        expected = set(ASSET_NAMES)
        actual = {path.name for path in out_dir.iterdir() if path.is_file()}
        if actual != expected:
            raise GalleryError(
                f"{out_dir} 里的文件不是预期的 11 个：多出 {sorted(actual - expected)}，"
                f"缺少 {sorted(expected - actual)}"
            )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="重生 README 演示素材（5 段 GIF + 6 张静图）"
    )
    parser.add_argument(
        "--out", default="docs/media", help="产物目录（默认 docs/media）"
    )
    parser.add_argument(
        "--only",
        choices=("all", "gif", "png"),
        default="all",
        help="只重生某一类素材（默认全部）",
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=DPI,
        help=f"静图分辨率（默认 {DPI:.0f}，下限 {MIN_DPI:.0f}）；GIF 画布由体积预算固定",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plt = _import_viz()
    if plt is None:
        return 2
    dpi = max(MIN_DPI, float(args.dpi))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[Made] = []
    try:
        if args.only in ("all", "gif"):
            for clip in CLIPS:
                print(f"[gif] {clip.stem} ← {clip.title}")
                made.append(_make_clip(clip, out_dir, plt))
        if args.only in ("all", "png"):
            for name in SCENES:
                print(f"[png] scene_{name}")
                made.append(_make_scene_png(name, out_dir, plt, dpi))
        _report(made, out_dir, args.only)
    except GalleryError as exc:
        print(f"素材生成失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
