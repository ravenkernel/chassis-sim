"""`chassis_sim.scenarios` —— YAML 驱动的场景集（含解析式动态障碍）。

四个公开入口::

    from chassis_sim.scenarios import (
        list_scenarios,                # () -> list[str]，data/ 目录下的场景名
        load_scenario,                 # (name | path) -> Scenario（严格校验）
        scenario_to_reset_options,     # (scenario, *, seed=None) -> dict（喂 env.reset）
        scenario_reset_options_factory, # (name|Scenario, *, base_seed=0) -> 回调
    )

    # ① 固定地图：同一个 dict 喂 n 次 reset ⇒ 同一张图重放 n 次（CI 不可引用）
    scenario = load_scenario("corridor")
    obs, info = env.reset(seed=scenario.seed,
                          options=scenario_to_reset_options(scenario))

    # ② 每回合重掷地图（evaluate 的 n 才是 n 个独立样本）
    factory = scenario_reset_options_factory("obstacles_dense", base_seed=0)
    result = evaluate(make_env, controller, n_episodes=10, reset_options=factory)

YAML schema（**硬契约**；任何偏离都抛 `ScenarioValidationError`，消息里必带
文件路径与键名；未知键还会给出 `did you mean ...` 提示）::

    name: str                                # 必填；data/ 下按名字加载时须 == 文件名 stem
    description: str                         # 必填
    world_bounds: [xmin, xmax, ymin, ymax]   # 必填；xmin < xmax、ymin < ymax
    obstacles:                               # 必填，可为 []（planning 场景写 []）
      - type: circle                         #   x, y, radius(>0), theta(可选, 默认 0)
      - type: rectangle                      #   x, y, length(>0), width(>0), theta(可选)
    path:                                    # 必填
      type: straight | waypoints | stage
      params:                                #   按 type 取下列之一（键集合固定，多余键报错）
        straight:  {start: [x, y(, theta)], goal: [x, y(, theta)], spacing: > 0}
        waypoints: {points: [[x, y(, theta)], ...], spacing: > 0}   # >=2 点，列数一致
        stage:     {stage: <curriculum.yaml 里存在的 stage 名>}
    dynamic_obstacles:                       # 可选，默认 []；逐项按 type 分派
      - type: linear                         #   匀速直线：start + velocity·phase(t)
        start: [x, y]                        #   必填，t=0 时的圆心（米）
        velocity: [vx, vy]                   #   必填，米/秒；有限值（可为 0）
        radius: > 0                          #   必填，**本体半径**（米）
        period: >= 0                         #   可选，默认 0 = 不循环；> 0 = 周期（秒）
        pingpong: bool                       #   可选，默认 false；需 period > 0
      - type: circular                       #   匀速圆周：center + radius_orbit·(cos φ, sin φ)
        center: [x, y]                       #   必填，轨道圆心（米）
        radius_orbit: > 0                    #   必填，**轨道半径**（米）
        omega: rad/s                         #   必填，角速度；有限值（负 = 顺时针）
        phase: rad                           #   可选，默认 0；t=0 的初相位
        radius: > 0                          #   必填，**本体半径**（米）
    seed: int                                # 必填
    requires: planning | null                # 可选，默认 null；planning ⇔ path.type == stage

两类场景（`data/` 目录下共 11 个）：

* **core**（`requires: null`）：障碍 + 显式路径全部写在 YAML 里，纯 numpy 重采样，
  不需要 numba。当前 5 个：`straight` / `circle` / `corridor` / `dense_static` /
  `dynamic_crossing`（唯一带动态障碍的场景）。
* **planning**（`requires: planning`）：YAML 只声明课程阶段名，障碍与参考路径由
  `MapGenerator` 现场生成，因此 **YAML 里不重复写 count/spacing 等数字** ——
  单一真相源是 `config/curriculum.yaml` 的对应 stage。当前 6 个：
  `obstacles_sparse` / `obstacles_medium` / `obstacles_dense` /
  `obstacles_dense_reverse` / `foldback_reverse` / `maze_grid`。

`dynamic_obstacles` 是**纯运动学、零随机**的解析运动：位置只由场景字段与模拟时间
`t` 决定，因此"同 seed 同轨迹"是构造性成立的。运动学归桥（`SimpleBridge`）所有，
本包只负责数据模型、严格校验与 `to_env_dict()`；`scenario_to_reset_options` 只在
场景**确实带动态障碍**时追加第 4 个键 `dynamic_obstacles`（静态场景 payload 与
加入该功能前逐键相同）。
"""

from chassis_sim.scenarios.loader import (
    DATA_DIR,
    curriculum_stages,
    list_scenarios,
    load_curriculum,
    load_scenario,
)
from chassis_sim.scenarios.reset_options import (
    ScenarioResetOptionsFactory,
    scenario_reset_options_factory,
    scenario_to_reset_options,
)
from chassis_sim.scenarios.scenario import (
    CircleObstacle,
    CircularDynamicObstacle,
    DynamicObstacle,
    LinearDynamicObstacle,
    PathSpec,
    RectangleObstacle,
    Scenario,
    ScenarioError,
    ScenarioNotFoundError,
    ScenarioValidationError,
)

__all__ = [
    "DATA_DIR",
    "CircleObstacle",
    "CircularDynamicObstacle",
    "DynamicObstacle",
    "LinearDynamicObstacle",
    "PathSpec",
    "RectangleObstacle",
    "Scenario",
    "ScenarioError",
    "ScenarioNotFoundError",
    "ScenarioResetOptionsFactory",
    "ScenarioValidationError",
    "curriculum_stages",
    "list_scenarios",
    "load_curriculum",
    "load_scenario",
    "scenario_reset_options_factory",
    "scenario_to_reset_options",
]
