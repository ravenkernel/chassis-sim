"""Gymnasium 环境注册表：`import chassis_sim` 即注册三个车型 id。

只依赖 gymnasium：环境本体用 entry_point **字符串**延迟 import，于是
`import chassis_sim` 不会因为注册而拉起 numba / matplotlib / torch 等可选重依赖
（tests/test_import_purity.py 在干净子进程里守着这条边界）。

注册的 kwargs 只有 chassis，观测口径（obs_spec / sensors）走 gymnasium 的常规
覆盖通道：`gym.make("ChassisSim-Ackermann-v0", obs_spec=...) ` 里的多余关键字
会被**原样传给构造器**，因此对象参数（ObsSpec、传感器 spec 列表）不需要也不该
塞进注册表（注册表是全局的，对象参数一旦写进去就变成进程级共享状态）。
默认观测是最小集 128 / 128 / 129 维。
"""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

__all__ = ["ENTRY_POINT", "ENV_IDS", "register_envs"]

ENTRY_POINT = "chassis_sim.envs:ChassisSimEnv"

#: gymnasium id -> 传给 ChassisSimEnv 的车型名（与 chassis/registry.py 同名）
ENV_IDS: dict[str, str] = {
    "ChassisSim-Ackermann-v0": "ackermann",
    "ChassisSim-DiffDrive-v0": "diff_drive",
    "ChassisSim-Mecanum-v0": "mecanum",
}


def register_envs() -> list[str]:
    """注册三个车型 id（幂等：已注册的跳过），返回本次新注册的 id。

    max_episode_steps 保持 None：回合长度由 env 自己的 max_steps（超时**终止**）
    决定。再套一层 TimeLimit 只会把 "env 判超时" 变成 "wrapper 截断"，
    info['timeout'] 与评测器的超时率会因此失真。
    """
    registered: list[str] = []
    for env_id, chassis in ENV_IDS.items():
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point=ENTRY_POINT,
            kwargs={"chassis": chassis},
            max_episode_steps=None,
        )
        registered.append(env_id)
    return registered


register_envs()
