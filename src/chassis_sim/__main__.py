"""`python -m chassis_sim` 的入口：与 console script `chassis-sim` 共用 cli.main。

用法::

    python -m chassis_sim list-scenarios
    python -m chassis_sim record --steps 12 --every 4 --out demo.gif
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
