"""未安装时也能 import chassis_sim：把 <repo>/src 放到 sys.path 前端。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
