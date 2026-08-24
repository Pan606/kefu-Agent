"""智能客服 Agent 启动入口。

用法：
    python main.py              # 启动 Web 服务 → http://127.0.0.1:8761
    python main.py --init       # 先重建知识库索引再启动
    python main.py --seed       # 先重新生成演示数据（FAQ + 模拟订单）再启动
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import memory, observability  # noqa: E402


def main() -> None:
    args = set(sys.argv[1:])
    if "--seed" in args:
        from scripts import seed_data
        seed_data.main()
    if "--init" in args:
        from scripts import init_kb
        init_kb.main()
    if "--server" in args or "--seed" in args or "--init" in args:
        from app.web_server import main as server_main
        server_main()
    else:
        memory.init_db()
        observability.init_observability()
        print("用法：")
        print("  python main.py           启动 Web 服务（http://127.0.0.1:8761）")
        print("  python main.py --init    重建知识库索引后启动")
        print("  python main.py --seed    重新生成演示数据后启动")
        print("  python tests/eval_set.py --run   运行评估集")


if __name__ == "__main__":
    main()
