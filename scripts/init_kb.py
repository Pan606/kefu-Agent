"""知识库索引构建脚本（重建式，对应文档 6.5）。

用法：
    python scripts/init_kb.py
说明：
    - 读取 data/kb/*.md，按标题结构切分后向量化写入 Chroma（collection=kb）；
    - 幂等：每次执行先清空旧 collection 再重建。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import config, rag_engine  # noqa: E402


def main() -> None:
    count = rag_engine.build_index()
    print(f"[OK] 知识库索引构建完成：{count} 个 chunk → {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
