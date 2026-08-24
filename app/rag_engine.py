"""RAG 引擎：知识库向量化、检索、相关性过滤与引用组装（对应文档第 6 章）。

设计要点：
- 切分：单条 FAQ 整条不切分（### 标题为一个 chunk），段落过大用 RecursiveCharacter 兜底
  （对应文档 6.2，当前演示 FAQ 均为短问答，主用标题切分）；
- 存储：Chroma 本地持久化，collection「kefu_kb」，元数据记录来源文件与章节（6.3）；
- 检索：Top-K 与相似度阈值经 config 配置（6.4），按分数过滤后返回，附来源引用；
- 更新：重建式（重跑 init_kb.py，6.5）。
"""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter

from . import config


def _split_kb() -> list[dict]:
    """读取 data/kb/*.md，按标题结构切分为 chunk。

    Returns: [{"content": str, "metadata": {"source": 文件名, "section": 章节名}}]
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "section"), ("###", "question")],
        strip_headers=False,
    )
    chunks: list[dict] = []
    for md_file in sorted(config.KB_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        for doc in splitter.split_text(text):
            if not doc.page_content.strip():
                continue
            meta = dict(doc.metadata)
            section = meta.get("section", meta.get("h1", md_file.stem))
            chunks.append({
                "content": doc.page_content,
                "metadata": {"source": md_file.name, "section": section},
            })
    return chunks


def build_index() -> int:
    """重建知识库索引（幂等，删除旧 collection 后重建）。"""
    chunks = _split_kb()
    embeddings = config.get_embeddings()
    # 重建式更新：删除旧 collection
    collection = Chroma(
        collection_name=config.CHROMA_COLLECTION, embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR))
    try:
        if collection._collection.count() > 0:
            collection._collection.delete()
    except Exception:
        pass
    Chroma.from_texts(
        texts=[c["content"] for c in chunks],
        embedding=embeddings,
        metadatas=[c["metadata"] for c in chunks],
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
    )
    return len(chunks)


def get_retriever():
    """返回持久化的 Chroma 检索器（假设索引已构建）。"""
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=config.get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    ).as_retriever(search_kwargs={"k": config.RETRIEVAL_TOP_K})


def retrieve(question: str, top_k: int | None = None,
             threshold: float | None = None) -> list[dict]:
    """检索 + 相似度过滤 + 引用组装。

    Returns: [{"content": str, "score": float, "source": str, "section": str}]
    """
    top_k = top_k or config.RETRIEVAL_TOP_K
    threshold = threshold if threshold is not None else config.RETRIEVAL_THRESHOLD

    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=config.get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
    docs = vectorstore.similarity_search_with_relevance_scores(question, k=top_k)
    results = []
    for doc, score in docs:
        if score < threshold:
            continue
        results.append({
            "content": doc.page_content,
            "score": round(float(score), 4),
            "source": doc.metadata.get("source", ""),
            "section": doc.metadata.get("section", ""),
        })
    return results


def format_context(docs: list[dict]) -> str:
    """把检索结果拼成提示词上下文（附来源章节）。"""
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}]（来源：{d['section']}）\n{d['content']}")
    return "\n\n".join(parts)


def format_citations(docs: list[dict]) -> list[str]:
    """生成回复末尾的来源引用列表。"""
    seen = set()
    cites = []
    for d in docs:
        key = f"{d['source']}·{d['section']}"
        if key not in seen:
            seen.add(key)
            cites.append(key)
    return cites
