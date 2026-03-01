"""检索结果上下文拼装。"""

from __future__ import annotations

from .retriever import SearchHit


def build_search_context(query: str, hits: list[SearchHit]) -> str:
    clean_query = str(query).strip()
    if not hits:
        return ""

    lines: list[str] = [f"用户问题：{clean_query}", "以下为本地角色设定检索结果，请优先据此回答："]
    for index, hit in enumerate(hits, start=1):
        lines.append(f"[{index}] 来源：{hit.source}")
        lines.append(f"[{index}] 摘要：{hit.snippet}")
    return "\n".join(lines)
