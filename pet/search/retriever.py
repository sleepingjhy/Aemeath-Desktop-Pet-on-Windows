"""本地角色设定检索器。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,}")


@dataclass(slots=True)
class SearchHit:
    source: str
    snippet: str
    score: float


class SearchRetriever:
    def __init__(self, data_dir: Path, max_doc_chars: int = 4000):
        self._data_dir = Path(data_dir)
        self._max_doc_chars = max(500, int(max_doc_chars))

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        normalized_query = str(query).strip()
        if not normalized_query:
            return []

        query_tokens = self._tokenize(normalized_query)
        if not query_tokens:
            return []

        docs = self._load_documents()
        scored: list[SearchHit] = []
        for source, content in docs:
            score = self._score(source, content, query_tokens)
            if score <= 0:
                continue
            snippet = self._build_snippet(content, query_tokens)
            scored.append(SearchHit(source=source, snippet=snippet, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(1, int(top_k))]

    def _load_documents(self) -> list[tuple[str, str]]:
        if not self._data_dir.exists() or not self._data_dir.is_dir():
            return []

        loaded: list[tuple[str, str]] = []
        for path in sorted(self._data_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in {".txt", ".md", ".json", ".yml", ".yaml"}:
                continue
            text = self._read_text(path)
            if not text:
                continue
            rel = path.relative_to(self._data_dir)
            loaded.append((str(rel).replace("\\", "/"), text[: self._max_doc_chars]))
        return loaded

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, (dict, list)):
                    return json.dumps(payload, ensure_ascii=False)
                return str(payload)
            return path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            return ""

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [part.lower() for part in _TOKEN_RE.findall(text)]

    @staticmethod
    def _score(source: str, content: str, query_tokens: Iterable[str]) -> float:
        text = content.lower()
        source_text = source.lower()
        score = 0.0
        for token in query_tokens:
            if token in source_text:
                score += 2.0
            count = text.count(token)
            if count <= 0:
                continue
            score += min(count, 6) * 1.0
        return score

    @staticmethod
    def _build_snippet(content: str, query_tokens: Iterable[str], max_len: int = 220) -> str:
        plain = " ".join(str(content).split())
        if not plain:
            return ""

        lower_plain = plain.lower()
        best_idx = -1
        for token in query_tokens:
            idx = lower_plain.find(token)
            if idx >= 0:
                best_idx = idx
                break

        if best_idx < 0:
            return plain[:max_len]

        start = max(0, best_idx - (max_len // 3))
        end = min(len(plain), start + max_len)
        snippet = plain[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(plain):
            snippet += "..."
        return snippet
