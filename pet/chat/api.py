"""聊天 Agent API 适配层。"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

from pet.config import ROOT_DIR
from pet.search import SearchRetriever, build_search_context

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - 新包未安装时回退旧包
    try:
        from duckduckgo_search import DDGS
    except ImportError:  # pragma: no cover - 运行环境未安装时降级
        DDGS = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - 运行环境未安装时降级
    OpenAI = None


class ChatAgentApi:
    """桌宠聊天 API 客户端。"""

    _SEED_TERMS = ("鸣潮", "爱弥斯", "设定", "背景")
    _ROLE_HINT_TERMS = {
        "爱弥斯",
        "鸣潮",
        "角色",
        "人设",
        "设定",
        "背景",
        "世界观",
        "身份",
        "经历",
        "关系",
        "性格",
        "故事",
        "档案",
        "传记",
        "台词",
        "喜好",
        "能力",
        "技能",
    }
    _TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,}")

    _SYSTEM_PROMPT = (
        "你是爱弥斯。请使用温柔、自然、简洁的中文对话。"
        "请优先依据给出的本地资料与联网搜索结果回答角色设定相关问题；"
        "若检索结果不足或不确定，请明确说明不确定。"
    )

    _DDG_RENAME_WARNING_PATTERN = r".*duckduckgo_search.*renamed to `ddgs`.*"
    _TIMELY_KEYWORDS = ("最新", "最近", "今天", "本周", "本月", "更新", "版本", "公告")
    _LOCAL_CONTEXT_MAX_CHARS = 1200
    _ONLINE_CONTEXT_MAX_CHARS = 1200
    _HISTORY_CONTEXT_MAX_CHARS = 800

    def __init__(self, model: str = "deepseek-chat", timeout_seconds: float = 20.0, top_k: int = 3):
        self._model = model
        self._timeout_seconds = float(timeout_seconds)
        self._top_k = max(1, int(top_k))
        self._retriever = SearchRetriever(Path(ROOT_DIR) / "pet" / "search" / "data")
        self._client = None
        self._client_api_key = ""

    def reply(self, user_message: str, history_records: list[str] | None = None) -> str:
        clean_user_message = str(user_message).strip()
        if not clean_user_message:
            return "你还没有输入内容。"

        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return "未检测到 DEEPSEEK_API_KEY，请先在环境变量中配置后再试。"
        if OpenAI is None:
            return "当前环境缺少 openai 依赖，请先执行 `pip install openai`。"

        retrieval_query = self._build_search_query(clean_user_message)
        local_context = self._build_local_search_context(retrieval_query, clean_user_message)
        local_context = self._truncate_text(local_context, self._LOCAL_CONTEXT_MAX_CHARS)

        should_use_online = self._should_use_online_search(clean_user_message, local_context)
        search_context = ""
        if should_use_online:
            search_context = self._build_online_search_context(retrieval_query, clean_user_message)
            search_context = self._truncate_text(search_context, self._ONLINE_CONTEXT_MAX_CHARS)

        messages = [{"role": "system", "content": self._SYSTEM_PROMPT}]
        if local_context:
            messages.append({"role": "system", "content": local_context})
        if search_context:
            messages.append({"role": "system", "content": search_context})

        if history_records:
            history_text = "\n".join(str(item).strip() for item in history_records if str(item).strip())
            if history_text:
                history_text = self._truncate_text(history_text, self._HISTORY_CONTEXT_MAX_CHARS)
                messages.append(
                    {
                        "role": "system",
                        "content": f"以下是最近对话历史与摘要，请保持语义连续：\n{history_text}",
                    }
                )

        messages.append({"role": "user", "content": clean_user_message})

        try:
            client = self._get_client(api_key)
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=False,
            )
        except Exception as exc:  # pragma: no cover - 依赖外部网络
            return f"调用 DeepSeek 失败：{exc}"

        if not response.choices:
            return "模型未返回有效结果，请稍后重试。"
        message = response.choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, list):
            merged_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = str(part.get("text", "")).strip()
                    if text:
                        merged_parts.append(text)
                else:
                    text = str(part).strip()
                    if text:
                        merged_parts.append(text)
            content = "\n".join(merged_parts)

        final_text = str(content).strip()
        if not final_text:
            return "模型返回为空，请稍后再试。"
        return final_text

    def _build_search_query(self, user_message: str) -> str:
        user_keywords = self._extract_user_role_keywords(user_message)
        parts: list[str] = [*self._SEED_TERMS, *user_keywords, user_message]
        deduplicated: list[str] = []
        for item in parts:
            clean = str(item).strip()
            if not clean or clean in deduplicated:
                continue
            deduplicated.append(clean)
        return " ".join(deduplicated)

    def _extract_user_role_keywords(self, user_message: str) -> list[str]:
        tokens = [token.lower() for token in self._TOKEN_RE.findall(str(user_message))]
        keywords: list[str] = []
        for token in tokens:
            clean = token.strip()
            if not clean:
                continue

            if clean in self._ROLE_HINT_TERMS:
                keywords.append(clean)
                continue

            if any(hint in clean for hint in ("设定", "背景", "角色", "人设", "经历", "身份", "世界观", "性格", "关系", "故事")):
                keywords.append(clean)

        unique_keywords: list[str] = []
        for item in keywords:
            if item not in unique_keywords:
                unique_keywords.append(item)
        return unique_keywords[:8]

    def _should_use_online_search(self, user_message: str, local_context: str) -> bool:
        if not local_context:
            return True
        clean_message = str(user_message).strip()
        if not clean_message:
            return False
        return any(keyword in clean_message for keyword in self._TIMELY_KEYWORDS)

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        raw = str(text).strip()
        if not raw:
            return ""
        if len(raw) <= max_chars:
            return raw
        return raw[:max_chars] + "\n...(已截断)"

    def _get_client(self, api_key: str):
        if self._client is not None and self._client_api_key == api_key:
            return self._client
        self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=self._timeout_seconds)
        self._client_api_key = api_key
        return self._client

    def _build_local_search_context(self, retrieval_query: str, user_message: str) -> str:
        try:
            search_hits = self._retriever.search(retrieval_query, top_k=self._top_k)
        except Exception as exc:  # pragma: no cover - 本地IO兜底
            return f"本地资料检索失败：{exc}"

        if not search_hits:
            return ""

        context = build_search_context(user_message, search_hits)
        if not context:
            return ""
        return f"以下为本地角色设定资料检索结果，请优先参考：\n{context}"

    def _build_online_search_context(self, retrieval_query: str, user_message: str) -> str:
        if DDGS is None:
            return "联网搜索不可用：缺少 duckduckgo-search 依赖。"

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=self._DDG_RENAME_WARNING_PATTERN,
                    category=RuntimeWarning,
                )
                with DDGS() as ddgs:
                    results = list(
                        ddgs.text(
                            keywords=retrieval_query,
                            region="cn-zh",
                            safesearch="moderate",
                            max_results=self._top_k,
                        )
                    )
        except Exception as exc:  # pragma: no cover - 依赖外网
            return f"联网搜索失败：{exc}"

        if not results:
            return f"用户问题：{user_message}\n检索词：{retrieval_query}\n未检索到有效网页结果。"

        lines: list[str] = [
            f"用户问题：{user_message}",
            f"检索词：{retrieval_query}",
            "以下为联网搜索结果，请优先基于这些结果回答：",
        ]
        for index, item in enumerate(results, start=1):
            title = str(item.get("title", "")).strip()
            body = str(item.get("body", "")).strip()
            if not (title or body):
                continue
            lines.append(f"[{index}] 标题：{title}")
            if body:
                lines.append(f"[{index}] 摘要：{body}")
        return "\n".join(lines)
