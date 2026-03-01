"""本地角色设定检索模块。"""

from .retriever import SearchHit, SearchRetriever
from .orchestrator import build_search_context

__all__ = ["SearchHit", "SearchRetriever", "build_search_context"]
