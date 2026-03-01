"""聊天会话状态管理。"""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from PySide6.QtCore import QObject, QThread, Signal, Slot

from .api import ChatAgentApi

Role = Literal["pet", "player"]
MessageKind = Literal["text", "image", "rich"]


class _ReplyWorker(QObject):
    """后台线程中的 API 调用执行器。"""

    finished = Signal(int, str)

    def __init__(self, request_id: int, api: ChatAgentApi, user_input: str, history_records: list[str]):
        super().__init__()
        self._request_id = int(request_id)
        self._api = api
        self._user_input = user_input
        self._history_records = history_records

    @Slot()
    def run(self):
        try:
            reply_text = self._api.reply(self._user_input, history_records=self._history_records)
        except Exception as exc:  # pragma: no cover - 兜底保护
            reply_text = f"调用 DeepSeek 失败：{exc}"
        self.finished.emit(self._request_id, str(reply_text).strip())


@dataclass(slots=True)
class _PendingReply:
    conversation_id: str
    user_input: str
    history_records: list[str]


@dataclass(slots=True)
class ChatConversation:
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessage] = field(default_factory=list)


@dataclass(slots=True)
class ChatMessage:
    role: Role
    kind: MessageKind
    content: str
    timestamp: datetime
    record: str = ""


class ChatSession(QObject):
    """共享聊天会话。支持多视图同步与关闭时清理缓存。"""

    _RECENT_TURNS = 4
    _SUMMARY_MAX_ITEMS = 6
    _SUMMARY_ITEM_MAX_CHARS = 42

    message_added = Signal(str, object)
    session_cleared = Signal()
    conversation_list_changed = Signal()
    active_conversation_changed = Signal(str)

    def __init__(self, api: ChatAgentApi | None = None, parent=None):
        super().__init__(parent)
        self._api = api or ChatAgentApi()
        self._conversations: dict[str, ChatConversation] = {}
        self._conversation_order: list[str] = []
        self._current_conversation_id = ""
        self._pending_replies: list[_PendingReply] = []
        self._request_counter = 0
        self._active_request_id: int | None = None
        self._request_to_conversation: dict[int, str] = {}
        self._reply_thread: QThread | None = None
        self._reply_worker: _ReplyWorker | None = None

        default_id = self.create_conversation(title="对话 1")
        self.switch_conversation(default_id)

    @property
    def messages(self) -> list[ChatMessage]:
        conversation = self._conversations.get(self._current_conversation_id)
        if conversation is None:
            return []
        return list(conversation.messages)

    @property
    def current_conversation_id(self) -> str:
        return self._current_conversation_id

    def create_conversation(self, title: str | None = None) -> str:
        next_index = len(self._conversation_order) + 1
        now = datetime.now()
        conversation_id = uuid4().hex
        conversation = ChatConversation(
            conversation_id=conversation_id,
            title=str(title or f"对话 {next_index}").strip() or f"对话 {next_index}",
            created_at=now,
            updated_at=now,
        )
        self._conversations[conversation_id] = conversation
        self._conversation_order.insert(0, conversation_id)
        self.conversation_list_changed.emit()
        return conversation_id

    def switch_conversation(self, conversation_id: str) -> bool:
        target_id = str(conversation_id).strip()
        if not target_id or target_id not in self._conversations:
            return False
        if target_id == self._current_conversation_id:
            return True
        self._current_conversation_id = target_id
        self.active_conversation_changed.emit(target_id)
        return True

    def delete_conversation(self, conversation_id: str) -> bool:
        target_id = str(conversation_id).strip()
        if not target_id or target_id not in self._conversations:
            return False

        self._conversations.pop(target_id, None)
        self._conversation_order = [item for item in self._conversation_order if item != target_id]
        self._pending_replies = [item for item in self._pending_replies if item.conversation_id != target_id]

        request_ids_to_remove = [
            request_id
            for request_id, mapped_conversation_id in self._request_to_conversation.items()
            if mapped_conversation_id == target_id
        ]
        for request_id in request_ids_to_remove:
            self._request_to_conversation.pop(request_id, None)
            if self._active_request_id == request_id:
                self._active_request_id = None

        if not self._conversation_order:
            fallback_id = self.create_conversation(title="对话 1")
            self.switch_conversation(fallback_id)
            self.session_cleared.emit()
            return True

        if self._current_conversation_id == target_id:
            self.switch_conversation(self._conversation_order[0])

        self.conversation_list_changed.emit()
        return True

    def list_conversations(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for conversation_id in self._conversation_order:
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                continue
            result.append(
                {
                    "id": conversation.conversation_id,
                    "title": conversation.title,
                    "updated_at": conversation.updated_at,
                    "message_count": len(conversation.messages),
                }
            )
        return result

    def _append_player_and_reply(self, kind: MessageKind, content: str, record_text: str):
        clean_content = str(content).strip()
        clean_record = str(record_text).strip()
        if not clean_content and not clean_record:
            return

        conversation_id = self._current_conversation_id
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return

        now = datetime.now()
        player_message = ChatMessage(
            role="player",
            kind=kind,
            content=clean_content,
            timestamp=now,
            record=clean_record,
        )
        conversation.messages.append(player_message)
        conversation.updated_at = now
        self._touch_conversation_order(conversation_id)
        self.message_added.emit(conversation_id, player_message)
        self.conversation_list_changed.emit()

        reply_input = clean_record or clean_content
        history_records = self._build_api_history_records(conversation_id)
        self._enqueue_reply_request(conversation_id, reply_input, history_records)

    def _enqueue_reply_request(self, conversation_id: str, reply_input: str, history_records: list[str]):
        self._pending_replies.append(
            _PendingReply(
                conversation_id=conversation_id,
                user_input=reply_input,
                history_records=history_records,
            )
        )
        self._start_next_reply_request_if_idle()

    def _start_next_reply_request_if_idle(self):
        if self._active_request_id is not None:
            return
        if not self._pending_replies:
            return

        next_reply = self._pending_replies.pop(0)
        self._start_reply_request(next_reply.conversation_id, next_reply.user_input, next_reply.history_records)

    def _start_reply_request(self, conversation_id: str, reply_input: str, history_records: list[str]):
        if self._active_request_id is not None:
            return

        self._request_counter += 1
        request_id = self._request_counter
        self._active_request_id = request_id
        self._request_to_conversation[request_id] = conversation_id

        thread = QThread(self)
        worker = _ReplyWorker(request_id=request_id, api=self._api, user_input=reply_input, history_records=history_records)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_reply_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_reply_thread_finished)

        self._reply_thread = thread
        self._reply_worker = worker
        thread.start()

    @Slot(int, str)
    def _on_reply_finished(self, request_id: int, reply_text: str):
        if self._active_request_id != int(request_id):
            return
        self._active_request_id = None
        final_reply = str(reply_text).strip() or "模型返回为空，请稍后再试。"
        target_conversation_id = self._request_to_conversation.pop(int(request_id), "")
        if target_conversation_id:
            self._append_pet_text(target_conversation_id, final_reply)
        self._start_next_reply_request_if_idle()

    @Slot()
    def _on_reply_thread_finished(self):
        if self._active_request_id is not None:
            request_id = self._active_request_id
            self._active_request_id = None
            target_conversation_id = self._request_to_conversation.pop(int(request_id), "")
            if target_conversation_id:
                self._append_pet_text(target_conversation_id, "请求异常中断，请重试。")
        self._reply_thread = None
        self._reply_worker = None
        self._start_next_reply_request_if_idle()

    def _append_pet_text(self, conversation_id: str, text: str):
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return
        now = datetime.now()
        pet_reply = ChatMessage(
            role="pet",
            kind="text",
            content=text,
            timestamp=now,
            record=text,
        )
        conversation.messages.append(pet_reply)
        conversation.updated_at = now
        self._touch_conversation_order(conversation_id)
        self.message_added.emit(conversation_id, pet_reply)
        self.conversation_list_changed.emit()

    def _touch_conversation_order(self, conversation_id: str):
        if conversation_id in self._conversation_order:
            self._conversation_order.remove(conversation_id)
        self._conversation_order.insert(0, conversation_id)

    def _build_api_history_records(self, conversation_id: str) -> list[str]:
        conversation = self._conversations.get(conversation_id)
        if conversation is None or not conversation.messages:
            return []

        serialized: list[str] = []
        for message in conversation.messages:
            text = str(message.record or message.content).strip()
            if not text:
                continue
            speaker = "玩家" if message.role == "player" else "爱弥斯"
            serialized.append(f"{speaker}：{text}")

        if not serialized:
            return []

        recent_message_count = self._RECENT_TURNS * 2
        if len(serialized) <= recent_message_count:
            return serialized

        older = serialized[:-recent_message_count]
        recent = serialized[-recent_message_count:]
        summary = self._summarize_older_records(older)

        history_payload: list[str] = []
        if summary:
            history_payload.append(f"历史摘要：{summary}")
        history_payload.append("最近4轮对话：")
        history_payload.extend(recent)
        return history_payload

    def _summarize_older_records(self, older_records: list[str]) -> str:
        if not older_records:
            return ""

        selected = older_records[-self._SUMMARY_MAX_ITEMS :]
        compact_parts: list[str] = []
        for item in selected:
            clipped = item.strip()
            if len(clipped) > self._SUMMARY_ITEM_MAX_CHARS:
                clipped = clipped[: self._SUMMARY_ITEM_MAX_CHARS] + "..."
            compact_parts.append(clipped)
        return "；".join(compact_parts)

    def send_text(self, text: str):
        clean_text = str(text).strip()
        if not clean_text:
            return
        self._append_player_and_reply("text", clean_text, clean_text)

    def send_image(self, image_path: str):
        clean_path = str(image_path).strip()
        if not clean_path:
            return
        record = f"[图片:{Path(clean_path).name}]"
        self._append_player_and_reply("image", clean_path, record)

    def send_composed(self, display_html: str, record_text: str):
        self._append_player_and_reply("rich", display_html, record_text)

    def dispose(self):
        self._pending_replies.clear()
        self._request_to_conversation.clear()
        self._active_request_id = None

        thread = self._reply_thread
        if thread is not None:
            try:
                thread.quit()
                thread.wait(1200)
                if thread.isRunning():
                    thread.terminate()
                    thread.wait(300)
            except Exception:
                pass

        self._reply_worker = None
        self._reply_thread = None
        self._conversations.clear()
        self._conversation_order.clear()
        self._current_conversation_id = ""

    def clear(self):
        self._active_request_id = None
        self._request_to_conversation.clear()
        self._pending_replies.clear()
        self._conversations.clear()
        self._conversation_order.clear()
        self._current_conversation_id = ""
        default_id = self.create_conversation(title="对话 1")
        self.switch_conversation(default_id)
        self.session_cleared.emit()
