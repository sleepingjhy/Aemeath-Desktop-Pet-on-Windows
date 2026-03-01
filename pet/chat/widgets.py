"""聊天 UI 组件。"""

from __future__ import annotations

import json
import os
import re
import warnings
import html as html_lib
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from PySide6.QtCore import Qt, QTimer, QRectF, QEvent, QEasingCurve, QPropertyAnimation, QParallelAnimationGroup, QPoint, Signal, QSize
from PySide6.QtGui import QPainter, QPainterPath, QPixmap, QColor, QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import APP_NAME, ROOT_DIR
from .session import ChatMessage, ChatSession

PET_AVATAR_PATH = Path(ROOT_DIR) / "gifs" / "photo.jpg"
PLAYER_AVATAR_PATH = Path(ROOT_DIR) / "gifs" / "player.png"
CHAT_BACKGROUND_PATH = Path(ROOT_DIR) / "gifs" / "chat_background.png"
EMOJI_ASSETS_ROOT = Path(ROOT_DIR) / "gifs" / "assets"


def _safe_disconnect(signal, slot):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            signal.disconnect(slot)
        except Exception:
            pass


class EmojiPickerPopup(QWidget):
    """微信表情选择弹层。"""

    emoji_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self._emoji_map: dict[str, Path] = {}
        self._recent_names: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title = QLabel("微信表情")
        title.setObjectName("EmojiPickerTitle")
        root.addWidget(title)

        recent_title_row = QWidget()
        recent_title_layout = QHBoxLayout(recent_title_row)
        recent_title_layout.setContentsMargins(0, 0, 0, 0)
        recent_title_layout.setSpacing(6)

        self.recent_title = QLabel("最近使用")
        self.recent_title.setObjectName("EmojiRecentTitle")

        self.clear_recent_btn = QToolButton()
        self.clear_recent_btn.setObjectName("EmojiClearBtn")
        self.clear_recent_btn.setText("🗑")
        self.clear_recent_btn.setToolTip("清空最近使用")
        self.clear_recent_btn.setFixedSize(24, 24)
        self.clear_recent_btn.clicked.connect(self._clear_recent_names)

        recent_title_layout.addWidget(self.recent_title, stretch=0)
        recent_title_layout.addStretch(1)
        recent_title_layout.addWidget(self.clear_recent_btn, stretch=0)
        root.addWidget(recent_title_row)

        self.recent_row = QWidget()
        self.recent_layout = QHBoxLayout(self.recent_row)
        self.recent_layout.setContentsMargins(0, 0, 0, 0)
        self.recent_layout.setSpacing(6)
        root.addWidget(self.recent_row)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("EmojiPickerScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.grid_holder = QWidget()
        self.grid_holder.setObjectName("EmojiGridHolder")
        self.grid_layout = QGridLayout(self.grid_holder)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(6)
        self.grid_layout.setVerticalSpacing(6)

        self.scroll.setWidget(self.grid_holder)
        root.addWidget(self.scroll, stretch=1)

        self.setFixedSize(380, 280)

        self.setStyleSheet(
            """
            EmojiPickerPopup {
                background: rgba(255, 245, 250, 0.96);
                border: 1px solid #f1c9dd;
                border-radius: 10px;
            }
            QLabel#EmojiPickerTitle {
                color: #2a2a2a;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#EmojiRecentTitle {
                color: #2a2a2a;
                font-size: 12px;
                font-weight: 600;
            }
            QToolButton#EmojiClearBtn {
                background: rgba(255, 230, 241, 0.92);
                border: 1px solid #efbfd6;
                border-radius: 12px;
                color: #333333;
                font-size: 12px;
            }
            QToolButton#EmojiClearBtn:hover {
                background: #ffd8ea;
            }
            QToolButton#EmojiItemBtn {
                background: rgba(255, 255, 255, 0.85);
                border: 1px solid #f0d6e3;
                border-radius: 6px;
                padding: 2px;
            }
            QToolButton#EmojiItemBtn:hover {
                background: #ffe6f1;
                border-color: #efbfd6;
            }
            QScrollArea#EmojiPickerScroll {
                background: rgba(255, 238, 246, 0.95);
                border: 1px solid #f1c9dd;
                border-radius: 8px;
            }
            QScrollArea#EmojiPickerScroll > QWidget > QWidget {
                background: rgba(255, 238, 246, 0.95);
            }
            QWidget#EmojiGridHolder {
                background: rgba(255, 238, 246, 0.95);
                border-radius: 6px;
            }
            QToolTip {
                background: #ffe6f1;
                color: #222222;
                border: 1px solid #f3bfd8;
                padding: 3px 6px;
            }
            """
        )

        self._build_emoji_grid()
        self._load_recent_names()
        self._refresh_recent_row()

    def _build_cache_path(self) -> Path:
        appdata = os.getenv("APPDATA")
        if appdata:
            base_dir = Path(appdata) / APP_NAME
        else:
            base_dir = Path.home() / f".{APP_NAME.lower()}"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / "recent_emojis.json"

    def _load_recent_names(self):
        cache_path = self._build_cache_path()
        if not cache_path.exists():
            self._recent_names = []
            return
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._recent_names = [str(item) for item in data if isinstance(item, str)]
            else:
                self._recent_names = []
        except Exception:
            self._recent_names = []

        self._recent_names = [name for name in self._recent_names if name in self._emoji_map][:10]
        self._save_recent_names()

    def _save_recent_names(self):
        cache_path = self._build_cache_path()
        try:
            cache_path.write_text(json.dumps(self._recent_names[:10], ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _clear_recent_names(self):
        self._recent_names = []
        self._save_recent_names()
        self._refresh_recent_row()

    def _push_recent_name(self, emoji_name: str):
        clean_name = str(emoji_name).strip()
        if not clean_name:
            return

        filtered = [name for name in self._recent_names if name != clean_name]
        self._recent_names = [clean_name] + filtered
        self._recent_names = self._recent_names[:10]
        self._save_recent_names()
        self._refresh_recent_row()

    def _refresh_recent_row(self):
        for i in reversed(range(self.recent_layout.count())):
            item = self.recent_layout.takeAt(i)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self._recent_names:
            empty = QLabel("暂无")
            empty.setStyleSheet("color:#666666; font-size:12px;")
            self.recent_layout.addWidget(empty)
            self.recent_layout.addStretch(1)
            return

        for emoji_name in self._recent_names[:10]:
            path = self._emoji_map.get(emoji_name)
            if path is None:
                continue

            btn = QToolButton()
            btn.setObjectName("EmojiItemBtn")
            btn.setFixedSize(34, 34)
            btn.setIcon(QIcon(str(path)))
            btn.setIconSize(QSize(24, 24))
            btn.setToolTip(emoji_name)
            btn.clicked.connect(lambda checked=False, n=emoji_name: self._on_pick(n))
            self.recent_layout.addWidget(btn)

        self.recent_layout.addStretch(1)

    def _build_emoji_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.takeAt(i)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not EMOJI_ASSETS_ROOT.exists():
            empty = QLabel("未找到表情资源")
            empty.setStyleSheet("color:#333333;")
            self.grid_layout.addWidget(empty, 0, 0)
            return

        self._emoji_map.clear()
        image_paths: list[Path] = []
        for folder in sorted(EMOJI_ASSETS_ROOT.iterdir()):
            if not folder.is_dir():
                continue
            image_paths.extend(
                sorted(
                    [
                        p
                        for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
                    ]
                )
            )

        for path in image_paths:
            if path.stem not in self._emoji_map:
                self._emoji_map[path.stem] = path

        col_count = 8
        for idx, path in enumerate(image_paths):
            row = idx // col_count
            col = idx % col_count

            btn = QToolButton()
            btn.setObjectName("EmojiItemBtn")
            btn.setFixedSize(38, 38)
            btn.setIcon(QIcon(str(path)))
            btn.setIconSize(QSize(28, 28))
            emoji_name = path.stem
            btn.setToolTip(emoji_name)
            btn.clicked.connect(lambda checked=False, n=emoji_name: self._on_pick(n))

            self.grid_layout.addWidget(btn, row, col)

    def _on_pick(self, emoji_name: str):
        self._push_recent_name(emoji_name)
        emoji_path = self._emoji_map.get(emoji_name)
        if emoji_path is not None:
            self.emoji_selected.emit(str(emoji_path))
        self.hide()


class ChatTopBar(QWidget):
    """微信风格顶部栏。"""

    def __init__(self, parent=None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(2)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)

        self.time_label = QLabel()
        self.time_label.setObjectName("ChatTopTime")
        self.signal_battery_label = QLabel("信号 ▮▮▮▮  电量 100%🔋")
        self.signal_battery_label.setObjectName("ChatTopSignal")

        status_row.addWidget(self.time_label, stretch=0, alignment=Qt.AlignmentFlag.AlignLeft)
        status_row.addStretch(1)
        status_row.addWidget(self.signal_battery_label, stretch=0, alignment=Qt.AlignmentFlag.AlignRight)

        self.name_label = QLabel("爱弥斯")
        self.name_label.setObjectName("ChatTopName")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.state_label = QLabel("状态：电子幽灵")
        self.state_label.setObjectName("ChatTopState")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        root.addLayout(status_row)
        root.addWidget(self.name_label)
        root.addWidget(self.state_label)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._refresh_clock)
        self._clock_timer.start(1000)
        self._refresh_clock()

    def dispose(self):
        self._clock_timer.stop()
        _safe_disconnect(self._clock_timer.timeout, self._refresh_clock)

    def _refresh_clock(self):
        self.time_label.setText(datetime.now().strftime("%H:%M:%S"))


class ChatTimeDivider(QWidget):
    """消息时间分割线。"""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(8)

        left = QFrame()
        left.setFrameShape(QFrame.Shape.HLine)
        left.setObjectName("ChatDividerLine")

        self.label = QLabel(text)
        self.label.setObjectName("ChatDividerText")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        right = QFrame()
        right.setFrameShape(QFrame.Shape.HLine)
        right.setObjectName("ChatDividerLine")

        layout.addWidget(left, stretch=1)
        layout.addWidget(self.label, stretch=0)
        layout.addWidget(right, stretch=1)


class ChatBubble(QWidget):
    """带尖角的圆角气泡。"""

    def __init__(self, content: str, is_player: bool, kind: str = "text", parent=None):
        super().__init__(parent)
        self._is_player = is_player
        self._kind = kind

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        if kind == "image":
            self.content_label = QLabel()
            self.content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = QPixmap(content)
            if not pix.isNull():
                pix = pix.scaled(220, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.content_label.setPixmap(pix)
            else:
                self.content_label.setText("图片加载失败")
        elif kind == "rich":
            self.content_label = QLabel()
            self.content_label.setTextFormat(Qt.TextFormat.RichText)
            self.content_label.setWordWrap(True)
            self.content_label.setOpenExternalLinks(False)
            self.content_label.setText(content)
        else:
            self.content_label = QLabel(content)
            self.content_label.setWordWrap(True)

        self.content_label.setStyleSheet("background: transparent; color: #2f2f2f; font-size: 14px;")
        layout.addWidget(self.content_label)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        bubble_color = QColor("#95ec69") if self._is_player else QColor("#ffffff")
        border_color = QColor("#d9d9d9")

        rect = self.rect().adjusted(0, 0, 0, -2)
        radius = 12
        tail_w = 8
        tail_h = 10

        if self._is_player:
            bubble_rect = QRectF(rect.left(), rect.top(), rect.width() - tail_w, rect.height() - 1)
        else:
            bubble_rect = QRectF(rect.left() + tail_w, rect.top(), rect.width() - tail_w, rect.height() - 1)

        path = QPainterPath()
        path.addRoundedRect(bubble_rect, radius, radius)

        if self._is_player:
            tail = QPainterPath()
            tail.moveTo(bubble_rect.right() - 1, bubble_rect.bottom() - 18)
            tail.lineTo(bubble_rect.right() + tail_w, bubble_rect.bottom() - 12)
            tail.lineTo(bubble_rect.right() - 1, bubble_rect.bottom() - 8)
            tail.closeSubpath()
            path.addPath(tail)
        else:
            tail = QPainterPath()
            tail.moveTo(bubble_rect.left() + 1, bubble_rect.bottom() - 18)
            tail.lineTo(bubble_rect.left() - tail_w, bubble_rect.bottom() - 12)
            tail.lineTo(bubble_rect.left() + 1, bubble_rect.bottom() - 8)
            tail.closeSubpath()
            path.addPath(tail)

        painter.fillPath(path, bubble_color)
        painter.setPen(border_color)
        painter.drawPath(path)


class ChatMessageRow(QWidget):
    """单条消息行。"""

    def __init__(self, message: ChatMessage, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        is_player = message.role == "player"
        self.setMinimumHeight(52)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        avatar = QLabel()
        avatar.setFixedSize(38, 38)
        avatar_pix = QPixmap(str(PLAYER_AVATAR_PATH if is_player else PET_AVATAR_PATH))
        if not avatar_pix.isNull():
            avatar.setPixmap(avatar_pix.scaled(38, 38, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation))

        bubble = ChatBubble(message.content, is_player=is_player, kind=message.kind)

        if is_player:
            layout.addStretch(1)
            layout.addWidget(bubble, stretch=0)
            layout.addWidget(avatar, stretch=0)
        else:
            layout.addWidget(avatar, stretch=0)
            layout.addWidget(bubble, stretch=0)
            layout.addStretch(1)


class ChatPanel(QWidget):
    """聊天面板，可复用于主界面页和独立聊天窗口。"""

    def __init__(self, session: ChatSession, parent=None):
        super().__init__(parent)
        self.session = session
        self._last_divider_key = ""
        self._bg_pixmap = QPixmap(str(CHAT_BACKGROUND_PATH))
        self._animations: list[QParallelAnimationGroup] = []
        self._is_waiting_reply = False
        self._reply_arrived = False
        self._min_wait_elapsed = False
        self._sending_timer = QTimer(self)
        self._sending_timer.setSingleShot(True)
        self._sending_timer.timeout.connect(self._on_sending_min_wait_done)
        self._sending_guard_timer = QTimer(self)
        self._sending_guard_timer.setSingleShot(True)
        self._sending_guard_timer.timeout.connect(self._on_sending_guard_timeout)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.top_bar = ChatTopBar(self)
        root.addWidget(self.top_bar, stretch=0)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("ChatScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.viewport().setAutoFillBackground(False)
        self.scroll.viewport().setStyleSheet("background: transparent;")

        self.list_holder = QWidget()
        self.list_holder.setObjectName("ChatListHolder")
        self.list_layout = QVBoxLayout(self.list_holder)
        self.list_layout.setContentsMargins(8, 8, 8, 8)
        self.list_layout.setSpacing(8)
        self.list_layout.addStretch(1)
        self.scroll.setWidget(self.list_holder)

        root.addWidget(self.scroll, stretch=1)

        input_row = QWidget()
        input_row.setObjectName("ChatInputRow")
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(6)

        self.emoji_btn = QPushButton("☺")
        self.emoji_btn.setObjectName("ChatIconBtn")
        self.emoji_btn.setFixedSize(34, 34)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("输入消息...")
        self.input_edit.setObjectName("ChatInputEdit")
        self.input_edit.setFixedHeight(86)
        self.input_edit.setViewportMargins(0, 0, 0, 0)
        self.input_edit.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.add_btn = QPushButton("+")
        self.add_btn.setObjectName("ChatIconBtn")
        self.add_btn.setFixedSize(34, 34)
        self.add_btn.setToolTip("选择本地图片")
        self.send_btn = QPushButton("发送")
        self.send_btn.setObjectName("ChatSendBtn")
        self.send_btn.setFixedSize(72, 34)

        self.emoji_picker = EmojiPickerPopup(self)
        self.emoji_picker.emoji_selected.connect(self._on_emoji_selected)

        input_layout.addWidget(self.emoji_btn, stretch=0)
        input_layout.addWidget(self.input_edit, stretch=1)
        input_layout.addWidget(self.add_btn, stretch=0)
        input_layout.addWidget(self.send_btn, stretch=0)

        root.addWidget(input_row, stretch=0)

        self.empty_hint_label = QLabel("不能发送空消息")
        self.empty_hint_label.setObjectName("ChatEmptyHint")
        self.empty_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint_label.hide()
        root.addWidget(self.empty_hint_label, stretch=0)

        self.setStyleSheet(
            """
            ChatPanel {
                background: transparent;
            }
            ChatTopBar {
                background: rgba(20, 20, 20, 0.36);
                border-bottom: 1px solid rgba(255, 255, 255, 0.28);
            }
            QLabel#ChatTopTime, QLabel#ChatTopSignal {
                color: #ffffff;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#ChatTopName {
                color: #ffffff;
                font-size: 17px;
                font-weight: 700;
            }
            QLabel#ChatTopState {
                color: #ffffff;
                font-size: 12px;
            }
            QScrollArea#ChatScroll, QWidget#ChatListHolder {
                background: transparent;
                border: none;
            }
            QLabel#ChatDividerText {
                color: #ffffff;
                font-size: 11px;
                padding: 2px 6px;
                background: rgba(255, 232, 241, 0.88);
                border-radius: 8px;
            }
            QFrame#ChatDividerLine {
                color: rgba(255, 255, 255, 0.65);
                background: rgba(255, 255, 255, 0.65);
                min-height: 1px;
                max-height: 1px;
                border: none;
            }
            QWidget#ChatInputRow {
                background: rgba(255, 255, 255, 0.22);
                border-top: 1px solid rgba(255, 255, 255, 0.38);
            }
            QPushButton#ChatIconBtn {
                background: rgba(255, 230, 241, 0.88);
                border: 1px solid #f2bfd7;
                border-radius: 17px;
                color: #1f1f1f;
                font-size: 16px;
            }
            QPushButton#ChatIconBtn:hover {
                background: #ffddec;
            }
            QTextEdit#ChatInputEdit {
                background: rgba(255, 230, 241, 0.9);
                color: #111111;
                border: 1px solid #f2bfd7;
                border-radius: 8px;
                padding: 6px 8px;
                font-size: 14px;
            }
            QTextEdit#ChatInputEdit::placeholder {
                color: #4a4a4a;
            }
            QPushButton#ChatSendBtn {
                background: #07c160;
                border: none;
                border-radius: 6px;
                color: white;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton#ChatSendBtn:hover {
                background: #06ad56;
            }
            QPushButton#ChatSendBtn:disabled {
                background: #8fdcb2;
                color: #eaf8f0;
            }
            QLabel#ChatEmptyHint {
                color: #1b1b1b;
                font-size: 12px;
                background: rgba(255, 230, 241, 0.9);
                border-top: 1px solid #f0bfd5;
                padding: 4px 0;
            }
            QToolTip {
                background: #ffe6f1;
                color: #222222;
                border: 1px solid #f3bfd8;
                padding: 3px 6px;
            }
            """
        )

        self.send_btn.clicked.connect(self._on_send_clicked)
        self.add_btn.clicked.connect(self._on_add_image_clicked)
        self.emoji_btn.clicked.connect(self._on_open_emoji_picker)
        self.input_edit.textChanged.connect(self._update_send_btn_state)
        self.input_edit.installEventFilter(self)
        self.session.message_added.connect(self._on_message_added)
        self.session.session_cleared.connect(self._on_session_cleared)
        self.session.active_conversation_changed.connect(self._on_active_conversation_changed)

        self._update_send_btn_state()

        self._reload_current_conversation_messages()
        QTimer.singleShot(0, self._scroll_to_latest_message)

    def dispose(self):
        self.top_bar.dispose()
        _safe_disconnect(self.send_btn.clicked, self._on_send_clicked)
        _safe_disconnect(self.add_btn.clicked, self._on_add_image_clicked)
        _safe_disconnect(self.session.message_added, self._on_message_added)
        _safe_disconnect(self.session.session_cleared, self._on_session_cleared)
        _safe_disconnect(self.session.active_conversation_changed, self._on_active_conversation_changed)
        _safe_disconnect(self.input_edit.textChanged, self._update_send_btn_state)
        try:
            self.input_edit.removeEventFilter(self)
        except Exception:
            pass
        try:
            self._sending_timer.stop()
        except Exception:
            pass
        try:
            self._sending_guard_timer.stop()
        except Exception:
            pass
        _safe_disconnect(self._sending_timer.timeout, self._on_sending_min_wait_done)
        _safe_disconnect(self._sending_guard_timer.timeout, self._on_sending_guard_timeout)
        _safe_disconnect(self.emoji_picker.emoji_selected, self._on_emoji_selected)
        for animation in self._animations:
            try:
                animation.stop()
            except Exception:
                pass
        self._animations.clear()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._bg_pixmap.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setOpacity(0.35)

        target = self.rect()
        scaled = self._bg_pixmap.scaled(
            target.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (target.width() - scaled.width()) // 2
        y = (target.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._scroll_to_latest_message)

    def eventFilter(self, watched, event):
        if watched is self.input_edit and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                    self._on_send_clicked()
                    return True
        return super().eventFilter(watched, event)

    def _append_message_widget(self, message: ChatMessage):
        divider_key = message.timestamp.strftime("%Y-%m-%d %H:%M")
        if divider_key != self._last_divider_key:
            divider_text = message.timestamp.strftime("%m-%d %H:%M")
            divider = ChatTimeDivider(divider_text)
            self.list_layout.insertWidget(self.list_layout.count() - 1, divider)
            self._last_divider_key = divider_key

        row = ChatMessageRow(message)
        self.list_layout.insertWidget(self.list_layout.count() - 1, row)
        self._play_message_animation(row)
        QTimer.singleShot(0, self._scroll_to_latest_message)

    def _clear_message_widgets(self):
        self._last_divider_key = ""
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _reload_current_conversation_messages(self):
        self._clear_message_widgets()
        for message in self.session.messages:
            self._append_message_widget(message)
        QTimer.singleShot(0, self._scroll_to_latest_message)

    def _on_open_emoji_picker(self):
        popup_x = 0
        popup_y = -self.emoji_picker.height() - 8
        global_pos = self.emoji_btn.mapToGlobal(QPoint(popup_x, popup_y))
        self.emoji_picker.move(global_pos)
        self.emoji_picker.show()

    def _on_emoji_selected(self, emoji_path: str):
        if not emoji_path:
            return
        uri = Path(emoji_path).resolve().as_uri()
        cursor = self.input_edit.textCursor()
        cursor.insertHtml(f'<img src="{uri}" width="20" height="20" />')
        self.input_edit.setTextCursor(cursor)
        self._update_send_btn_state()

    def _scroll_to_bottom(self):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _scroll_to_latest_message(self):
        self._scroll_to_bottom()

        last_row = None
        for idx in range(self.list_layout.count() - 1, -1, -1):
            item = self.list_layout.itemAt(idx)
            if item is None:
                continue
            widget = item.widget()
            if isinstance(widget, ChatMessageRow):
                last_row = widget
                break

        if last_row is not None:
            self.scroll.ensureWidgetVisible(last_row, 0, 18)

    @staticmethod
    def _extract_inline_image_paths(html: str) -> list[str]:
        paths: list[str] = []
        if not html:
            return paths

        for src in re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE):
            parsed = urlparse(src)
            if parsed.scheme == "file":
                local_path = unquote(parsed.path)
                if os.name == "nt" and local_path.startswith("/") and len(local_path) > 2 and local_path[2] == ":":
                    local_path = local_path[1:]
                if local_path:
                    paths.append(local_path)
            elif src:
                paths.append(src)
        return paths

    @staticmethod
    def _normalize_local_path(src: str) -> str:
        source = str(src).strip()
        if not source:
            return ""
        parsed = urlparse(source)
        if parsed.scheme == "file":
            local_path = unquote(parsed.path)
            if os.name == "nt" and local_path.startswith("/") and len(local_path) > 2 and local_path[2] == ":":
                local_path = local_path[1:]
            return local_path
        return source

    @staticmethod
    def _is_emoji_asset(path: str) -> bool:
        if not path:
            return False
        try:
            normalized = Path(path).resolve()
            root = EMOJI_ASSETS_ROOT.resolve()
            return normalized == root or root in normalized.parents
        except Exception:
            return False

    def _build_compose_payload(self) -> tuple[str, str]:
        plain_text = self.input_edit.toPlainText()
        html = self.input_edit.toHtml()
        image_sources = self._extract_inline_image_paths(html)

        display_parts: list[str] = []
        record_parts: list[str] = []
        image_index = 0

        for char in plain_text:
            if char == "\uFFFC":
                source = image_sources[image_index] if image_index < len(image_sources) else ""
                image_index += 1
                local_path = self._normalize_local_path(source)
                if not local_path:
                    continue

                resolved_path = Path(local_path)
                uri = resolved_path.resolve().as_uri()
                escaped_uri = html_lib.escape(uri, quote=True)

                if self._is_emoji_asset(local_path):
                    display_parts.append(f'<img src="{escaped_uri}" width="20" height="20" />')
                    record_parts.append(f"[{resolved_path.stem}]")
                else:
                    display_parts.append(f'<img src="{escaped_uri}" width="120" height="120" />')
                    record_parts.append(f"[图片:{resolved_path.name}]")
                continue

            if char == "\n":
                display_parts.append("<br/>")
                record_parts.append("\n")
            else:
                escaped_char = html_lib.escape(char)
                display_parts.append(escaped_char)
                record_parts.append(char)

        while image_index < len(image_sources):
            local_path = self._normalize_local_path(image_sources[image_index])
            image_index += 1
            if not local_path:
                continue

            resolved_path = Path(local_path)
            uri = resolved_path.resolve().as_uri()
            escaped_uri = html_lib.escape(uri, quote=True)

            if self._is_emoji_asset(local_path):
                display_parts.append(f'<img src="{escaped_uri}" width="20" height="20" />')
                record_parts.append(f"[{resolved_path.stem}]")
            else:
                display_parts.append(f'<img src="{escaped_uri}" width="120" height="120" />')
                record_parts.append(f"[图片:{resolved_path.name}]")

        display_html = "".join(display_parts).strip()
        record_text = "".join(record_parts).strip()
        return display_html, record_text

    def _on_send_clicked(self):
        display_html, record_text = self._build_compose_payload()
        if not display_html and not record_text:
            self._show_empty_hint()
            return

        self._start_sending_state()
        self.session.send_composed(display_html, record_text)
        self.input_edit.clear()
        self._update_send_btn_state()

    def _on_add_image_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            uri = Path(path).resolve().as_uri()
            cursor = self.input_edit.textCursor()
            cursor.insertHtml(f'<img src="{uri}" width="120" height="120" />')
            self.input_edit.setTextCursor(cursor)
            self._update_send_btn_state()

    def _on_message_added(self, conversation_id: str, message: ChatMessage):
        if str(conversation_id) != str(self.session.current_conversation_id):
            return
        self._append_message_widget(message)
        if self._is_waiting_reply and message.role == "pet":
            self._reply_arrived = True
            self._try_finish_sending_state()

    def _on_session_cleared(self):
        self._clear_message_widgets()

        self._is_waiting_reply = False
        self._reply_arrived = False
        self._min_wait_elapsed = False
        self._sending_guard_timer.stop()
        self.send_btn.setText("发送")
        self._update_send_btn_state()

    def _on_active_conversation_changed(self, conversation_id: str):
        _ = conversation_id
        self._is_waiting_reply = False
        self._reply_arrived = False
        self._min_wait_elapsed = False
        self._sending_timer.stop()
        self._sending_guard_timer.stop()
        self.send_btn.setText("发送")
        self._update_send_btn_state()
        self._reload_current_conversation_messages()

    def _show_empty_hint(self):
        self.empty_hint_label.show()
        QTimer.singleShot(1400, self.empty_hint_label.hide)

    def _start_sending_state(self):
        self._is_waiting_reply = True
        self._reply_arrived = False
        self._min_wait_elapsed = False
        self.empty_hint_label.hide()
        self.send_btn.setText("发送中…")
        self.send_btn.setEnabled(False)
        self._sending_timer.stop()
        self._sending_timer.start(1000)
        self._sending_guard_timer.stop()
        self._sending_guard_timer.start(45000)

    def _on_sending_min_wait_done(self):
        self._min_wait_elapsed = True
        self._try_finish_sending_state()

    def _try_finish_sending_state(self):
        if not self._is_waiting_reply:
            return
        if not (self._reply_arrived and self._min_wait_elapsed):
            return

        self._is_waiting_reply = False
        self._sending_guard_timer.stop()
        self.send_btn.setText("发送")
        self._update_send_btn_state()

    def _on_sending_guard_timeout(self):
        if not self._is_waiting_reply:
            return
        self._is_waiting_reply = False
        self._reply_arrived = False
        self._min_wait_elapsed = False
        self.send_btn.setText("发送")
        self._update_send_btn_state()
        self.empty_hint_label.setText("请求超时，已恢复发送")
        self.empty_hint_label.show()
        QTimer.singleShot(1400, self.empty_hint_label.hide)
        QTimer.singleShot(1600, lambda: self.empty_hint_label.setText("不能发送空消息"))

    def _play_message_animation(self, widget: QWidget):
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.2)
        widget.setGraphicsEffect(effect)

        opacity_animation = QPropertyAnimation(effect, b"opacity", widget)
        opacity_animation.setDuration(500)
        opacity_animation.setStartValue(0.2)
        opacity_animation.setEndValue(1.0)
        opacity_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        group = QParallelAnimationGroup(widget)
        group.addAnimation(opacity_animation)

        def _cleanup_group(finished_group=group, target_widget=widget):
            try:
                target_widget.setGraphicsEffect(None)
            except Exception:
                pass
            self._scroll_to_latest_message()
            if finished_group in self._animations:
                self._animations.remove(finished_group)

        group.finished.connect(_cleanup_group)
        self._animations.append(group)
        group.start()

    def _update_send_btn_state(self):
        if self._is_waiting_reply:
            self.send_btn.setEnabled(False)
            return

        display_html, record_text = self._build_compose_payload()
        self.send_btn.setEnabled(bool(display_html or record_text))
