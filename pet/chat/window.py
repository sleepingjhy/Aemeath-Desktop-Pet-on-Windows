"""独立聊天窗口。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .session import ChatSession
from .widgets import ChatPanel


class ChatWindow(QMainWindow):
    """桌宠聊天独立窗口。"""

    def __init__(self, session: ChatSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        root_widget = QWidget(self)
        root_layout = QHBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        left_panel = QWidget(root_widget)
        left_panel.setObjectName("ChatConversationPanel")
        left_panel.setFixedWidth(210)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        self.conversation_title = QLabel("会话列表", left_panel)
        self.conversation_title.setObjectName("ChatConversationTitle")

        self.new_conversation_btn = QPushButton("新对话", left_panel)
        self.new_conversation_btn.setObjectName("ChatConversationNewBtn")
        self.delete_conversation_btn = QPushButton("删除对话", left_panel)
        self.delete_conversation_btn.setObjectName("ChatConversationDeleteBtn")

        self.conversation_list = QListWidget(left_panel)
        self.conversation_list.setObjectName("ChatConversationList")
        self.conversation_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        left_layout.addWidget(self.conversation_title)
        left_layout.addWidget(self.new_conversation_btn)
        left_layout.addWidget(self.delete_conversation_btn)
        left_layout.addWidget(self.conversation_list, stretch=1)

        self.chat_panel = ChatPanel(session=self.session, parent=root_widget)

        root_layout.addWidget(left_panel, stretch=0)
        root_layout.addWidget(self.chat_panel, stretch=1)
        self.setCentralWidget(root_widget)

        self.setWindowTitle("Ameath Chat")
        self.setMinimumSize(760, 700)

        self.setStyleSheet(
            self.styleSheet()
            + """
            QWidget#ChatConversationPanel {
                background: rgba(255, 245, 250, 0.92);
                border-right: 1px solid rgba(220, 190, 208, 0.85);
            }
            QLabel#ChatConversationTitle {
                color: #333333;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton#ChatConversationNewBtn {
                background: #07c160;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                min-height: 30px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton#ChatConversationNewBtn:hover {
                background: #06ad56;
            }
            QPushButton#ChatConversationDeleteBtn {
                background: #f04f59;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                min-height: 30px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton#ChatConversationDeleteBtn:hover {
                background: #d6444e;
            }
            QListWidget#ChatConversationList {
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid #efc5da;
                border-radius: 8px;
                padding: 4px;
            }
            QListWidget#ChatConversationList::item {
                padding: 6px 8px;
                margin: 2px 0;
                border-radius: 6px;
                color: #333333;
            }
            QListWidget#ChatConversationList::item:selected {
                background: #ffdcec;
                color: #1f1f1f;
            }
            """
        )

        self.new_conversation_btn.clicked.connect(self._on_new_conversation_clicked)
        self.delete_conversation_btn.clicked.connect(self._on_delete_conversation_clicked)
        self.conversation_list.itemClicked.connect(self._on_conversation_item_clicked)
        self.conversation_list.customContextMenuRequested.connect(self._on_conversation_context_menu)
        self.session.conversation_list_changed.connect(self._refresh_conversation_list)
        self.session.active_conversation_changed.connect(self._on_active_conversation_changed)

        self._refresh_conversation_list()
        self._on_active_conversation_changed(self.session.current_conversation_id)

    def _refresh_conversation_list(self):
        active_id = self.session.current_conversation_id
        self.conversation_list.blockSignals(True)
        self.conversation_list.clear()

        for conversation in self.session.list_conversations():
            conversation_id = str(conversation.get("id", ""))
            title = str(conversation.get("title", "")) or "未命名会话"
            updated_at = conversation.get("updated_at")
            if hasattr(updated_at, "strftime"):
                time_text = updated_at.strftime("%m-%d %H:%M")
            else:
                time_text = "--"
            item = QListWidgetItem(f"{title}\n{time_text}")
            item.setData(Qt.ItemDataRole.UserRole, conversation_id)
            self.conversation_list.addItem(item)
            if conversation_id == active_id:
                self.conversation_list.setCurrentItem(item)

        self.conversation_list.blockSignals(False)

    def _on_new_conversation_clicked(self):
        conversation_id = self.session.create_conversation()
        self.session.switch_conversation(conversation_id)

    def _on_delete_conversation_clicked(self):
        item = self.conversation_list.currentItem()
        self._delete_conversation_item(item)

    def _on_conversation_context_menu(self, pos):
        item = self.conversation_list.itemAt(pos)
        if item is None:
            return

        self.conversation_list.setCurrentItem(item)
        menu = QMenu(self)
        delete_action = menu.addAction("删除对话")
        selected_action = menu.exec(self.conversation_list.mapToGlobal(pos))
        menu.deleteLater()
        if selected_action == delete_action:
            self._delete_conversation_item(item)

    def _delete_conversation_item(self, item: QListWidgetItem | None):
        if item is None:
            return

        conversation_id = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not conversation_id:
            return

        title_text = str(item.text() or "").split("\n", 1)[0].strip() or "当前会话"
        result = QMessageBox.question(
            self,
            "删除对话",
            f"确定删除“{title_text}”吗？\n此操作会同步清除该对话的内存缓存。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return

        self.session.delete_conversation(conversation_id)

    def _on_conversation_item_clicked(self, item: QListWidgetItem):
        if item is None:
            return
        conversation_id = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if conversation_id:
            self.session.switch_conversation(conversation_id)

    def _on_active_conversation_changed(self, conversation_id: str):
        target_id = str(conversation_id).strip()
        if not target_id:
            return
        self.conversation_list.blockSignals(True)
        for index in range(self.conversation_list.count()):
            item = self.conversation_list.item(index)
            if item is None:
                continue
            item_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if item_id == target_id:
                self.conversation_list.setCurrentItem(item)
                break
        self.conversation_list.blockSignals(False)

    def show_window(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def prepare_for_exit(self):
        if hasattr(self, "chat_panel") and self.chat_panel is not None:
            self.chat_panel.dispose()
        try:
            self.new_conversation_btn.clicked.disconnect(self._on_new_conversation_clicked)
        except Exception:
            pass
        try:
            self.delete_conversation_btn.clicked.disconnect(self._on_delete_conversation_clicked)
        except Exception:
            pass
        try:
            self.conversation_list.itemClicked.disconnect(self._on_conversation_item_clicked)
        except Exception:
            pass
        try:
            self.conversation_list.customContextMenuRequested.disconnect(self._on_conversation_context_menu)
        except Exception:
            pass
        try:
            self.session.conversation_list_changed.disconnect(self._refresh_conversation_list)
        except Exception:
            pass
        try:
            self.session.active_conversation_changed.disconnect(self._on_active_conversation_changed)
        except Exception:
            pass

    def closeEvent(self, event: QCloseEvent):
        self.prepare_for_exit()
        event.accept()
