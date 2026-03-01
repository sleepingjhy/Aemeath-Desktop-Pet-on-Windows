"""该模块是前端主界面。包含设置页和关于页。"""
# EN: This module defines the main frontend window, including the Settings and About pages.

from __future__ import annotations

import os

from PySide6.QtCore import QEvent, Qt, QSize
from PySide6.QtGui import QCloseEvent, QIcon, QMovie, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyledItemDelegate,
    QSizePolicy,
    QStackedWidget,
    QLineEdit,
    QToolButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import (
    ABOUT_GIF_PATH,
    APP_ICON_PATH,
    DISPLAY_MODE_ALWAYS_ON_TOP,
    DISPLAY_MODE_DESKTOP_ONLY,
    DISPLAY_MODE_FULLSCREEN_HIDE,
    INSTANCE_COUNT_MAX,
    INSTANCE_COUNT_MIN,
    OPACITY_DEFAULT_PERCENT,
    OPACITY_PERCENT_MAX,
    OPACITY_PERCENT_MIN,
    SCALE_MAX,
    SCALE_MIN,
)
from .i18n import get_language_items, normalize_language, tr


class MusicDeleteCheckDelegate(QStyledItemDelegate):
    """删除模式下在左侧自绘蓝框与白色勾。"""
    """EN: Draw left-side blue checkbox and white tick in delete mode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._delete_mode = False

    def set_delete_mode(self, enabled: bool):
        self._delete_mode = bool(enabled)

    def paint(self, painter: QPainter, option, index):
        super().paint(painter, option, index)

        if not self._delete_mode:
            return

        size = 16
        left = option.rect.left() + 8
        top = option.rect.top() + (option.rect.height() - size) // 2

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        state_data = index.data(Qt.ItemDataRole.CheckStateRole)
        checked = False
        if state_data is not None:
            try:
                checked = int(state_data) == int(Qt.CheckState.Checked)
            except Exception:
                checked = state_data == Qt.CheckState.Checked
        border_color = QColor("#2f75ff") if checked else QColor("#6aa5ff")
        fill_color = QColor("#3b82f6") if checked else QColor("#d7e8ff")

        painter.setPen(QPen(border_color, 1))
        painter.setBrush(fill_color)
        painter.drawRoundedRect(left, top, size, size, 3, 3)

        if checked:
            tick_pen = QPen(QColor("#ffffff"), 2)
            tick_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            tick_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(tick_pen)
            painter.drawLine(left + 4, top + 8, left + 7, top + 11)
            painter.drawLine(left + 7, top + 11, left + 12, top + 5)

        painter.restore()


class AppWindow(QMainWindow):
    """应用主界面窗口。负责设置与关于页面展示。"""
    """EN: The main interface window of the app. Responsible for setting up and presenting about the page."""

    PAGE_SETTINGS = 0
    PAGE_CHAT = 1
    PAGE_MUSIC = 2
    PAGE_ABOUT = 3

    def __init__(
        self,
        pet,
        settings_store,
        close_policy,
        request_quit,
        tray_controller=None,
        music_player=None,
        chat_session=None,
        on_open_chat_window=None,
    ):
        """初始化界面、样式和交互绑定。"""
        """EN: Initialize the interface, styles, and interactive bindings."""
        super().__init__()
        self.pet = pet
        self.settings_store = settings_store
        self.close_policy = close_policy
        self.request_quit = request_quit
        self.tray_controller = tray_controller
        self.music_player = music_player
        self.chat_session = chat_session
        self.on_open_chat_window = on_open_chat_window
        self._last_non_chat_page_index = self.PAGE_SETTINGS
        self._is_exiting = False
        self.language = normalize_language(self.settings_store.get_language())

        self.setWindowTitle("Ameath Desktop Pet")
        self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.setMinimumSize(980, 720)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self.about_text_content_map = {
            "zh-CN": (
                "爱弥斯——变身~\n"
                "群星，皎映明日！\n"
                "此夜，星海澈明！\n"
                "救世之刻，已至！\n"
                "但愿我会让你感到骄傲\n"
                "但愿我不会让你失望😭\n\n"
                "❤️❤️❤️ 爱来自 jhy ❤️❤️❤️"
            ),
            "en": (
                "Ameath — transform~\n"
                "Stars shine toward tomorrow!\n"
                "Tonight, the sea of stars is crystal clear!\n"
                "The moment of salvation has arrived!\n"
                "I hope I make you proud\n"
                "I hope I won't let you down 😭\n\n"
                "❤️❤️❤️ Love from jhy ❤️❤️❤️"
            ),
            "ja": (
                "アミース——変身！\n"
                "星々は明日を照らす！\n"
                "今宵、星の海は澄みわたる！\n"
                "救済の刻は来た！\n"
                "あなたの誇りになれますように\n"
                "あなたを失望させませんように😭\n\n"
                "❤️❤️❤️ jhy からの愛 ❤️❤️❤️"
            ),
            "ko": (
                "아메스——변신~\n"
                "별들이 내일을 비춘다!\n"
                "오늘 밤, 별바다는 맑고 투명해!\n"
                "구원의 순간이 왔어!\n"
                "네가 자랑스러워할 수 있기를\n"
                "실망시키지 않기를 바래😭\n\n"
                "❤️❤️❤️ jhy의 사랑 ❤️❤️❤️"
            ),
            "fr": (
                "Ameath — transformation~\n"
                "Les étoiles éclairent demain !\n"
                "Ce soir, la mer d'étoiles est limpide !\n"
                "L'heure du salut est arrivée !\n"
                "J'espère te rendre fier\n"
                "J'espère ne pas te décevoir 😭\n\n"
                "❤️❤️❤️ Avec amour, jhy ❤️❤️❤️"
            ),
        }

        self._build_ui()
        self._bind_pet_state_sync()
        self._apply_theme()

    def _build_ui(self):
        """构建主界面布局和两个页面。"""
        """EN: Build the main interface layout and two pages."""
        root = QWidget(self)
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        nav_card = QFrame()
        nav_card.setObjectName("NavCard")
        nav_card.setFixedWidth(180)
        nav_layout = QVBoxLayout(nav_card)
        nav_layout.setContentsMargins(12, 12, 12, 12)
        nav_layout.setSpacing(8)
        nav_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        icon_label = QLabel()
        icon_label.setPixmap(QIcon(str(APP_ICON_PATH)).pixmap(44, 44))
        nav_layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.settings_btn = QPushButton(tr(self.language, "app.settings"))
        self.chat_btn = QPushButton(tr(self.language, "app.chat"))
        self.music_btn = QPushButton(tr(self.language, "app.music"))
        self.about_btn = QPushButton(tr(self.language, "app.about"))
        self.settings_btn.setObjectName("NavButton")
        self.chat_btn.setObjectName("NavButton")
        self.music_btn.setObjectName("NavButton")
        self.about_btn.setObjectName("NavButton")
        self.settings_btn.setFixedWidth(120)
        self.chat_btn.setFixedWidth(120)
        self.music_btn.setFixedWidth(120)
        self.about_btn.setFixedWidth(120)

        self.settings_btn.clicked.connect(lambda: self._open_page(self.PAGE_SETTINGS))
        self.chat_btn.clicked.connect(self._on_chat_nav_clicked)
        self.music_btn.clicked.connect(lambda: self._open_page(self.PAGE_MUSIC))
        self.about_btn.clicked.connect(lambda: self._open_page(self.PAGE_ABOUT))

        nav_layout.addWidget(self.settings_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addWidget(self.chat_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addWidget(self.music_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addWidget(self.about_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addStretch(1)

        self.pages = QStackedWidget()
        self.pages.currentChanged.connect(self._on_page_changed)
        self.pages.addWidget(self._build_settings_page())   # index 0
        self.pages.addWidget(self._build_chat_page())       # index 1
        self.pages.addWidget(self._build_music_page())      # index 2
        self.pages.addWidget(self._build_about_page())      # index 3

        main_layout.addWidget(nav_card, stretch=0)
        main_layout.addWidget(self.pages, stretch=1)

    def _bind_pet_state_sync(self):
        """绑定桌宠状态信号，实现设置页控件实时同步。"""
        """EN: Bind the table pet status signal to achieve real-time synchronization of the settings page control."""
        if hasattr(self.pet, "follow_changed"):
            self.pet.follow_changed.connect(self._on_pet_follow_changed)
        if hasattr(self.pet, "move_enabled_changed"):
            self.pet.move_enabled_changed.connect(self._on_pet_move_enabled_changed)
        if hasattr(self.pet, "scale_changed"):
            self.pet.scale_changed.connect(self._on_pet_scale_changed)
        if hasattr(self.pet, "autostart_changed"):
            self.pet.autostart_changed.connect(self._on_pet_autostart_changed)
        if hasattr(self.pet, "display_mode_changed"):
            self.pet.display_mode_changed.connect(self._on_pet_display_mode_changed)
        if hasattr(self.pet, "instance_count_changed"):
            self.pet.instance_count_changed.connect(self._on_pet_instance_count_changed)
        if hasattr(self.pet, "opacity_changed"):
            self.pet.opacity_changed.connect(self._on_pet_opacity_changed)
        if hasattr(self.pet, "language_changed"):
            self.pet.language_changed.connect(self._on_pet_language_changed)

        # 绑定音乐播放器信号
        # EN: Bind Music Player Signal
        if self.music_player is not None:
            self.music_player.track_changed.connect(self._on_music_track_changed)
            self.music_player.state_changed.connect(self._on_music_state_changed)
            self.music_player.playlist_reordered.connect(self._on_music_playlist_reordered)
            self.music_player.volume_changed.connect(self._on_music_volume_changed)
            self.music_player.mode_changed.connect(self._on_music_mode_changed)
            self.music_player.duration_changed.connect(self._on_music_duration_changed)
            self.music_player.position_changed.connect(self._on_music_position_changed)

        self._sync_controls_from_pet()

    def _sync_controls_from_pet(self):
        """将设置页控件与当前桌宠状态对齐。"""
        """EN: Align the settings page control with the current table darling state."""
        self._on_pet_follow_changed(self.pet.state.follow_mouse)
        self._on_pet_move_enabled_changed(self._resolve_pet_move_enabled())
        self._on_pet_scale_changed(self.pet.scale_factor)
        self._on_pet_autostart_changed(self.pet.get_autostart_enabled())
        self._on_pet_display_mode_changed(self._resolve_pet_display_mode())
        self._on_pet_instance_count_changed(self._resolve_pet_instance_count())
        self._on_pet_opacity_changed(self._resolve_pet_opacity_percent())

    def _resolve_pet_display_mode(self) -> str:
        """读取当前显示模式。优先调用标准 getter。"""
        """EN: Reads the current display mode. Call the standard getter first."""
        if hasattr(self.pet, "get_display_mode") and callable(self.pet.get_display_mode):
            mode = self.pet.get_display_mode()
            if isinstance(mode, str):
                return mode

        mode = getattr(self.pet, "display_mode", DISPLAY_MODE_ALWAYS_ON_TOP)
        if isinstance(mode, str):
            return mode
        return DISPLAY_MODE_ALWAYS_ON_TOP

    def _resolve_pet_instance_count(self) -> int:
        """读取当前实例数量。优先调用标准 getter。"""
        """EN: Reads the current number of instances. Call the standard getter first."""
        if hasattr(self.pet, "get_instance_count") and callable(self.pet.get_instance_count):
            value = self.pet.get_instance_count()
        else:
            value = getattr(self.pet, "target_count", INSTANCE_COUNT_MIN)

        try:
            count = int(value)
        except (TypeError, ValueError):
            count = INSTANCE_COUNT_MIN
        return max(INSTANCE_COUNT_MIN, min(INSTANCE_COUNT_MAX, count))

    def _resolve_pet_opacity_percent(self) -> int:
        """读取当前透明度百分比。优先调用标准 getter。"""
        """EN: Reads the current transparency percentage. Call the standard getter first."""
        if hasattr(self.pet, "get_opacity_percent") and callable(self.pet.get_opacity_percent):
            value = self.pet.get_opacity_percent()
        else:
            value = getattr(self.pet, "opacity_percent", OPACITY_DEFAULT_PERCENT)

        try:
            opacity = int(value)
        except (TypeError, ValueError):
            opacity = OPACITY_DEFAULT_PERCENT
        return max(OPACITY_PERCENT_MIN, min(OPACITY_PERCENT_MAX, opacity))

    def _resolve_pet_move_enabled(self) -> bool:
        """读取当前移动开关。优先调用标准 getter。"""
        """EN: Reads the current movement switch. Call the standard getter first."""
        if hasattr(self.pet, "get_move_enabled") and callable(self.pet.get_move_enabled):
            return bool(self.pet.get_move_enabled())
        return bool(getattr(self.pet.state, "move_enabled", True))

    def _on_pet_move_enabled_changed(self, enabled: bool):
        """接收移动开关变化并更新设置页按钮文案。"""
        """EN: Receive mobile switch changes and update settings page button copy."""
        if not hasattr(self, "stop_move_btn"):
            return
        self.stop_move_btn.setText(tr(self.language, "menu.move.resume") if not bool(enabled) else tr(self.language, "menu.move.stop"))

    def _on_pet_language_changed(self, language: str):
        """接收语言变化并重建界面文案。"""
        """EN: Receive language changes and rebuild the interface copy."""
        normalized = normalize_language(language)
        if normalized == self.language:
            return

        self.language = normalized
        self.settings_store.set_language(normalized)
        self._rebuild_ui_for_language()

    def _rebuild_ui_for_language(self):
        """按当前语言重建主界面。"""
        """EN: Rebuild the main screen in the current language."""
        current_index = self.pages.currentIndex() if hasattr(self, "pages") else 0
        if hasattr(self, "about_movie") and self.about_movie is not None:
            self.about_movie.stop()
            self.about_movie.deleteLater()
            self.about_movie = None
        old_widget = self.centralWidget()
        self._build_ui()
        if old_widget is not None:
            old_widget.deleteLater()
        self.pages.setCurrentIndex(current_index)
        self._sync_controls_from_pet()
        if self.music_player is not None:
            self._sync_music_ui_from_player()
        self._update_about_gif_size()

    def _on_pet_follow_changed(self, follow_enabled: bool):
        """接收桌宠跟随状态变化并更新设置页控件。"""
        """EN: Receive table pets to follow status changes and update settings page controls."""
        self.follow_checkbox.blockSignals(True)
        self.follow_checkbox.setChecked(bool(follow_enabled))
        self.follow_checkbox.blockSignals(False)

    def _on_pet_scale_changed(self, scale_value: float):
        """接收桌宠缩放变化并更新设置页控件。"""
        """EN: Receive table pet zoom changes and update settings page controls."""
        min_slider = int(round(SCALE_MIN * 10))
        max_slider = int(round(SCALE_MAX * 10))
        slider_value = int(round(float(scale_value) * 10))
        slider_value = max(min_slider, min(max_slider, slider_value))

        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(slider_value)
        self.scale_slider.blockSignals(False)
        self.scale_value_label.setText(f"{slider_value / 10:.1f}x")

    def _on_pet_autostart_changed(self, enabled: bool):
        """接收桌宠开机自启变化并更新设置页控件。"""
        """EN: Receive table darling startup changes and update the settings page controls."""
        self.autostart_checkbox.blockSignals(True)
        self.autostart_checkbox.setChecked(bool(enabled))
        self.autostart_checkbox.blockSignals(False)

    def _on_pet_display_mode_changed(self, mode: str):
        """接收显示模式变化并更新设置页下拉。"""
        """EN: Receive display mode changes and update settings page dropdown."""
        index = self.display_mode_combo.findData(mode)
        if index < 0:
            index = self.display_mode_combo.findData(DISPLAY_MODE_ALWAYS_ON_TOP)
            if index < 0:
                index = 0

        self.display_mode_combo.blockSignals(True)
        self.display_mode_combo.setCurrentIndex(index)
        self.display_mode_combo.blockSignals(False)

    def _on_pet_instance_count_changed(self, count: int):
        """接收实例数量变化并更新设置页数值控件。"""
        """EN: Receives changes in the number of instances and updates the settings page numeric control."""
        try:
            value = int(count)
        except (TypeError, ValueError):
            value = INSTANCE_COUNT_MIN
        value = max(INSTANCE_COUNT_MIN, min(INSTANCE_COUNT_MAX, value))

        self.instance_count_spin.blockSignals(True)
        self.instance_count_spin.setValue(value)
        self.instance_count_spin.blockSignals(False)

    def _on_pet_opacity_changed(self, opacity: int):
        """接收透明度变化并更新设置页滑块。"""
        """EN: Receive transparency changes and update the settings page slider."""
        try:
            value = int(opacity)
        except (TypeError, ValueError):
            value = OPACITY_DEFAULT_PERCENT
        value = max(OPACITY_PERCENT_MIN, min(OPACITY_PERCENT_MAX, value))

        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(value)
        self.opacity_slider.blockSignals(False)
        self.opacity_value_label.setText(f"{value}%")

    def _build_settings_page(self) -> QWidget:
        """构建设置页。包含所有右键菜单可配置项。"""
        """EN: Build the setup page. Contains all the right-click menu configurable items."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(16, 16, 16, 16)
        page_layout.setSpacing(0)

        self.settings_scroll_area = QScrollArea()
        self.settings_scroll_area.setObjectName("SettingsScrollArea")
        self.settings_scroll_area.setWidgetResizable(True)
        self.settings_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.settings_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        settings_content = QWidget()
        settings_content.setObjectName("SettingsPageContent")
        layout = QVBoxLayout(settings_content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel(tr(self.language, "app.settings"))
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        form_card = QFrame()
        form_card.setObjectName("Card")
        form_layout = QFormLayout(form_card)
        form_layout.setContentsMargins(16, 16, 16, 16)
        form_layout.setVerticalSpacing(12)
        form_layout.setHorizontalSpacing(16)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.stop_move_btn = QPushButton(tr(self.language, "menu.move.stop"))
        self.stop_move_btn.setMinimumWidth(180)
        self.stop_move_btn.clicked.connect(self._on_toggle_move_clicked)
        form_layout.addRow(self._l("移动控制", "Movement", "移動", "이동", "Déplacement"), self.stop_move_btn)

        self.move_scope_hint = QLabel(
            self._l(
                "说明：此处“停止/恢复移动”会作用于全部实例；右键菜单仅作用于当前实例。",
                "Note: Stop/Resume here applies to all instances; right-click menu applies to current instance only.",
                "注: ここでの停止/再開は全インスタンス対象、右クリックメニューは現在のインスタンスのみです。",
                "참고: 여기의 정지/재개는 전체 인스턴스에 적용되며, 우클릭 메뉴는 현재 인스턴스에만 적용됩니다.",
                "Remarque : ici, Arrêter/Reprendre s'applique à toutes les instances ; le menu contextuel s'applique uniquement à l'instance courante.",
            )
        )
        self.move_scope_hint.setWordWrap(True)
        self.move_scope_hint.setObjectName("FormHint")
        form_layout.addRow("", self.move_scope_hint)
        form_layout.addRow(self._create_form_separator())

        self.follow_checkbox = QCheckBox(tr(self.language, "menu.follow_mouse"))
        self.follow_checkbox.setChecked(self.pet.state.follow_mouse)
        self.follow_checkbox.toggled.connect(self._on_follow_toggled)
        form_layout.addRow(tr(self.language, "menu.follow_mouse"), self.follow_checkbox)
        form_layout.addRow(self._create_form_separator())

        scale_slider_row = QWidget()
        scale_slider_layout = QHBoxLayout(scale_slider_row)
        scale_slider_layout.setContentsMargins(0, 0, 0, 0)
        scale_slider_layout.setSpacing(8)

        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(int(round(SCALE_MIN * 10)))
        self.scale_slider.setMaximum(int(round(SCALE_MAX * 10)))
        self.scale_slider.setSingleStep(1)
        self.scale_slider.setPageStep(1)
        self.scale_slider.setValue(int(round(self.pet.scale_factor * 10)))
        self.scale_slider.valueChanged.connect(self._on_scale_slider_changed)

        self.scale_value_label = QLabel(f"{self.scale_slider.value() / 10:.1f}x")
        self.scale_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.scale_value_label.setMinimumWidth(44)

        scale_slider_layout.addWidget(self.scale_slider, stretch=1)
        scale_slider_layout.addWidget(self.scale_value_label, stretch=0)
        form_layout.addRow(tr(self.language, "menu.scale"), scale_slider_row)
        form_layout.addRow(self._create_form_separator())

        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItem(tr(self.language, "menu.display.always_top"), userData=DISPLAY_MODE_ALWAYS_ON_TOP)
        self.display_mode_combo.addItem(tr(self.language, "menu.display.fullscreen_hide"), userData=DISPLAY_MODE_FULLSCREEN_HIDE)
        self.display_mode_combo.addItem(tr(self.language, "menu.display.desktop_only"), userData=DISPLAY_MODE_DESKTOP_ONLY)
        self.display_mode_combo.installEventFilter(self)
        self.display_mode_combo.currentIndexChanged.connect(self._on_display_mode_combo_changed)
        form_layout.addRow(tr(self.language, "menu.display_mode"), self.display_mode_combo)
        form_layout.addRow(self._create_form_separator())

        self.instance_count_spin = QSpinBox()
        self.instance_count_spin.setRange(INSTANCE_COUNT_MIN, INSTANCE_COUNT_MAX)
        self.instance_count_spin.setSingleStep(1)
        self.instance_count_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.instance_count_spin.setAccelerated(True)
        self.instance_count_spin.setKeyboardTracking(False)
        self.instance_count_spin.installEventFilter(self)
        self.instance_count_spin.setValue(self._resolve_pet_instance_count())
        self.instance_count_spin.valueChanged.connect(self._on_instance_count_spin_changed)

        instance_count_row = QWidget()
        instance_count_layout = QHBoxLayout(instance_count_row)
        instance_count_layout.setContentsMargins(0, 0, 0, 0)
        instance_count_layout.setSpacing(6)

        self.instance_count_down_btn = QPushButton("▼")
        self.instance_count_down_btn.setObjectName("SpinArrowButton")
        self.instance_count_down_btn.setFixedWidth(40)
        self.instance_count_down_btn.clicked.connect(self._on_instance_count_decrease)

        self.instance_count_up_btn = QPushButton("▲")
        self.instance_count_up_btn.setObjectName("SpinArrowButton")
        self.instance_count_up_btn.setFixedWidth(40)
        self.instance_count_up_btn.clicked.connect(self._on_instance_count_increase)

        instance_count_layout.addWidget(self.instance_count_spin, stretch=1)
        instance_count_layout.addWidget(self.instance_count_up_btn, stretch=0)
        instance_count_layout.addWidget(self.instance_count_down_btn, stretch=0)

        form_layout.addRow(self._l("多开数量", "Instance Count", "インスタンス数", "인스턴스 수", "Nombre d'instances"), instance_count_row)
        form_layout.addRow(self._create_form_separator())

        opacity_row = QWidget()
        opacity_layout = QHBoxLayout(opacity_row)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.setSpacing(8)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(OPACITY_PERCENT_MIN, OPACITY_PERCENT_MAX)
        self.opacity_slider.setSingleStep(1)
        self.opacity_slider.setPageStep(1)
        self.opacity_slider.setValue(self._resolve_pet_opacity_percent())
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)

        self.opacity_value_label = QLabel(f"{self.opacity_slider.value()}%")
        self.opacity_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.opacity_value_label.setMinimumWidth(44)

        opacity_layout.addWidget(self.opacity_slider, stretch=1)
        opacity_layout.addWidget(self.opacity_value_label, stretch=0)
        form_layout.addRow(tr(self.language, "menu.opacity"), opacity_row)
        form_layout.addRow(self._create_form_separator())

        self.autostart_checkbox = QCheckBox(tr(self.language, "menu.autostart"))
        self.autostart_checkbox.setChecked(self.pet.get_autostart_enabled())
        self.autostart_checkbox.toggled.connect(self._on_autostart_toggled)
        form_layout.addRow(self._l("系统启动", "System Startup", "システム起動", "시스템 시작", "Démarrage système"), self.autostart_checkbox)
        form_layout.addRow(self._create_form_separator())

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setObjectName("ApiKeyEdit")
        self.api_key_edit.setPlaceholderText(
            self._l(
                "请输入 DeepSeek API Key",
                "Enter DeepSeek API key",
                "DeepSeek API キーを入力",
                "DeepSeek API 키 입력",
                "Saisissez la clé API DeepSeek",
            )
        )
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        saved_api_key = self.settings_store.get_api_key()
        self.api_key_edit.setText(saved_api_key)

        api_key_row = QWidget()
        api_key_layout = QHBoxLayout(api_key_row)
        api_key_layout.setContentsMargins(0, 0, 0, 0)
        api_key_layout.setSpacing(6)

        self.api_key_visibility_btn = QToolButton()
        self.api_key_visibility_btn.setText("👁")
        self.api_key_visibility_btn.setCheckable(True)
        self.api_key_visibility_btn.setToolTip(
            self._l(
                "显示/隐藏 API Key",
                "Show/Hide API key",
                "API キーの表示/非表示",
                "API 키 표시/숨기기",
                "Afficher/Masquer la clé API",
            )
        )
        self.api_key_visibility_btn.setFixedWidth(36)
        self.api_key_visibility_btn.toggled.connect(self._on_api_key_visibility_toggled)

        self.api_key_confirm_btn = QPushButton(self._l("确认", "Confirm", "確認", "확인", "Confirmer"))
        self.api_key_confirm_btn.setObjectName("ApiKeyActionBtn")
        self.api_key_confirm_btn.setMinimumWidth(90)
        self.api_key_confirm_btn.clicked.connect(self._on_api_key_confirm_clicked)

        self.api_key_modify_btn = QPushButton(
            self._l(
                "修改API_key",
                "Edit API Key",
                "API キーを編集",
                "API 키 수정",
                "Modifier la clé API",
            )
        )
        self.api_key_modify_btn.setObjectName("ApiKeyActionBtn")
        self.api_key_modify_btn.setMinimumWidth(120)
        self.api_key_modify_btn.clicked.connect(self._on_api_key_modify_clicked)

        api_key_layout.addWidget(self.api_key_edit, stretch=1)
        api_key_layout.addWidget(self.api_key_visibility_btn, stretch=0)
        api_key_layout.addWidget(self.api_key_confirm_btn, stretch=0)
        api_key_layout.addWidget(self.api_key_modify_btn, stretch=0)

        has_saved_api_key = bool(saved_api_key)
        self.api_key_edit.setReadOnly(has_saved_api_key)
        self.api_key_confirm_btn.setEnabled(not has_saved_api_key)
        self.api_key_modify_btn.setEnabled(has_saved_api_key)

        form_layout.addRow(
            self._l(
                "DeepSeek API 密钥",
                "DeepSeek API Key",
                "DeepSeek API キー",
                "DeepSeek API 키",
                "Clé API DeepSeek",
            ),
            api_key_row,
        )
        form_layout.addRow(self._create_form_separator())

        self.close_behavior_combo = QComboBox()
        self.close_behavior_combo.addItem(self._l("每次询问", "Ask every time", "毎回確認", "매번 묻기", "Toujours demander"), userData="ask")
        self.close_behavior_combo.addItem(self._l("直接退出应用", "Quit directly", "そのまま終了", "바로 종료", "Quitter directement"), userData="quit")
        self.close_behavior_combo.addItem(self._l("最小化到系统托盘", "Minimize to tray", "トレイに最小化", "트레이로 최소화", "Réduire dans la zone de notification"), userData="tray")
        behavior = self.settings_store.get_close_behavior()
        for i in range(self.close_behavior_combo.count()):
            if self.close_behavior_combo.itemData(i) == behavior:
                self.close_behavior_combo.setCurrentIndex(i)
                break
        self.close_behavior_combo.installEventFilter(self)
        self.close_behavior_combo.currentIndexChanged.connect(self._on_close_behavior_changed)
        form_layout.addRow(self._l("点击“×”行为", "Click × behavior", "× ボタン動作", "× 버튼 동작", "Comportement du bouton ×"), self.close_behavior_combo)
        form_layout.addRow(self._create_form_separator())

        self.quit_btn = QPushButton(self._l("退出应用程序", "Quit Application", "アプリを終了", "앱 종료", "Quitter l'application"))
        self.quit_btn.setMinimumWidth(180)
        self.quit_btn.clicked.connect(self.request_quit)
        form_layout.addRow(self._l("应用操作", "App Actions", "アプリ操作", "앱 동작", "Actions de l'application"), self.quit_btn)

        layout.addWidget(form_card)
        layout.addStretch(1)

        self.settings_scroll_area.setWidget(settings_content)
        page_layout.addWidget(self.settings_scroll_area)
        return page

    def _create_form_separator(self) -> QWidget:
        """创建设置页分隔线。用于在功能模块间留出间隔。"""
        """EN: Create a settings page divider. Used to leave gaps between functional modules."""
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 14, 0, 14)
        wrapper_layout.setSpacing(0)

        line = QFrame()
        line.setObjectName("FormSeparator")
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        wrapper_layout.addWidget(line)
        return wrapper

    def _build_music_page(self) -> QWidget:
        """构建音乐播放器页面。包含播放控制、进度条、播放模式、音量与播放列表。"""
        """EN: Build the music player page. Contains playback controls, progress bars, play modes, volume, and playlists."""
        from .music import PLAY_MODE_LIST, PLAY_MODE_SINGLE, PLAY_MODE_RANDOM, MODE_ICONS

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(16, 16, 16, 16)
        page_layout.setSpacing(12)

        title = QLabel(tr(self.language, "app.music"))
        title.setObjectName("SectionTitle")
        page_layout.addWidget(title)

        # ---- 歌曲名 ----
        # EN: ---- Song title ----
        self.music_track_label = QLabel(self._l("（未播放）", "(Not Playing)", "（未再生）", "(재생 안 함)", "(Aucune lecture)"))
        self.music_track_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.music_track_label.setObjectName("MusicTrackLabel")
        self.music_track_label.setWordWrap(True)
        page_layout.addWidget(self.music_track_label)

        # ---- 进度条 ----
        # EN: Progress Bar
        self.music_progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.music_progress_slider.setRange(0, 0)
        self.music_progress_slider.setSingleStep(1000)
        self.music_progress_slider.setObjectName("MusicProgressSlider")
        self._music_seeking = False
        self.music_progress_slider.sliderPressed.connect(lambda: setattr(self, "_music_seeking", True))
        self.music_progress_slider.sliderReleased.connect(self._on_music_seek)
        page_layout.addWidget(self.music_progress_slider)

        # 时间标签行
        # EN: Time Stamp Row
        time_row = QWidget()
        time_layout = QHBoxLayout(time_row)
        time_layout.setContentsMargins(0, 0, 0, 0)
        self.music_pos_label = QLabel("0:00")
        self.music_dur_label = QLabel("0:00")
        time_layout.addWidget(self.music_pos_label)
        time_layout.addStretch(1)
        time_layout.addWidget(self.music_dur_label)
        page_layout.addWidget(time_row)

        # ---- 主控按钮行 ----
        # EN: ----Master Button Row----
        ctrl_row = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_row)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(10)

        self.music_prev_btn = QPushButton(tr(self.language, "menu.music.prev"))
        self.music_play_btn = QPushButton(tr(self.language, "menu.music.play"))
        self.music_play_btn.setObjectName("MusicPlayBtn")
        self.music_next_btn = QPushButton(tr(self.language, "menu.music.next"))

        self.music_prev_btn.clicked.connect(self._on_music_prev)
        self.music_play_btn.clicked.connect(self._on_music_toggle_pause)
        self.music_next_btn.clicked.connect(self._on_music_next)

        ctrl_layout.addWidget(self.music_prev_btn, stretch=1)
        ctrl_layout.addWidget(self.music_play_btn, stretch=2)
        ctrl_layout.addWidget(self.music_next_btn, stretch=1)
        page_layout.addWidget(ctrl_row)

        # ---- 播放模式按钮 ----
        # EN: ---- Play Mode Button ----
        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)

        self._music_mode_btns = {}
        for mode_key, mode_label in [
            (PLAY_MODE_LIST, f"{MODE_ICONS[PLAY_MODE_LIST]} {tr(self.language, 'menu.music.mode.list')}"),
            (PLAY_MODE_SINGLE, f"{MODE_ICONS[PLAY_MODE_SINGLE]} {tr(self.language, 'menu.music.mode.single')}"),
            (PLAY_MODE_RANDOM, f"{MODE_ICONS[PLAY_MODE_RANDOM]} {tr(self.language, 'menu.music.mode.random')}"),
        ]:
            btn = QPushButton(mode_label)
            btn.setObjectName("MusicModeBtn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked=False, m=mode_key: self._on_music_set_mode(m))
            self._music_mode_btns[mode_key] = btn
            mode_layout.addWidget(btn, stretch=1)
        page_layout.addWidget(mode_row)

        # ---- 音量行 ----
        # EN: ---- Volume Row ----
        vol_row = QWidget()
        vol_layout = QHBoxLayout(vol_row)
        vol_layout.setContentsMargins(0, 0, 0, 0)
        vol_layout.setSpacing(8)

        vol_icon = QLabel("🔈")
        self.music_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.music_vol_slider.setRange(0, 100)
        self.music_vol_slider.setValue(50)
        self.music_vol_slider.setObjectName("MusicVolSlider")
        self.music_vol_label = QLabel("50%")
        self.music_vol_label.setMinimumWidth(44)
        self.music_vol_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.music_vol_slider.valueChanged.connect(self._on_music_vol_changed)

        vol_layout.addWidget(vol_icon)
        vol_layout.addWidget(self.music_vol_slider, stretch=1)
        vol_layout.addWidget(self.music_vol_label)
        page_layout.addWidget(vol_row)

        # ---- 播放列表 ----
        # EN: P l a y l i s t
        list_title = QLabel(self._l("播放列表", "Playlist", "プレイリスト", "재생목록", "Liste de lecture"))
        list_title.setObjectName("MusicListTitle")
        page_layout.addWidget(list_title)

        list_action_row = QWidget()
        list_action_layout = QHBoxLayout(list_action_row)
        list_action_layout.setContentsMargins(0, 0, 0, 0)
        list_action_layout.setSpacing(8)

        self.music_add_btn = QPushButton(self._l("加入本地歌曲", "Add Local Tracks", "ローカル曲を追加", "로컬 곡 추가", "Ajouter des morceaux locaux"))
        self.music_add_btn.setObjectName("MusicListActionBtn")
        self.music_add_btn.clicked.connect(self._on_music_add_local)

        self.music_remove_btn = QPushButton(self._l("删除列表歌曲", "Remove Tracks", "曲を削除", "곡 삭제", "Supprimer des morceaux"))
        self.music_remove_btn.setObjectName("MusicListActionBtn")
        self.music_remove_btn.clicked.connect(self._on_music_remove_selected)

        list_action_layout.addWidget(self.music_add_btn, stretch=1)
        list_action_layout.addWidget(self.music_remove_btn, stretch=1)
        page_layout.addWidget(list_action_row)

        self.music_list_widget = QListWidget()
        self.music_list_widget.setObjectName("MusicListWidget")
        self.music_list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.music_list_widget.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.music_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._music_delete_delegate = MusicDeleteCheckDelegate(self.music_list_widget)
        self.music_list_widget.setItemDelegate(self._music_delete_delegate)
        self.music_list_widget.viewport().installEventFilter(self)
        self.music_list_widget.itemClicked.connect(self._on_music_list_item_clicked)
        self.music_list_widget.model().rowsMoved.connect(self._on_music_list_rows_moved)
        self.music_list_widget.customContextMenuRequested.connect(self._on_music_list_context_menu)
        page_layout.addWidget(self.music_list_widget, stretch=1)

        self._music_delete_mode = False

        self.music_confirm_delete_btn = QPushButton(self._l("确认删除", "Confirm Delete", "削除を確定", "삭제 확인", "Confirmer la suppression"))
        self.music_confirm_delete_btn.setObjectName("MusicListActionBtn")
        self.music_confirm_delete_btn.clicked.connect(self._on_music_confirm_batch_remove)
        self.music_confirm_delete_btn.hide()

        self.music_cancel_delete_btn = QPushButton(self._l("取消删除", "Cancel Delete", "削除をキャンセル", "삭제 취소", "Annuler la suppression"))
        self.music_cancel_delete_btn.setObjectName("MusicListActionBtn")
        self.music_cancel_delete_btn.clicked.connect(self._on_music_cancel_batch_remove)
        self.music_cancel_delete_btn.hide()

        self.music_delete_mode_hint = QLabel(
            self._l(
                "已进入删除模式，请勾选歌曲后确认",
                "Delete mode is active. Select tracks and confirm.",
                "削除モードです。曲を選択して確定してください。",
                "삭제 모드입니다. 곡을 선택 후 확인하세요.",
                "Mode suppression activé. Sélectionnez des morceaux puis confirmez.",
            )
        )
        self.music_delete_mode_hint.setObjectName("MusicDeleteModeHint")
        self.music_delete_mode_hint.hide()

        list_action_layout.addWidget(self.music_confirm_delete_btn, stretch=1)
        list_action_layout.addWidget(self.music_cancel_delete_btn, stretch=1)
        page_layout.addWidget(self.music_delete_mode_hint)

        # 初始化列表与状态
        # EN: Initialization List and Status
        self._music_refresh_playlist()
        if self.music_player is not None:
            self._sync_music_ui_from_player()

        return page

    def _build_chat_page(self) -> QWidget:
        """构建聊天入口页。应用内仅保留跳转到独立聊天窗口。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        tip = QLabel(self._l("聊天仅在独立窗口中使用", "Chat is available only in standalone window", "チャットは独立ウィンドウのみで利用できます", "채팅은 독립 창에서만 사용할 수 있습니다", "Le chat est disponible uniquement dans une fenêtre autonome"))
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setObjectName("SectionTitle")
        layout.addWidget(tip, stretch=1)

        self.open_chat_window_btn = QPushButton(self._l("打开独立聊天窗口", "Open Chat Window", "チャットウィンドウを開く", "채팅 창 열기", "Ouvrir la fenêtre de chat"))
        self.open_chat_window_btn.setObjectName("MusicListActionBtn")
        self.open_chat_window_btn.setFixedHeight(36)
        self.open_chat_window_btn.clicked.connect(self._on_open_chat_window)
        layout.addWidget(self.open_chat_window_btn, stretch=0)

        return page

    def _open_page(self, page_index: int):
        self.pages.setCurrentIndex(page_index)

    def _on_chat_nav_clicked(self):
        self._on_open_chat_window()
        if hasattr(self, "pages"):
            self.pages.setCurrentIndex(self._last_non_chat_page_index)

    def _build_about_page(self) -> QWidget:
        """构建关于页。中心显示 648x648 的 ameath.gif。"""
        """EN: Build an about page. The center displays ameath.gif of 648x648."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(16, 16, 16, 16)
        page_layout.setSpacing(0)

        self.about_scroll_area = QScrollArea()
        self.about_scroll_area.setObjectName("AboutScrollArea")
        self.about_scroll_area.setWidgetResizable(True)
        self.about_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.about_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.about_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.about_page = QWidget()
        self.about_page.setObjectName("AboutPageContent")
        layout = QVBoxLayout(self.about_page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel(tr(self.language, "app.about"))
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.gif_wrap = QFrame()
        self.gif_wrap.setObjectName("Card")
        gif_layout = QVBoxLayout(self.gif_wrap)
        gif_layout.setContentsMargins(16, 16, 16, 16)

        self.about_gif_label = QLabel()
        self.about_gif_label.setMinimumSize(96, 96)
        self.about_gif_label.setMaximumSize(648, 648)
        self.about_gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.about_gif_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.about_movie = QMovie(str(ABOUT_GIF_PATH))
        self.about_movie.setParent(self)
        self.about_movie.setCacheMode(QMovie.CacheMode.CacheNone)
        self.about_movie.setScaledSize(QSize(648, 648))
        self.about_gif_label.setMovie(self.about_movie)
        self.about_movie.start()

        gif_layout.addWidget(self.about_gif_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.gif_wrap, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        language_row = QWidget()
        language_row_layout = QHBoxLayout(language_row)
        language_row_layout.setContentsMargins(0, 0, 0, 0)
        language_row_layout.setSpacing(8)
        self.about_language_label = QLabel(tr(self.language, "about.language"))
        self.about_language_combo = QComboBox()
        for code, name in get_language_items():
            self.about_language_combo.addItem(name, userData=code)
        lang_index = self.about_language_combo.findData(self.language)
        if lang_index >= 0:
            self.about_language_combo.setCurrentIndex(lang_index)
        self.about_language_combo.currentIndexChanged.connect(self._on_language_combo_changed)
        language_row_layout.addWidget(self.about_language_label)
        language_row_layout.addWidget(self.about_language_combo, stretch=1)
        layout.addWidget(language_row)

        text_card = QFrame()
        text_card.setObjectName("Card")
        text_layout = QVBoxLayout(text_card)
        text_layout.setContentsMargins(16, 16, 16, 16)

        self.about_text_edit = QTextEdit()
        self.about_text_edit.setObjectName("AboutText")
        self.about_text_edit.setReadOnly(True)
        self.about_text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.about_text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.about_text_edit.setHtml(
            "<div style='text-align:center; line-height:1.6;'>"
            + self.about_text_content_map.get(self.language, self.about_text_content_map["zh-CN"]).replace("\n", "<br>")
            + "</div>"
        )
        self.about_text_edit.setMinimumHeight(180)
        text_layout.addWidget(self.about_text_edit)

        layout.addWidget(text_card)
        layout.addStretch(1)
        self.about_scroll_area.setWidget(self.about_page)
        page_layout.addWidget(self.about_scroll_area)
        return page

    def _apply_theme(self):
        """应用浅粉白主基调样式。"""
        """EN: Apply a light pink main tone style."""
        style = """
            QMainWindow {
                background: #fff8fb;
            }
            QScrollArea#SettingsScrollArea, QScrollArea#AboutScrollArea {
                background: #fff8fb;
                border: none;
            }
            QWidget#SettingsPageContent, QWidget#AboutPageContent {
                background: #fff8fb;
            }
            QFrame#NavCard, QFrame#Card {
                background: #fffdfd;
                border: 1px solid #f5d9e6;
                border-radius: 12px;
            }
            QFrame#FormSeparator {
                background: #f3d8e4;
                max-height: 1px;
                min-height: 1px;
                border: none;
            }
            QLabel {
                color: #6e4b5a;
                font-size: 16px;
            }
            QLabel#SectionTitle {
                font-size: 26px;
                font-weight: 700;
                color: #8e5f73;
            }
            QTextEdit#AboutText {
                font-size: 22px;
                line-height: 1.6;
                font-weight: 600;
                color: #7a4b60;
                background: #ffffff;
                border: 1px solid #f0c7d8;
                border-radius: 8px;
                padding: 8px;
            }
            QTextEdit#AboutText QScrollBar:vertical {
                background: #fff4f9;
                width: 12px;
                margin: 4px 2px 4px 2px;
                border-radius: 6px;
            }
            QTextEdit#AboutText QScrollBar::handle:vertical {
                background: #f6cde0;
                min-height: 28px;
                border-radius: 6px;
                border: 1px solid #efbdd5;
            }
            QTextEdit#AboutText QScrollBar::handle:vertical:hover {
                background: #efb8d2;
            }
            QTextEdit#AboutText QScrollBar::add-line:vertical,
            QTextEdit#AboutText QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
                border: none;
            }
            QTextEdit#AboutText QScrollBar::add-page:vertical,
            QTextEdit#AboutText QScrollBar::sub-page:vertical {
                background: transparent;
            }
            QScrollArea#SettingsScrollArea QScrollBar:vertical,
            QScrollArea#AboutScrollArea QScrollBar:vertical {
                background: #fff4f9;
                width: 12px;
                margin: 4px 2px 4px 2px;
                border-radius: 6px;
            }
            QScrollArea#SettingsScrollArea QScrollBar::handle:vertical,
            QScrollArea#AboutScrollArea QScrollBar::handle:vertical {
                background: #f6cde0;
                min-height: 28px;
                border-radius: 6px;
                border: 1px solid #efbdd5;
            }
            QScrollArea#SettingsScrollArea QScrollBar::handle:vertical:hover,
            QScrollArea#AboutScrollArea QScrollBar::handle:vertical:hover {
                background: #efb8d2;
            }
            QScrollArea#SettingsScrollArea QScrollBar::add-line:vertical,
            QScrollArea#SettingsScrollArea QScrollBar::sub-line:vertical,
            QScrollArea#AboutScrollArea QScrollBar::add-line:vertical,
            QScrollArea#AboutScrollArea QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
                border: none;
            }
            QScrollArea#SettingsScrollArea QScrollBar::add-page:vertical,
            QScrollArea#SettingsScrollArea QScrollBar::sub-page:vertical,
            QScrollArea#AboutScrollArea QScrollBar::add-page:vertical,
            QScrollArea#AboutScrollArea QScrollBar::sub-page:vertical {
                background: transparent;
            }
            QPushButton {
                background: #ffeaf3;
                border: 1px solid #f3c6da;
                border-radius: 10px;
                padding: 10px 14px;
                min-height: 44px;
                color: #7a4b60;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton#NavButton {
                font-size: 18px;
                font-weight: 700;
            }
            QPushButton#SpinArrowButton {
                font-size: 18px;
                font-weight: 700;
                min-height: 48px;
                padding: 6px 0px;
            }
            QPushButton:hover {
                background: #ffdfeF;
            }
            QComboBox, QTextEdit {
                background: #ffffff;
                border: 1px solid #f0c7d8;
                border-radius: 8px;
                padding: 8px;
                color: #6d4657;
                font-size: 16px;
            }
            QLineEdit#ApiKeyEdit {
                background: #ffffff;
                color: #111111;
                border: 1px solid #111111;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                selection-background-color: #ececec;
                selection-color: #111111;
            }
            QLineEdit#ApiKeyEdit:focus {
                background: #ffffff;
                color: #111111;
                border: 2px solid #111111;
            }
            QPushButton#ApiKeyActionBtn {
                min-height: 36px;
                font-size: 14px;
                padding: 6px 10px;
            }
            QSpinBox {
                background: #ffffff;
                border: 1px solid #f0c7d8;
                border-radius: 8px;
                padding: 8px;
                color: #6d4657;
                font-size: 16px;
                min-height: 24px;
            }
            QCheckBox {
                color: #6d4657;
                font-size: 16px;
            }
            QLabel#MusicTrackLabel {
                font-size: 18px;
                font-weight: 600;
                color: #7a4b60;
                padding: 4px 0;
            }
            QLabel#MusicListTitle {
                font-size: 15px;
                font-weight: 600;
                color: #8e6070;
            }
            QSlider#MusicProgressSlider::groove:horizontal,
            QSlider#MusicVolSlider::groove:horizontal {
                height: 6px;
                background: #f3d8e4;
                border-radius: 3px;
            }
            QSlider#MusicProgressSlider::handle:horizontal,
            QSlider#MusicVolSlider::handle:horizontal {
                background: #e08ab0;
                border: 1px solid #d070a0;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider#MusicProgressSlider::sub-page:horizontal,
            QSlider#MusicVolSlider::sub-page:horizontal {
                background: #e8a0c0;
                border-radius: 3px;
            }
            QPushButton#MusicPlayBtn {
                font-size: 18px;
                font-weight: 700;
                background: #ffd6ec;
            }
            QPushButton#MusicModeBtn {
                font-size: 15px;
                min-height: 36px;
            }
            QPushButton#MusicModeBtn:checked {
                background: #f9c0d8;
                border: 1.5px solid #e080b0;
                color: #8e3060;
            }
            QPushButton#MusicListActionBtn {
                min-height: 36px;
                font-size: 15px;
            }
            QListWidget#MusicListWidget {
                background: #fff8fb;
                border: 1px solid #f0c7d8;
                border-radius: 8px;
                font-size: 15px;
                color: #6d4657;
            }
            QListWidget#MusicListWidget::item:selected {
                background: #ffd6ec;
                color: #7a2550;
            }
            QListWidget#MusicListWidget::item:hover {
                background: #fff0f7;
            }
            QLabel#MusicDeleteModeHint {
                color: #1d4ed8;
                font-size: 14px;
                font-weight: 600;
                background: #dbeafe;
                border: 1px solid #93c5fd;
                border-radius: 8px;
                padding: 8px 10px;
            }
            """
        self.setStyleSheet(style)

    def _on_follow_toggled(self, checked: bool):
        """设置页跟随开关回调。"""
        """EN: The settings page follows the callback of the switch."""
        if self.pet.state.follow_mouse != checked:
            self.pet.on_toggle_follow()

    def _on_scale_slider_changed(self, slider_value: int):
        """设置页缩放滑块变化回调。"""
        """EN: Set the page zoom slider change callback."""
        scale = float(slider_value) / 10
        self.scale_value_label.setText(f"{scale:.1f}x")
        self.pet.on_set_scale(scale)

    def _on_autostart_toggled(self, checked: bool):
        """设置页开机自启开关回调。"""
        """EN: Set the page boot auto-open switch callback."""
        self.pet.on_toggle_autostart(checked)

    def _on_toggle_move_clicked(self):
        """设置页移动开关按钮回调。作用于全部实例。"""
        """EN: Set page move switch button callback. Acts on all instances."""
        if hasattr(self.pet, "on_toggle_move_all") and callable(self.pet.on_toggle_move_all):
            self.pet.on_toggle_move_all()
            return

        if hasattr(self.pet, "on_stop_move") and callable(self.pet.on_stop_move):
            self.pet.on_stop_move()

    def _on_display_mode_combo_changed(self):
        """设置页显示模式下拉变化回调。"""
        """EN: Setting page display mode drop-down change callback."""
        mode = self.display_mode_combo.currentData()
        if isinstance(mode, str):
            self.pet.on_set_display_mode(mode)

    def _on_instance_count_spin_changed(self, value: int):
        """设置页实例数量变化回调。"""
        """EN: Setup page instance number change callback."""
        self.pet.on_set_instance_count(value)

    def _on_instance_count_decrease(self):
        """多开数量减少按钮回调。"""
        """EN: More open quantity reduction button callback."""
        self.instance_count_spin.stepDown()

    def _on_instance_count_increase(self):
        """多开数量增加按钮回调。"""
        """EN: More open quantity increase button callback."""
        self.instance_count_spin.stepUp()

    def _on_opacity_slider_changed(self, value: int):
        """设置页透明度滑块变化回调。"""
        """EN: Set the page transparency slider variation callback."""
        self.opacity_value_label.setText(f"{int(value)}%")
        self.pet.on_set_opacity_percent(int(value))

    def _on_close_behavior_changed(self):
        """设置页关闭行为配置回调。"""
        """EN: The settings page closes the behavior configuration callback."""
        behavior = self.close_behavior_combo.currentData()
        if isinstance(behavior, str):
            self.settings_store.set_close_behavior(behavior)

    def _on_api_key_editing_finished(self):
        """设置页 API Key 输入回调。"""
        """EN: Settings page API key input callback."""
        if not hasattr(self, "api_key_edit"):
            return
        api_key = str(self.api_key_edit.text() or "").strip()
        self.settings_store.set_api_key(api_key)
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)

    def _on_api_key_confirm_clicked(self):
        """确认保存 API Key，并锁定输入框。"""
        """EN: Confirm and save API key, then lock input."""
        self._on_api_key_editing_finished()
        if hasattr(self, "api_key_edit"):
            self.api_key_edit.setReadOnly(True)
        if hasattr(self, "api_key_confirm_btn"):
            self.api_key_confirm_btn.setEnabled(False)
        if hasattr(self, "api_key_modify_btn"):
            self.api_key_modify_btn.setEnabled(True)

    def _on_api_key_modify_clicked(self):
        """进入 API Key 修改模式。"""
        """EN: Enter API key edit mode."""
        if hasattr(self, "api_key_edit"):
            self.api_key_edit.setReadOnly(False)
            self.api_key_edit.setFocus()
            self.api_key_edit.selectAll()
        if hasattr(self, "api_key_confirm_btn"):
            self.api_key_confirm_btn.setEnabled(True)
        if hasattr(self, "api_key_modify_btn"):
            self.api_key_modify_btn.setEnabled(False)

    def _on_api_key_visibility_toggled(self, checked: bool):
        """设置页 API Key 可见性切换回调。"""
        """EN: Settings page API key visibility toggle callback."""
        if not hasattr(self, "api_key_edit"):
            return
        if checked:
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            if hasattr(self, "api_key_visibility_btn"):
                self.api_key_visibility_btn.setText("🙈")
            return
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        if hasattr(self, "api_key_visibility_btn"):
            self.api_key_visibility_btn.setText("👁")

    def eventFilter(self, watched, event):
        """过滤指定下拉框的滚轮事件，避免滚轮误改值。"""
        """EN: Filter the scrollwheel events of the specified drop-down box to avoid scrollwheel misalignment."""
        if (
            hasattr(self, "music_list_widget")
            and watched is self.music_list_widget.viewport()
            and self._music_delete_mode
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            item = self.music_list_widget.itemAt(event.pos())
            if item is not None:
                self._set_music_item_checked(item, not self._is_music_item_checked(item))
                self.music_list_widget.viewport().update()
            return True

        if event.type() == QEvent.Type.Wheel and watched in {
            self.display_mode_combo,
            self.close_behavior_combo,
            self.instance_count_spin,
        }:
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def show_window(self):
        """显示并激活主界面。"""
        """EN: Displays and activates the main screen."""
        self.show()
        self.raise_()
        self.activateWindow()
        self._update_about_gif_size()

    def _on_page_changed(self, index: int):
        """页面切换回调。在关于页显示时同步刷新 GIF 大小。"""
        """EN: Page toggle callback. Synchronously refreshes the GIF size when the About page is displayed."""
        if index == self.PAGE_CHAT:
            self._on_open_chat_window()
            self.pages.setCurrentIndex(self._last_non_chat_page_index)
            return

        self._last_non_chat_page_index = index

        if index == self.PAGE_ABOUT:
            self._update_about_gif_size()

    def _update_about_gif_size(self):
        """根据当前界面尺寸更新关于页 GIF 显示大小。全屏时上限 648x648。"""
        """EN: Update the about page GIF display size according to the current interface size. Fullscreen time is capped at 648x648."""
        if not hasattr(self, "about_gif_label"):
            return

        if not hasattr(self, "about_movie") or self.about_movie is None:
            return

        if self.pages.currentIndex() != self.PAGE_ABOUT:
            return

        if hasattr(self, "about_scroll_area"):
            page_rect = self.about_scroll_area.viewport().contentsRect()
        else:
            page_rect = self.about_page.contentsRect()
        available_width = max(96, page_rect.width() - 64)
        available_height = max(96, int(page_rect.height() * 0.55))

        target_size = min(648, available_width, available_height)
        target_size = max(96, target_size)

        self.about_gif_label.setFixedSize(target_size, target_size)
        self.about_movie.setScaledSize(QSize(target_size, target_size))
        self.about_gif_label.update()

    def _on_open_chat_window(self):
        if callable(self.on_open_chat_window):
            self.on_open_chat_window()

    def set_tray_controller(self, tray_controller):
        """设置托盘控制器引用。用于最小化到托盘时通知。"""
        """EN: Sets the tray controller reference. Used to minimize notifications when reaching the pallet."""
        self.tray_controller = tray_controller

    def prepare_for_exit(self):
        """准备退出。标记当前窗口允许直接关闭。"""
        """EN: Ready to exit. Marking the current window allows to close directly."""
        self._is_exiting = True

        if hasattr(self, "music_list_widget") and self.music_list_widget is not None:
            try:
                self.music_list_widget.viewport().removeEventFilter(self)
            except Exception:
                pass
            try:
                self.music_list_widget.customContextMenuRequested.disconnect(self._on_music_list_context_menu)
            except Exception:
                pass
            try:
                for i in range(self.music_list_widget.count()):
                    item = self.music_list_widget.item(i)
                    row_widget = self.music_list_widget.itemWidget(item)
                    if row_widget is not None:
                        row_widget.deleteLater()
            except Exception:
                pass

        if self.music_player is not None:
            try:
                self.music_player.track_changed.disconnect(self._on_music_track_changed)
            except Exception:
                pass
            try:
                self.music_player.state_changed.disconnect(self._on_music_state_changed)
            except Exception:
                pass
            try:
                self.music_player.playlist_reordered.disconnect(self._on_music_playlist_reordered)
            except Exception:
                pass
            try:
                self.music_player.volume_changed.disconnect(self._on_music_volume_changed)
            except Exception:
                pass
            try:
                self.music_player.mode_changed.disconnect(self._on_music_mode_changed)
            except Exception:
                pass
            try:
                self.music_player.duration_changed.disconnect(self._on_music_duration_changed)
            except Exception:
                pass
            try:
                self.music_player.position_changed.disconnect(self._on_music_position_changed)
            except Exception:
                pass

        if hasattr(self.pet, "language_changed"):
            try:
                self.pet.language_changed.disconnect(self._on_pet_language_changed)
            except Exception:
                pass

        if hasattr(self, "music_list_widget") and self.music_list_widget is not None:
            try:
                self.music_list_widget.itemClicked.disconnect(self._on_music_list_item_clicked)
            except Exception:
                pass
            try:
                self.music_list_widget.model().rowsMoved.disconnect(self._on_music_list_rows_moved)
            except Exception:
                pass

        if hasattr(self, "api_key_visibility_btn"):
            try:
                self.api_key_visibility_btn.toggled.disconnect(self._on_api_key_visibility_toggled)
            except Exception:
                pass
        if hasattr(self, "api_key_confirm_btn"):
            try:
                self.api_key_confirm_btn.clicked.disconnect(self._on_api_key_confirm_clicked)
            except Exception:
                pass
        if hasattr(self, "api_key_modify_btn"):
            try:
                self.api_key_modify_btn.clicked.disconnect(self._on_api_key_modify_clicked)
            except Exception:
                pass

        if hasattr(self, "about_movie") and self.about_movie is not None:
            self.about_movie.stop()
            self.about_movie.deleteLater()
            self.about_movie = None

    def closeEvent(self, event: QCloseEvent):
        """拦截关闭事件，应用关闭策略。"""
        """EN: Block close events and apply close policies."""
        if self._is_exiting:
            event.accept()
            return

        decision = self.close_policy.decide(self)
        if decision == "tray":
            event.ignore()
            self.hide()
            if self.tray_controller is not None:
                self.tray_controller.notify_minimized()
            return

        if decision == "quit":
            event.accept()
            self.request_quit()
            return

        event.ignore()

    def resizeEvent(self, event):
        """窗口尺寸变化时同步更新关于页 GIF 大小。"""
        """EN: Synchronously update about page GIF size when window size changes."""
        super().resizeEvent(event)
        self._update_about_gif_size()
    # ---------------------------------------------------------------
    # 音乐播放器辅助工具
    # EN: Music Player Accessibility
    # ---------------------------------------------------------------

    @staticmethod
    def _fmt_ms(ms: int) -> str:
        """将毫秒格式化为 m:ss 字符串。"""
        """EN: Format milliseconds as an m: ss string."""
        try:
            s = max(0, int(ms) // 1000)
        except (TypeError, ValueError):
            return "0:00"
        return f"{s // 60}:{s % 60:02d}"

    def _music_refresh_playlist(self):
        """刷新播放列表控件内容。"""
        """EN: Refreshes the contents of the playlist control."""
        if not hasattr(self, "music_list_widget"):
            return

        if hasattr(self, "_music_delete_delegate"):
            self._music_delete_delegate.set_delete_mode(self._music_delete_mode)

        # 先释放旧 item widget，避免残留控件对象堆积。
        # EN: Release the old item widget first to avoid the accumulation of residual control objects.
        for i in range(self.music_list_widget.count()):
            item = self.music_list_widget.item(i)
            row_widget = self.music_list_widget.itemWidget(item)
            if row_widget is not None:
                row_widget.deleteLater()

        self.music_list_widget.blockSignals(True)
        self.music_list_widget.clear()
        if self.music_player is not None:
            for i, track in enumerate(self.music_player.playlist):
                base_text = f"{i + 1}. {track.stem}"
                text = f"      {base_text}" if self._music_delete_mode else base_text
                item = QListWidgetItem(text)
                if self._music_delete_mode:
                    self._set_music_item_checked(item, False)
                else:
                    self._set_music_item_checked(item, False)
                self.music_list_widget.addItem(item)
            self._highlight_current_track()
        self.music_list_widget.blockSignals(False)
        self.music_list_widget.viewport().update()

    @staticmethod
    def _is_music_item_checked(item: QListWidgetItem) -> bool:
        state_data = item.data(Qt.ItemDataRole.CheckStateRole)
        if state_data is None:
            return False
        try:
            return int(state_data) == int(Qt.CheckState.Checked)
        except Exception:
            return state_data == Qt.CheckState.Checked

    @staticmethod
    def _set_music_item_checked(item: QListWidgetItem, checked: bool):
        item.setData(
            Qt.ItemDataRole.CheckStateRole,
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
        )

    def _highlight_current_track(self):
        """高亮当前播放歌曲行。"""
        """EN: Highlight the currently playing song row."""
        if self.music_player is None or not hasattr(self, "music_list_widget"):
            return
        idx = self.music_player.current_index
        for i in range(self.music_list_widget.count()):
            item = self.music_list_widget.item(i)
            font = item.font()
            font.setBold(i == idx)
            item.setFont(font)

    def _sync_music_ui_from_player(self):
        """将播放器当前完整状态同步到 UI。"""
        """EN: Synchronize the player's current full state to the UI."""
        if self.music_player is None:
            return
        from .music import PLAY_MODE_LIST
        # 歌曲名
        # EN: Song name
        self.music_track_label.setText(self.music_player.current_track_name)
        # 播放/暂停按钮
        # EN: Play/Pause Button
        self.music_play_btn.setText(
            tr(self.language, "menu.music.pause") if self.music_player.is_playing else tr(self.language, "menu.music.play")
        )
        # 模式按钮
        # EN: Mode Button
        for mode_key, btn in self._music_mode_btns.items():
            btn.setChecked(self.music_player.play_mode == mode_key)
        # 音量
        # EN: Volume
        vol_val = int(self.music_player.volume * 100)
        self.music_vol_slider.blockSignals(True)
        self.music_vol_slider.setValue(vol_val)
        self.music_vol_slider.blockSignals(False)
        self.music_vol_label.setText(f"{vol_val}%")
        # 时长
        # EN: Duration
        dur = self.music_player.duration
        self.music_progress_slider.setRange(0, max(0, dur))
        self.music_dur_label.setText(self._fmt_ms(dur))
        # 位置
        # EN: Locations
        pos = self.music_player.position
        if not self._music_seeking:
            self.music_progress_slider.blockSignals(True)
            self.music_progress_slider.setValue(pos)
            self.music_progress_slider.blockSignals(False)
        self.music_pos_label.setText(self._fmt_ms(pos))
        # 高亮
        # EN: Gao Liang
        self._highlight_current_track()

    # ---------------------------------------------------------------
    # 音乐播放器 — 控件回调
    # EN: Music Player — Control Callbacks
    # ---------------------------------------------------------------

    def _on_music_prev(self):
        if self.music_player is not None:
            self.music_player.prev()

    def _on_music_toggle_pause(self):
        if self.music_player is not None:
            self.music_player.toggle_pause()

    def _on_music_next(self):
        if self.music_player is not None:
            self.music_player.next()

    def _on_music_set_mode(self, mode: str):
        if self.music_player is not None:
            self.music_player.set_mode(mode)

    def _on_music_vol_changed(self, value: int):
        self.music_vol_label.setText(f"{value}%")
        if self.music_player is not None:
            self.music_player.set_volume(value / 100.0)

    def _on_music_seek(self):
        self._music_seeking = False
        if self.music_player is not None:
            self.music_player.seek(self.music_progress_slider.value())

    def _on_music_list_context_menu(self, pos):
        """播放列表右键菜单。提供歌曲重命名操作。"""
        """EN: Right-click the playlist menu. Provides a song rename action."""
        if self.music_player is None:
            return
        if self._music_delete_mode:
            return

        item = self.music_list_widget.itemAt(pos)
        if item is None:
            return

        row = self.music_list_widget.row(item)
        if row < 0 or row >= len(self.music_player.playlist):
            return

        menu = QMenu(self.music_list_widget)
        rename_action = menu.addAction(self._l("重命名歌曲", "Rename Track", "曲名変更", "곡 이름 변경", "Renommer le morceau"))
        action = menu.exec(self.music_list_widget.viewport().mapToGlobal(pos))
        menu.deleteLater()

        if action is rename_action:
            self._on_music_rename_track(row)

    def _on_music_rename_track(self, row: int):
        """重命名指定行歌曲，确认后同步修改 music 目录文件名。"""
        """EN: Rename the song in the specified row, and after confirmation, change the music directory file name synchronously."""
        if self.music_player is None:
            return
        if not (0 <= row < len(self.music_player.playlist)):
            return

        old_stem = self.music_player.playlist[row].stem
        new_name, ok = QInputDialog.getText(
            self,
            self._l("重命名歌曲", "Rename Track", "曲名変更", "곡 이름 변경", "Renommer le morceau"),
            self._l("请输入新名称（不含扩展名）：", "Enter new name (without extension):", "新しい名前（拡張子なし）を入力してください：", "새 이름(확장자 제외)을 입력하세요:", "Saisissez le nouveau nom (sans extension) :"),
            text=old_stem,
        )
        if not ok:
            return

        new_name = str(new_name).strip()
        if not new_name:
            QMessageBox.warning(
                self,
                self._l("提示", "Notice", "ヒント", "안내", "Info"),
                self._l("新名称不能为空", "New name cannot be empty", "新しい名前は空にできません", "새 이름은 비워둘 수 없습니다", "Le nouveau nom ne peut pas être vide"),
            )
            return

        if new_name == old_stem:
            return

        answer = QMessageBox.question(
            self,
            self._l("确认重命名", "Confirm Rename", "名前変更の確認", "이름 변경 확인", "Confirmer le renommage"),
            self._l(
                f"确认将\n\n{old_stem}\n\n重命名为\n\n{new_name}\n\n并同步修改 music 目录中的文件名吗？",
                f"Rename\n\n{old_stem}\n\nto\n\n{new_name}\n\nand update the file name in the music folder?",
                f"\n\n{old_stem}\n\nを\n\n{new_name}\n\nに変更し、music フォルダのファイル名も更新しますか？",
                f"\n\n{old_stem}\n\n을(를)\n\n{new_name}\n\n으로 변경하고 music 폴더의 파일명도 변경할까요?",
                f"Renommer\n\n{old_stem}\n\nen\n\n{new_name}\n\net mettre à jour le nom du fichier dans le dossier music ?",
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        ok_rename, msg = self.music_player.rename_track(row, new_name)
        if not ok_rename:
            QMessageBox.warning(self, self._l("重命名失败", "Rename Failed", "名前変更に失敗", "이름 변경 실패", "Échec du renommage"), msg)
            return

        self._music_refresh_playlist()
        self._sync_music_ui_from_player()

    def _on_music_add_local(self):
        """从本地选择音频文件导入到 music/ 目录并加入播放列表。"""
        """EN: Import from a locally selected audio file into the music/directory and join the playlist."""
        if self.music_player is None:
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            self._l("选择本地歌曲", "Select Local Tracks", "ローカル曲を選択", "로컬 곡 선택", "Sélectionner des morceaux locaux"),
            "",
            self._l(
                "音频文件 (*.ogg *.mp3 *.wav *.flac *.m4a);;所有文件 (*.*)",
                "Audio Files (*.ogg *.mp3 *.wav *.flac *.m4a);;All Files (*.*)",
                "音声ファイル (*.ogg *.mp3 *.wav *.flac *.m4a);;すべてのファイル (*.*)",
                "오디오 파일 (*.ogg *.mp3 *.wav *.flac *.m4a);;모든 파일 (*.*)",
                "Fichiers audio (*.ogg *.mp3 *.wav *.flac *.m4a);;Tous les fichiers (*.*)",
            ),
        )
        if not files:
            return

        success = 0
        failed_messages = []
        for path in files:
            ok, msg = self.music_player.add_track_from_file(path)
            if ok:
                success += 1
            else:
                failed_messages.append(f"{path}: {msg}")

        self._music_refresh_playlist()
        self._sync_music_ui_from_player()

        if failed_messages:
            QMessageBox.warning(
                self,
                self._l("导入结果", "Import Result", "インポート結果", "가져오기 결과", "Résultat de l'import"),
                self._l(
                    f"成功导入 {success} 首，失败 {len(failed_messages)} 首。\n\n",
                    f"Imported {success} track(s), failed {len(failed_messages)} track(s).\n\n",
                    f"{success} 曲をインポート、{len(failed_messages)} 曲失敗しました。\n\n",
                    f"{success}곡 가져오기 성공, {len(failed_messages)}곡 실패.\n\n",
                    f"{success} morceau(x) importé(s), {len(failed_messages)} échec(s).\n\n",
                )
                + "\n".join(failed_messages[:3]),
            )

    def _on_music_remove_selected(self):
        """点击删除按钮后进入勾选模式，支持多选后统一确认删除。"""
        """EN: Click the delete button to enter the check mode. Multiple selections are supported to confirm the deletion."""
        if self.music_player is None:
            return

        if self._music_delete_mode:
            return

        self._music_delete_mode = True
        self.music_list_widget.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.music_remove_btn.hide()
        self.music_confirm_delete_btn.show()
        self.music_cancel_delete_btn.show()
        self.music_delete_mode_hint.show()
        self._music_refresh_playlist()

    def _on_music_cancel_batch_remove(self):
        """取消批量删除模式。"""
        """EN: Cancels the bulk delete mode."""
        self._music_delete_mode = False
        self.music_list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.music_remove_btn.show()
        self.music_confirm_delete_btn.hide()
        self.music_cancel_delete_btn.hide()
        self.music_delete_mode_hint.hide()
        self._music_refresh_playlist()

    def _on_music_confirm_batch_remove(self):
        """批量确认删除已勾选歌曲。"""
        """EN: Bulk confirm deletion of checked songs."""
        if self.music_player is None:
            return

        checked_rows = []
        for i in range(self.music_list_widget.count()):
            item = self.music_list_widget.item(i)
            if self._is_music_item_checked(item):
                checked_rows.append(i)

        if not checked_rows:
            QMessageBox.information(
                self,
                self._l("提示", "Notice", "ヒント", "안내", "Info"),
                self._l("请先勾选要删除的歌曲", "Please select tracks to delete first", "先に削除する曲を選択してください", "먼저 삭제할 곡을 선택하세요", "Veuillez d'abord sélectionner les morceaux à supprimer"),
            )
            return

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setWindowTitle(self._l("批量删除歌曲", "Batch Delete", "一括削除", "일괄 삭제", "Suppression par lot"))
        dialog.setText(
            self._l(
                f"已勾选 {len(checked_rows)} 首歌曲，请选择删除方式：",
                f"{len(checked_rows)} track(s) selected. Choose delete mode:",
                f"{len(checked_rows)} 曲を選択しました。削除方法を選択してください：",
                f"{len(checked_rows)}곡 선택됨. 삭제 방식을 선택하세요:",
                f"{len(checked_rows)} morceau(x) sélectionné(s). Choisissez le mode de suppression :",
            )
        )
        dialog.setInformativeText(
            self._l(
                "可仅从列表移除，或同时删除本地文件。",
                "You can remove from list only, or remove and delete local files.",
                "リストからのみ削除、またはローカルファイルも削除できます。",
                "목록에서만 제거하거나 로컬 파일도 함께 삭제할 수 있습니다.",
                "Vous pouvez retirer de la liste uniquement, ou supprimer aussi les fichiers locaux.",
            )
        )

        remove_only_btn = dialog.addButton(self._l("仅从列表移除", "Remove from list only", "リストからのみ削除", "목록에서만 제거", "Retirer de la liste seulement"), QMessageBox.ButtonRole.ActionRole)
        remove_and_delete_btn = dialog.addButton(self._l("从列表移除并删除本地文件", "Remove and delete local files", "リストから削除しローカルファイルも削除", "목록에서 제거하고 로컬 파일도 삭제", "Retirer de la liste et supprimer les fichiers locaux"), QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(cancel_btn)
        dialog.exec()

        clicked = dialog.clickedButton()
        if clicked is cancel_btn:
            return

        delete_file = clicked is remove_and_delete_btn
        if clicked not in {remove_only_btn, remove_and_delete_btn}:
            return

        success_count = 0
        failed_msgs = []
        for row in sorted(checked_rows, reverse=True):
            ok, msg = self.music_player.remove_track(row, delete_file=delete_file)
            if ok:
                success_count += 1
            else:
                failed_msgs.append(msg)

        self._music_delete_mode = False
        self.music_list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.music_remove_btn.show()
        self.music_confirm_delete_btn.hide()
        self.music_cancel_delete_btn.hide()
        self.music_delete_mode_hint.hide()
        self._music_refresh_playlist()
        self._sync_music_ui_from_player()

        if failed_msgs:
            QMessageBox.warning(
                self,
                self._l("删除结果", "Delete Result", "削除結果", "삭제 결과", "Résultat de suppression"),
                self._l(
                    f"成功删除 {success_count} 首，失败 {len(failed_msgs)} 首。",
                    f"Deleted {success_count} track(s), failed {len(failed_msgs)} track(s).",
                    f"{success_count} 曲削除、{len(failed_msgs)} 曲失敗しました。",
                    f"{success_count}곡 삭제, {len(failed_msgs)}곡 실패.",
                    f"{success_count} morceau(x) supprimé(s), {len(failed_msgs)} échec(s).",
                ),
            )

    def _on_music_list_item_clicked(self, item: QListWidgetItem):
        if self.music_player is None:
            return
        if self._music_delete_mode:
            return
        row = self.music_list_widget.row(item)
        if row != self.music_player.current_index:
            self.music_player.play(row)
        else:
            self.music_player.toggle_pause()

    def _on_music_list_rows_moved(self, parent, src_start, src_end, dest_parent, dest_row):
        """拖拽排序完成后同步播放器内部列表顺序。"""
        """EN: Synchronizes the player internal list order after drag sorting completes."""
        if self.music_player is None:
            return
        if self._music_delete_mode:
            return
        to_index = dest_row if dest_row <= src_start else dest_row - 1
        self.music_player.move_track(src_start, to_index)
        # 刷新编号显示
        # EN: Refresh number display
        for i in range(self.music_list_widget.count()):
            item = self.music_list_widget.item(i)
            if self.music_player and i < len(self.music_player.playlist):
                item.setText(f"{i + 1}. {self.music_player.playlist[i].stem}")
        self._highlight_current_track()

    # ---------------------------------------------------------------
    # 音乐播放器 — Signal 回调（来自 MusicPlayer）
    # EN: Music Player — Signal Callback (from MusicPlayer)
    # ---------------------------------------------------------------

    def _on_music_track_changed(self, index: int):
        if self.music_player is None:
            return
        self.music_track_label.setText(self.music_player.current_track_name)
        self._highlight_current_track()

    def _on_music_state_changed(self, state: str):
        if not hasattr(self, "music_play_btn"):
            return
        self.music_play_btn.setText(tr(self.language, "menu.music.pause") if state == "playing" else tr(self.language, "menu.music.play"))

    def _on_music_playlist_reordered(self):
        self._music_refresh_playlist()

    def _on_music_volume_changed(self, volume: float):
        if not hasattr(self, "music_vol_slider"):
            return
        val = int(volume * 100)
        self.music_vol_slider.blockSignals(True)
        self.music_vol_slider.setValue(val)
        self.music_vol_slider.blockSignals(False)
        self.music_vol_label.setText(f"{val}%")

    def _on_music_mode_changed(self, mode: str):
        if not hasattr(self, "_music_mode_btns"):
            return
        for mode_key, btn in self._music_mode_btns.items():
            btn.setChecked(mode_key == mode)

    def _on_music_duration_changed(self, duration: int):
        if not hasattr(self, "music_progress_slider"):
            return
        self.music_progress_slider.setRange(0, max(0, duration))
        self.music_dur_label.setText(self._fmt_ms(duration))

    def _on_music_position_changed(self, position: int):
        if not hasattr(self, "music_progress_slider") or self._music_seeking:
            return
        self.music_progress_slider.blockSignals(True)
        self.music_progress_slider.setValue(position)
        self.music_progress_slider.blockSignals(False)
        self.music_pos_label.setText(self._fmt_ms(position))

    def _on_language_combo_changed(self):
        """关于页语言下拉回调。"""
        """EN: About page language drop-down callback."""
        if not hasattr(self, "about_language_combo"):
            return
        language = self.about_language_combo.currentData()
        if not isinstance(language, str):
            return

        if hasattr(self.pet, "on_set_language") and callable(self.pet.on_set_language):
            self.pet.on_set_language(language)
            return

        self.settings_store.set_language(language)
        self._on_pet_language_changed(language)

    def _l(self, zh: str, en: str, ja: str, ko: str, fr: str) -> str:
        """按当前语言返回内联文案。"""
        """EN: Returns inline copy in the current language."""
        mapping = {
            "zh-CN": zh,
            "en": en,
            "ja": ja,
            "ko": ko,
            "fr": fr,
        }
        return mapping.get(self.language, zh)
