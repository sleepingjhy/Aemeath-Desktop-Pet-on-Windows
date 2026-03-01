"""这是应用入口模块。负责初始化桌宠窗口、主界面与系统托盘。"""
"""EN: Application entry point. Initializes pet windows, app window, and system tray."""

import os
import sys

_existing_qt_logging_rules = os.environ.get("QT_LOGGING_RULES", "").strip()
_ffmpeg_log_rule = "qt.multimedia.ffmpeg=false"
if _ffmpeg_log_rule not in _existing_qt_logging_rules:
    if _existing_qt_logging_rules:
        os.environ["QT_LOGGING_RULES"] = f"{_existing_qt_logging_rules};{_ffmpeg_log_rule}"
    else:
        os.environ["QT_LOGGING_RULES"] = _ffmpeg_log_rule

from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from pet.app_window import AppWindow
from pet.chat import ChatSession, ChatWindow
from pet.close_policy import ClosePolicyManager
from pet.config import APP_ICON_PATH
from pet.instance_manager import PetInstanceManager
from pet.music import MusicPlayer
from pet.settings_store import SettingsStore
from pet.tray_controller import TrayController
from pet.window import DesktopPet


def main():
    """启动应用主流程。统一拉起桌宠、主界面与托盘控制。"""
    """EN: Start the main app flow and bootstrap pet, app window, and tray controller."""
    # 先创建 Qt 应用对象。所有 QWidget 生命周期都依赖它。
    # EN: Create the Qt application first. All QWidget lifecycles depend on it.
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(QIcon(str(APP_ICON_PATH)))

    app_font: QFont = app.font()
    if app_font.pointSize() <= 0:
        app_font.setPointSize(10)
        app.setFont(app_font)

    # 初始化设置存储与关闭策略。
    # EN: Initialize settings store and close-policy manager.
    settings_store = SettingsStore()
    stored_api_key = settings_store.get_api_key()
    if stored_api_key:
        os.environ["DEEPSEEK_API_KEY"] = stored_api_key
    close_policy = ClosePolicyManager(settings_store)

    # 初始化音乐播放器单例。
    # EN: Initialize the global music player singleton.
    music_player = MusicPlayer()

    # 准备退出防重入标志。
    # EN: Re-entrancy guard for unified quit flow.
    is_quitting = False

    # 先声明变量，便于回调里统一访问。
    # EN: Predeclare references for use in nested callbacks.
    manager = None
    app_window = None
    tray_controller = None
    chat_session = ChatSession()
    chat_window = None

    def open_chat_window():
        """打开独立聊天窗口（全局单例）。"""
        nonlocal chat_window

        if chat_window is None:
            chat_window = ChatWindow(session=chat_session)
            chat_window.destroyed.connect(lambda *_: _reset_chat_window_ref())
        chat_window.show_window()

    def _reset_chat_window_ref():
        nonlocal chat_window
        chat_window = None

    def create_pet():
        """创建并注册单个桌宠实例。"""
        """EN: Create and register one desktop pet instance."""
        nonlocal manager, tray_controller

        pet = DesktopPet(
            on_open_main=open_main_window,
            on_open_chat=open_chat_window,
            on_request_quit=request_quit,
            close_policy=close_policy,
            instance_manager=manager,
            music_player=music_player,
        )
        manager.register_pet(pet)

        if tray_controller is not None:
            pet.set_tray_controller(tray_controller)

        return pet

    def request_quit():
        """执行统一退出流程。包含防重入、资源清理和应用退出。"""
        """EN: Execute unified quit flow with guard, cleanup, and app exit."""
        nonlocal is_quitting, manager, app_window, tray_controller, chat_window

        if is_quitting:
            return
        is_quitting = True

        if manager is not None:
            if hasattr(manager, "shutdown") and callable(manager.shutdown):
                manager.shutdown()
            else:
                for pet in manager.pets:
                    if hasattr(pet, "prepare_for_exit") and callable(pet.prepare_for_exit):
                        pet.prepare_for_exit()
                    if hasattr(pet, "close") and callable(pet.close):
                        pet.close()

        if app_window is not None:
            if hasattr(app_window, "prepare_for_exit") and callable(app_window.prepare_for_exit):
                app_window.prepare_for_exit()
            if hasattr(app_window, "close") and callable(app_window.close):
                app_window.close()
            if hasattr(app_window, "deleteLater") and callable(app_window.deleteLater):
                app_window.deleteLater()

        if chat_window is not None:
            if hasattr(chat_window, "prepare_for_exit") and callable(chat_window.prepare_for_exit):
                chat_window.prepare_for_exit()
            if hasattr(chat_window, "close") and callable(chat_window.close):
                chat_window.close()
            if hasattr(chat_window, "deleteLater") and callable(chat_window.deleteLater):
                chat_window.deleteLater()
            chat_window = None

        if chat_session is not None and hasattr(chat_session, "dispose") and callable(chat_session.dispose):
            chat_session.dispose()

        if tray_controller is not None:
            if hasattr(tray_controller, "dispose") and callable(tray_controller.dispose):
                tray_controller.dispose()
            else:
                tray_controller.hide()

        if music_player is not None and hasattr(music_player, "dispose") and callable(music_player.dispose):
            music_player.dispose()

        app.quit()

    def open_main_window():
        """显示并激活前端主界面。"""
        """EN: Show and activate the frontend app window."""
        if app_window is None:
            return
        app_window.show_window()

    # 初始化多开管理器，并创建首个桌宠实例。
    # EN: Initialize multi-instance manager and create the first pet instance.
    manager = PetInstanceManager(settings_store=settings_store, request_quit=request_quit, music_player=music_player)
    manager.set_spawn_callback(create_pet)
    create_pet()

    # 初始化主界面实例。
    # EN: Initialize the frontend app window instance.
    app_window = AppWindow(
        pet=manager,
        settings_store=settings_store,
        close_policy=close_policy,
        request_quit=request_quit,
        tray_controller=None,
        music_player=music_player,
        chat_session=chat_session,
        on_open_chat_window=open_chat_window,
    )

    # 初始化系统托盘并显示。
    # EN: Initialize and show the system tray controller.
    tray_controller = TrayController(
        icon_path=APP_ICON_PATH,
        on_open=open_main_window,
        on_exit=request_quit,
        settings_store=settings_store,
        music_player=music_player,
    )
    tray_controller.show()

    # 注入托盘控制器。优先走方法注入，不存在时回退为属性注入。
    # EN: Inject tray controller. Prefer method injection, fallback to direct attribute assignment.
    if not hasattr(app_window, "set_tray_controller") or not callable(getattr(app_window, "set_tray_controller")):
        setattr(
            app_window,
            "set_tray_controller",
            lambda controller: setattr(app_window, "tray_controller", controller),
        )

    for pet in manager.pets:
        if not hasattr(pet, "set_tray_controller") or not callable(getattr(pet, "set_tray_controller")):
            setattr(pet, "set_tray_controller", lambda controller, p=pet: setattr(p, "tray_controller", controller))
        pet.set_tray_controller(tray_controller)
    app_window.set_tray_controller(tray_controller)

    # 按配置补齐到目标实例数量。
    # EN: Expand/shrink instances to match configured target count.
    manager.on_set_instance_count(settings_store.get_instance_count())

    # 启动时同时展示桌宠窗口与主界面。
    # EN: Show both pet window(s) and app window on startup.
    app_window.show_window()

    # 最后进入事件循环。用户关闭窗口后才会返回。
    # EN: Enter the Qt event loop; returns only after app exits.
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
