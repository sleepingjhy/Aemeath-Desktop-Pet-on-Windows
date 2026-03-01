# Ameath Desktop Pet Instructions

## 目录与文件功能索引

### 根目录
- `main.py`：应用入口，负责初始化设置存储、音乐播放器、实例管理器、桌宠窗口、前端窗口与系统托盘。
- `requirements.txt`：项目依赖列表（PySide6、PySide6-Addons、PySide6-Essentials、PyInstaller）。
- `README.md`：用户使用说明、交互说明、打包说明与项目结构说明。
- `resources.qrc`：Qt 资源清单文件（gifs 前缀资源）。

### 音乐目录
- `music/`：音乐资源目录（`.ogg`），供桌宠右键菜单与应用前端播放器使用。

### `.github`
- `.github/copilot-instructions.md`：Copilot 工作指引与项目文件功能快速索引。

### `pet` 包
- `pet/__init__.py`：包级说明文件。
- `pet/config.py`：全局常量与资源路径配置（动画路径、显示模式、实例数量、透明度范围等）。
- `pet/animation.py`：GIF 播放、缩放与镜像绘制能力。
- `pet/state_machine.py`：桌宠核心状态机（拖拽、跟随、休息、移动）。
- `pet/movement.py`：桌宠移动控制（自动移动、跟随、边界检测、触边停顿）。
- `pet/input.py`：鼠标输入处理（左键拖拽、右键菜单、释放状态恢复）。
- `pet/idle.py`：随机休息调度与休息状态切换。
- `pet/menu.py`：右键菜单构建（移动/跟随/缩放/透明度/显示优先级/多开/关闭菜单）。
- `pet/music/music_player.py`：全局音乐播放器（QMediaPlayer + QAudioOutput），负责播放列表、切歌、播放模式、音量与进度信号。
- `pet/chat/api.py`：聊天 Agent API 适配层（DeepSeek + 本地检索 + 可选联网检索）。
- `pet/chat/session.py`：聊天会话状态管理（多会话、消息缓存、异步回复队列、删除会话与缓存清理、退出释放）。
- `pet/chat/widgets.py`：聊天 UI 组件（头像、气泡、输入框、图片上传按钮、消息流、表情面板、最近使用、图文混排发送）。
- `pet/chat/window.py`：独立聊天窗口（会话列表、新建/删除会话、右键会话项删除）。
- `pet/autostart.py`：Windows 开机自启注册表读写。
- `pet/window.py`：单个桌宠窗口主类，负责动画调度、事件处理、菜单动作响应、右键实例高亮描边与本地应用设置。
- `pet/instance_manager.py`：多开实例管理与全局同步中心，负责实例增减、批量设置同步、显示策略轮询。
- `pet/app_window.py`：前端应用窗口（设置页、聊天页、音乐页、关于页），支持与右键设置及音乐状态实时双向同步。
- `pet/tray_controller.py`：系统托盘图标与托盘菜单控制（打开/退出/提示）。
- `pet/settings_store.py`：本地 JSON 设置持久化（关闭行为、显示优先级、多开数量、透明度）。
- `pet/close_policy.py`：点击关闭按钮时的策略弹窗与“记住选择”逻辑。
- `pet/resources_rc.py`：由 `resources.qrc` 生成的 Qt 资源编译文件。

### `exe`
- `exe/build_exe.ps1`：EXE 打包脚本（PyInstaller）。
- `exe/README.md`：EXE 目录与打包产物说明。

### 资源目录
- `gifs/`：桌宠动画与图标资源（`move.gif`、`drag.gif`、`ameath.gif`、`idle*.gif`、`ameath.ico`）。
- `gifs/check_white.svg`：音乐页批量删除勾选框的白色勾图标资源。
- `music/`：音乐资源（`.ogg`），用于播放器列表与播放控制。

## 常见定位建议
- 修改桌宠行为优先看：`pet/window.py`、`pet/movement.py`、`pet/state_machine.py`。
- 修改右键菜单优先看：`pet/menu.py`、`pet/window.py`。
- 修改音乐播放优先看：`pet/music/music_player.py`、`pet/menu.py`、`pet/app_window.py`、`main.py`。
- 修改聊天界面与 Agent API 优先看：`pet/chat/widgets.py`、`pet/chat/session.py`、`pet/chat/api.py`、`pet/chat/window.py`、`main.py`、`pet/window.py`。
- 修改会话删除与会话列表交互优先看：`pet/chat/window.py`（按钮/右键菜单）与 `pet/chat/session.py`（`delete_conversation`）。
- 修改聊天图文混排发送（文字+表情+本地图）优先看：`pet/chat/widgets.py`（`_build_compose_payload`、`_on_send_clicked`、`_on_add_image_clicked`）与 `pet/chat/session.py`（`send_composed`）。
- 修改聊天时间分割样式（如 `02-22 20:53`）优先看：`pet/chat/widgets.py`（`QLabel#ChatDividerText` 样式）。
- 修改表情面板样式与“最近使用/清空”优先看：`pet/chat/widgets.py`（`EmojiPickerPopup`）。
- 修改音乐批量删除与勾选交互优先看：`pet/app_window.py`（删除模式提示、勾选框样式、整行点击勾选）。
- 排查内存泄漏优先看：`pet/window.py`、`pet/music/music_player.py`、`pet/chat/window.py`、`pet/chat/widgets.py`、`pet/instance_manager.py`、`pet/app_window.py`（信号断连、定时器停止、deleteLater 链路）。
- 聊天线程与退出释放优先看：`pet/chat/session.py`（`dispose`、回复线程）与 `main.py`（`request_quit` 释放链路）。
- 修改设置页与联动优先看：`pet/app_window.py`、`pet/instance_manager.py`、`pet/settings_store.py`。
- 修改启动流程与多开初始化优先看：`main.py`、`pet/instance_manager.py`。
