Param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# 统一终端编码，避免中文提示在不同 PowerShell 终端中出现乱码。
try {
    chcp 65001 | Out-Null
} catch {
}
try {
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [Console]::InputEncoding = $utf8NoBom
    [Console]::OutputEncoding = $utf8NoBom
    $OutputEncoding = $utf8NoBom
} catch {
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

if ($Clean) {
    Remove-Item -Recurse -Force "$root\exe\build" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$root\exe\dist" -ErrorAction SilentlyContinue
}

& "$root\.venv\Scripts\pyside6-rcc.exe" "$root\resources.qrc" -o "$root\pet\resources_rc.py"

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name AmeathDesktopPet `
    --icon "$root\gifs\ameath.ico" `
    --distpath "$root\exe\dist" `
    --workpath "$root\exe\build" `
    --specpath "$root\exe" `
    "$root\main.py"

$distRoot = Join-Path $root "exe\dist\AmeathDesktopPet"

# 同步聊天/表情等文件资源目录。
$distGifs = Join-Path $distRoot "gifs"
if (Test-Path $distGifs) {
    Remove-Item -Recurse -Force $distGifs -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $distGifs -Force | Out-Null

$srcGifs = Join-Path $root "gifs"
if (Test-Path $srcGifs) {
    Copy-Item -Path (Join-Path $srcGifs "*") -Destination $distGifs -Recurse -Force
}

# 将 music 目录单独放到产物根目录，便于普通用户自行增删音乐文件。
$distMusic = Join-Path $distRoot "music"
if (Test-Path $distMusic) {
    Remove-Item -Recurse -Force $distMusic -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $distMusic -Force | Out-Null

$srcMusic = Join-Path $root "music"
if (Test-Path $srcMusic) {
    Get-ChildItem -Path $srcMusic -Force |
        Where-Object { $_.Name -notin @('.gitignore', '.gitkeep') } |
        ForEach-Object {
            Copy-Item -Path $_.FullName -Destination (Join-Path $distMusic $_.Name) -Recurse -Force
        }
}

# 同步普通用户说明文档到产物根目录（排除开发用 requirements.txt）。
Get-ChildItem -Path $root -File -Filter "*.txt" |
    Where-Object { $_.Name -ne "requirements.txt" } |
    ForEach-Object {
        Copy-Item -Path $_.FullName -Destination (Join-Path $distRoot $_.Name) -Force
    }

Write-Host "打包完成，输出目录: $distRoot" -ForegroundColor Green
