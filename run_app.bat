@echo off
chcp 65001 >nul 2>&1

:: 定义颜色代码
set "COLOR_RESET=[0m"
set "COLOR_RED=[91m"
set "COLOR_GREEN=[92m"
set "COLOR_CYAN=[96m"
set "COLOR_MAGENTA=[95m"
set "COLOR_YELLOW=[93m"

title 🏃‍♂️ Running CC - 代码运行平台

:: 打印Logo动画
call :animate_logo

echo %COLOR_CYAN%═══ 系统初始化 ═══%COLOR_RESET%
echo.

:: 检查Python环境
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %COLOR_RED%错误: 未找到Python! 请安装Python 3.8+%COLOR_RESET%
    pause
    exit /b 1
)

:: 检查pip
python -m pip --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %COLOR_RED%错误: pip未正确安装!%COLOR_RESET%
    pause
    exit /b 1
)

:: 创建或激活虚拟环境
if not exist "venv\Scripts\activate.bat" (
    echo 正在创建虚拟环境...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo %COLOR_RED%虚拟环境创建失败!%COLOR_RESET%
        pause
        exit /b 1
    )
)

:: 激活虚拟环境
call "venv\Scripts\activate.bat"

:: 安装依赖
echo 正在安装依赖...
python -m pip install -r requirements.txt >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %COLOR_RED%依赖安装失败!%COLOR_RESET%
    pause
    exit /b 1
)

:: 启动应用
echo %COLOR_GREEN%✓ 系统启动成功%COLOR_RESET%
echo 启动应用...
echo 浏览器将自动打开 http://localhost:8506
echo 按 Ctrl+C 停止服务
echo.

:: 启动Streamlit应用
python -m streamlit run ui/app.py --server.port=8506 --server.headless=false

pause
exit /b 0

:animate_logo
echo %COLOR_MAGENTA%
echo.

:: 逐行显示Logo，模拟AI生成速度
echo    ██████╗ ██╗   ██╗███╗   ██╗███╗   ██╗██╗███╗   ██╗ ██████╗      ██████╗ ██████╗
call :delay 800

echo    ██╔══██╗██║   ██║████╗  ██║████╗  ██║██║████╗  ██║██╔════╝     ██╔════╝██╔════╝
call :delay 800

echo    ██████╔╝██║   ██║██╔██╗ ██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗    ██║     ██║
call :delay 800

echo    ██╔══██╗██║   ██║██║╚██╗██║██║╚██╗██║██║██║╚██╗██║██║   ██║    ██║     ██║
call :delay 800

echo    ██║  ██║╚██████╔╝██║ ╚████║██║ ╚████║██║██║ ╚████║╚██████╔╝    ╚██████╗╚██████╗
call :delay 800

echo    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝      ╚═════╝ ╚═════╝
call :delay 800

echo.
call :delay 400

echo         代码运行平台 - 快速编译执行可视化
call :delay 600

echo %COLOR_RESET%
exit /b 0

:delay
setlocal
set /a "times=%1/10"
for /l %%i in (1,1,%times%) do (
    >nul ping -n 1 127.0.0.1
)
endlocal

exit /b 0
