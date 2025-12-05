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

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python 未安装或未在PATH中
    pause
    exit /b
)

:: 检查并激活虚拟环境
if exist ".venv\Scripts\python.exe" (
    echo [*] 正在激活.venv虚拟环境...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\python.exe" (
    echo [*] 正在激活venv虚拟环境...
    call venv\Scripts\activate.bat
) else (
    echo [*] 未发现虚拟环境，使用系统Python...
)

:: 检查依赖安装文件
if exist "requirements.txt" (
    echo [*] 正在安装依赖...
    pip install -r requirements.txt >nul 2>&1
    if errorlevel 1 (
        echo [!] 依赖安装失败
        pause
        exit /b
    )
) else (
    echo [*] 未找到requirements.txt，跳过依赖安装...
)

:: 启动应用
echo [✓] 启动成功
echo    运行命令: python main.py
echo    停止程序: Ctrl+C
echo.
echo 当前工作目录: %cd%
echo 检查main.py是否存在: if exist "%cd%\main.py" (echo 文件存在) else (echo 文件不存在)

echo %COLOR_YELLOW%═══ 程序运行中 ═══%COLOR_RESET%
echo.
python "%cd%\main.py"

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

echo         [3m[2mCC RUNNING - 快速实现可视化[0m
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