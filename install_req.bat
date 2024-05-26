@echo off
:: 如果需要找到requirements.txt，首先定位当前脚本目录
set SCRIPT_DIR=%~dp0
:: 然后，转到嵌入式Python所在的目录
cd /d "%SCRIPT_DIR%../../../python_embeded"

:: 安装requirements.txt中的依赖项
python.exe -m pip install -r "%SCRIPT_DIR%requirements.txt"

:: 暂停以查看输出
pause
