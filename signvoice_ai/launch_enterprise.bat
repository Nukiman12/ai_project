@echo off
echo ==========================================
echo  SignVoiceAI Enterprise Edition
echo ==========================================
echo.

REM Активация виртуального окружения если существует
if exist "venv311\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv311\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Starting SignVoiceAI Enterprise...
echo.

python main_enterprise_gui.py

if errorlevel 1 (
    echo.
    echo Error occurred during execution
    pause
)




