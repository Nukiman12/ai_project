@echo off
REM ====================================================================
REM SignVoiceAI Modern GUI - Скрипт запуска для Windows
REM ====================================================================

echo.
echo ================================================================
echo   SignVoiceAI - Modern GUI Launcher
echo ================================================================
echo.

REM Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден! Установите Python 3.8 или выше.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python найден
echo.

REM Проверка наличия зависимостей
echo Проверка зависимостей...
python -c "import customtkinter" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] CustomTkinter не установлен!
    echo Устанавливаем зависимости...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Не удалось установить зависимости!
        pause
        exit /b 1
    )
) else (
    echo [OK] Зависимости установлены
)

echo.
echo ================================================================
echo   Запуск SignVoiceAI Modern GUI...
echo ================================================================
echo.

REM Запуск приложения
python main_modern_gui.py

echo.
echo ================================================================
echo   Приложение закрыто
echo ================================================================
echo.
pause

