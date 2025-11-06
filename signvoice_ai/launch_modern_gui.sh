#!/bin/bash
# ====================================================================
# SignVoiceAI Modern GUI - Скрипт запуска для Linux/macOS
# ====================================================================

echo ""
echo "================================================================"
echo "  SignVoiceAI - Modern GUI Launcher"
echo "================================================================"
echo ""

# Проверка наличия Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python не найден! Установите Python 3.8 или выше."
    echo "https://www.python.org/downloads/"
    exit 1
fi

echo "[OK] Python найден"
echo ""

# Проверка наличия зависимостей
echo "Проверка зависимостей..."
python3 -c "import customtkinter" &> /dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "[WARNING] CustomTkinter не установлен!"
    echo "Устанавливаем зависимости..."
    echo ""
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Не удалось установить зависимости!"
        exit 1
    fi
else
    echo "[OK] Зависимости установлены"
fi

echo ""
echo "================================================================"
echo "  Запуск SignVoiceAI Modern GUI..."
echo "================================================================"
echo ""

# Запуск приложения
python3 main_modern_gui.py

echo ""
echo "================================================================"
echo "  Приложение закрыто"
echo "================================================================"
echo ""

