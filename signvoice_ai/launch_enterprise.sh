#!/bin/bash

echo "=========================================="
echo " SignVoiceAI Enterprise Edition"
echo "=========================================="
echo ""

# Активация виртуального окружения если существует
if [ -d "venv311" ]; then
    echo "Activating virtual environment..."
    source venv311/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Starting SignVoiceAI Enterprise..."
echo ""

python3 main_enterprise_gui.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error occurred during execution"
    read -p "Press Enter to continue..."
fi




