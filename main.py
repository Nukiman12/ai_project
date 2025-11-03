"""
Точка входа в приложение SignVoiceAI.
Запускает приложение из папки signvoice_ai.
"""

import sys
import os
import runpy

# Запускаем главный модуль из папки signvoice_ai
if __name__ == "__main__":
    signvoice_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signvoice_ai', 'main.py')
    if os.path.exists(signvoice_path):
        sys.path.insert(0, os.path.dirname(signvoice_path))
        runpy.run_path(signvoice_path)
    else:
        print("Ошибка: не найден файл signvoice_ai/main.py")
        print("Убедитесь, что вы запускаете приложение из корневой папки проекта")
        sys.exit(1)

