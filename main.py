"""
Точка входа в приложение SignVoiceAI.
Запускает приложение из папки signvoice_ai.
"""

import sys
import os
import runpy
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SignVoiceAI - Распознавание жестов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  python main.py                    # Классический интерфейс (OpenCV)
  python main.py --gui              # Новый современный GUI
  python main.py --gui --model models/gesture_model.pth
        '''
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Запустить с современным графическим интерфейсом (рекомендуется)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Путь к файлу обученной модели (.pth или .pt)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Индекс камеры (по умолчанию 0)'
    )
    
    args = parser.parse_args()
    
    # Определяем какой интерфейс запускать
    if args.gui:
        target_file = 'main_gui.py'
        print("Запуск с современным GUI...")
    else:
        target_file = 'main.py'
        print("Запуск с классическим интерфейсом...")
    
    signvoice_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signvoice_ai', target_file)
    
    if os.path.exists(signvoice_path):
        sys.path.insert(0, os.path.dirname(signvoice_path))
        
        # Передаем аргументы в запускаемый скрипт
        sys.argv = [signvoice_path]
        if args.model:
            sys.argv.extend(['--model', args.model])
        if args.camera != 0:
            sys.argv.extend(['--camera', str(args.camera)])
        
        runpy.run_path(signvoice_path, run_name='__main__')
    else:
        print(f"Ошибка: не найден файл signvoice_ai/{target_file}")
        print("Убедитесь, что вы запускаете приложение из корневой папки проекта")
        sys.exit(1)

