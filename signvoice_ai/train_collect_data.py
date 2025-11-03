"""
Скрипт для сбора данных жестов с камеры.

Использование:
    python train_collect_data.py --gesture Hello --output data/Hello

Скрипт записывает координаты суставов руки в файлы CSV.
Для каждого жеста нужно собрать достаточно примеров (рекомендуется 50-100+).
"""

import cv2
import numpy as np
import argparse
import os
import csv
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.camera import Camera
from utils.gestures import GestureDetector


class DataCollector:
    """
    Класс для сбора данных жестов с камеры.
    """
    
    def __init__(self, gesture_name, output_dir, camera_index=0):
        """
        Инициализация сборщика данных.
        
        Args:
            gesture_name: Название жеста (Hello, Thanks, Yes, No, Love)
            output_dir: Папка для сохранения данных
            camera_index: Индекс камеры
        """
        self.gesture_name = gesture_name
        self.output_dir = output_dir
        self.camera_index = camera_index
        
        # Создаем папку для данных если её нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Счетчики
        self.samples_collected = 0
        self.samples_to_collect = 50  # По умолчанию собираем 50 образцов
        
        # Файл для сохранения данных
        csv_file = os.path.join(output_dir, f"{gesture_name}_samples.csv")
        self.csv_file = csv_file
        
        # Проверяем, существует ли файл
        file_exists = os.path.exists(csv_file)
        
        # Открываем CSV файл для записи
        self.csv_file_handle = open(csv_file, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file_handle)
        
        # Если файл новый, записываем заголовок
        if not file_exists:
            # Заголовок: gesture_name, x1, y1, z1, x2, y2, z2, ..., x21, y21, z21 (63 значения)
            header = ['gesture_name'] + [f'{coord}_{i}' for i in range(21) for coord in ['x', 'y', 'z']]
            self.csv_writer.writerow(header)
            print(f"Создан новый файл: {csv_file}")
        else:
            # Подсчитываем существующие записи
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                existing_samples = sum(1 for row in reader) - 1  # -1 для заголовка
                self.samples_collected = existing_samples
                print(f"Файл существует, найдено {existing_samples} образцов")
        
        print(f"Данные будут сохраняться в: {csv_file}")
    
    def collect(self):
        """
        Основной цикл сбора данных.
        """
        print("\n" + "=" * 60)
        print(f"Сбор данных для жеста: {self.gesture_name}")
        print("=" * 60)
        print("\nИнструкции:")
        print("- Показывайте жест перед камерой")
        print("- Нажмите 'SPACE' для сохранения образца")
        print("- Нажмите 'q' для завершения сбора")
        print(f"- Цель: собрать {self.samples_to_collect} образцов")
        print(f"- Собрано: {self.samples_collected}")
        print("-" * 60)
        
        camera = Camera(camera_index=self.camera_index)
        gesture_detector = GestureDetector()
        
        if not camera.open():
            print("Ошибка: не удалось открыть камеру")
            return
        
        collecting = False
        last_sample_time = 0
        
        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    break
                
                # Обнаруживаем жесты
                landmarks, annotated_frame = gesture_detector.detect(frame)
                
                # Отображаем информацию
                h, w = frame.shape[:2]
                
                # Фон для текста
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                # Текст с инструкциями
                text1 = f"Жест: {self.gesture_name}"
                cv2.putText(annotated_frame, text1, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                if landmarks is not None:
                    status_text = "Рука обнаружена - Нажмите SPACE для сохранения"
                    color = (0, 255, 0)
                else:
                    status_text = "Покажите жесты перед камерой"
                    color = (0, 165, 255)
                
                cv2.putText(annotated_frame, status_text, (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                samples_text = f"Собрано: {self.samples_collected}/{self.samples_to_collect}"
                cv2.putText(annotated_frame, samples_text, (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if collecting:
                    saving_text = "Сохранение..."
                    cv2.putText(annotated_frame, saving_text, (20, 140),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Отображаем кадр
                cv2.imshow('Сбор данных - SignVoiceAI', annotated_frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' ') and landmarks is not None:
                    # Сохраняем образец
                    import time
                    current_time = time.time()
                    
                    # Защита от слишком частого сохранения (минимум 0.3 секунды между сохранениями)
                    if current_time - last_sample_time > 0.3:
                        self._save_sample(landmarks)
                        last_sample_time = current_time
                        collecting = True
                    else:
                        collecting = False
                else:
                    collecting = False
                
        except KeyboardInterrupt:
            print("\n\nПрервано пользователем")
        finally:
            camera.release()
            gesture_detector.close()
            self.csv_file_handle.close()
            cv2.destroyAllWindows()
            
            print(f"\nСбор данных завершен!")
            print(f"Всего собрано образцов для жеста '{self.gesture_name}': {self.samples_collected}")
            print(f"Данные сохранены в: {self.csv_file}")
    
    def _save_sample(self, landmarks):
        """
        Сохраняет один образец жеста.
        
        Args:
            landmarks: Массив координат суставов (63 значения)
        """
        # Нормализуем координаты
        landmarks_array = np.array(landmarks).reshape(21, 3)
        wrist = landmarks_array[0]
        normalized = landmarks_array - wrist
        normalized_flat = normalized.flatten()
        
        # Записываем в CSV: название жеста + 63 координаты
        row = [self.gesture_name] + normalized_flat.tolist()
        self.csv_writer.writerow(row)
        self.csv_file_handle.flush()  # Сразу записываем на диск
        
        self.samples_collected += 1
        print(f"Образец {self.samples_collected} сохранен")


def main():
    parser = argparse.ArgumentParser(
        description='Сбор данных жестов с камеры для обучения модели'
    )
    parser.add_argument(
        '--gesture',
        type=str,
        required=True,
        choices=['Hello', 'Thanks', 'Yes', 'No', 'Love'],
        help='Название жеста для сбора данных'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Папка для сохранения данных (по умолчанию: data)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Индекс камеры (по умолчанию 0)'
    )
    
    args = parser.parse_args()
    
    collector = DataCollector(
        gesture_name=args.gesture,
        output_dir=args.output,
        camera_index=args.camera
    )
    
    collector.collect()


if __name__ == "__main__":
    main()

