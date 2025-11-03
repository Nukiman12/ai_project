"""
SignVoiceAI - Приложение для распознавания жестов глухонемых в реальном времени.

Главный файл приложения. Запускает приложение, которое:
1. Захватывает видео с веб-камеры
2. Обнаруживает жесты через Mediapipe Hands
3. Классифицирует жесты через PyTorch модель
4. Отображает результат на экране
5. Озвучивает результат через pyttsx3
"""

import cv2
import sys
import os

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.camera import Camera
from utils.gestures import GestureDetector
from utils.speech import TextToSpeech
from model.gesture_model import GestureModelWrapper, GESTURE_CLASSES


class SignVoiceAI:
    """
    Главный класс приложения SignVoiceAI.
    """
    
    def __init__(self, model_path=None, camera_index=0):
        """
        Инициализация приложения.
        
        Args:
            model_path: Путь к обученной модели (опционально)
            camera_index: Индекс камеры
        """
        self.camera_index = camera_index
        self.camera = None
        self.gesture_detector = None
        self.gesture_model = None
        self.tts = None
        
        # Текущее состояние
        self.current_gesture = None
        self.last_gesture = None
        self.gesture_confidence = 0.0
        self.frame_count = 0
        self.gesture_stable_count = 0  # Счетчик стабильности жеста
        self.stability_threshold = 5  # Минимальное количество кадров для признания жеста стабильным
        
        # Инициализация компонентов
        print("=" * 60)
        print("SignVoiceAI - Распознавание жестов в реальном времени")
        print("=" * 60)
        
        # Инициализация модели
        print("\n[1/4] Инициализация модели распознавания жестов...")
        self.gesture_model = GestureModelWrapper(model_path=model_path, use_dummy=True)
        
        # Инициализация детектора жестов
        print("\n[2/4] Инициализация детектора жестов (Mediapipe)...")
        self.gesture_detector = GestureDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Инициализация синтезатора речи
        print("\n[3/4] Инициализация синтезатора речи...")
        self.tts = TextToSpeech(rate=150, volume=0.8)
        
        # Инициализация камеры
        print("\n[4/4] Инициализация камеры...")
        self.camera = Camera(camera_index=camera_index, width=640, height=480)
        
        print("\n" + "=" * 60)
        print("Инициализация завершена!")
        print("=" * 60)
    
    def run(self):
        """
        Запускает основной цикл приложения.
        """
        # Открываем камеру
        if not self.camera.open():
            print("Ошибка: не удалось открыть камеру")
            return
        
        print("\nПриложение запущено!")
        print("Инструкции:")
        print("- Покажите жесты перед камерой")
        print("- Нажмите 'q' для выхода")
        print("- Нажмите 'r' для повторного озвучивания текущего жеста")
        print("\nРаспознаваемые жесты:", ", ".join(GESTURE_CLASSES))
        print("-" * 60)
        
        try:
            while True:
                # Читаем кадр с камеры
                ret, frame = self.camera.read()
                
                if not ret:
                    print("Ошибка: не удалось прочитать кадр с камеры")
                    break
                
                # Обнаруживаем жесты
                landmarks, annotated_frame = self.gesture_detector.detect(frame)
                
                # Если рука обнаружена
                if landmarks is not None:
                    # Нормализуем координаты
                    try:
                        normalized_landmarks = self.gesture_model.normalize_landmarks(landmarks)
                    except Exception as e:
                        print(f"Ошибка нормализации: {e}")
                        normalized_landmarks = landmarks
                    
                    # Классифицируем жест
                    try:
                        gesture, confidence = self.gesture_model.predict(normalized_landmarks)
                        
                        # Обновляем текущий жест
                        if gesture == self.current_gesture:
                            self.gesture_stable_count += 1
                        else:
                            self.current_gesture = gesture
                            self.gesture_stable_count = 1
                        
                        self.gesture_confidence = confidence
                        
                        # Если жест стабилен и изменился, озвучиваем
                        if (self.gesture_stable_count >= self.stability_threshold and 
                            gesture != self.last_gesture):
                            self.last_gesture = gesture
                            self.tts.speak(gesture)
                            print(f"Жест распознан: {gesture} (уверенность: {confidence:.2f})")
                        
                        # Отображаем результат на кадре
                        self._draw_gesture_info(annotated_frame, gesture, confidence)
                        
                    except Exception as e:
                        print(f"Ошибка классификации жеста: {e}")
                        self._draw_no_hand(annotated_frame)
                else:
                    # Рука не обнаружена
                    self.current_gesture = None
                    self.gesture_stable_count = 0
                    self._draw_no_hand(annotated_frame)
                
                # Отображаем кадр
                cv2.imshow('SignVoiceAI - Распознавание жестов', annotated_frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nВыход из приложения...")
                    break
                elif key == ord('r'):
                    # Повторное озвучивание
                    if self.last_gesture:
                        print(f"Повторное озвучивание: {self.last_gesture}")
                        self.tts.speak(self.last_gesture, force=True)
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\n\nПрервано пользователем")
        except Exception as e:
            print(f"\nОшибка во время работы: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def _draw_gesture_info(self, frame, gesture, confidence):
        """
        Рисует информацию о распознанном жесте на кадре.
        
        Args:
            frame: Кадр для отрисовки
            gesture: Название жеста
            confidence: Уверенность распознавания
        """
        h, w = frame.shape[:2]
        
        # Фон для текста
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Текст с жестом
        gesture_text = f"Жест: {gesture}"
        cv2.putText(frame, gesture_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Уверенность
        confidence_text = f"Уверенность: {confidence:.2f}"
        cv2.putText(frame, confidence_text, (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Статус стабильности
        if self.gesture_stable_count >= self.stability_threshold:
            status_text = "Стабильно"
            color = (0, 255, 0)
        else:
            status_text = "Определение..."
            color = (0, 165, 255)
        
        cv2.putText(frame, status_text, (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_no_hand(self, frame):
        """
        Рисует сообщение об отсутствии руки на кадре.
        
        Args:
            frame: Кадр для отрисовки
        """
        h, w = frame.shape[:2]
        
        # Фон для текста
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Текст
        text = "Покажите жесты перед камерой"
        cv2.putText(frame, text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    def cleanup(self):
        """
        Освобождает все ресурсы приложения.
        """
        print("\nОчистка ресурсов...")
        
        if self.camera:
            self.camera.release()
        
        if self.gesture_detector:
            self.gesture_detector.close()
        
        if self.tts:
            self.tts.stop()
        
        cv2.destroyAllWindows()
        
        print("Очистка завершена. До свидания!")


def main():
    """
    Точка входа в приложение.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SignVoiceAI - Распознавание жестов глухонемых в реальном времени'
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
    
    # Создаем и запускаем приложение
    app = SignVoiceAI(model_path=args.model, camera_index=args.camera)
    app.run()


if __name__ == "__main__":
    main()

