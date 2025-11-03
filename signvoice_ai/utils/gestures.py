"""
Модуль для обработки жестов через Mediapipe Hands.
"""

import cv2
import mediapipe as mp
import numpy as np


class GestureDetector:
    """
    Класс для обнаружения и извлечения координат руки через Mediapipe.
    """
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Инициализация детектора жестов.
        
        Args:
            min_detection_confidence: Минимальная уверенность для обнаружения руки
            min_tracking_confidence: Минимальная уверенность для отслеживания руки
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Отслеживаем одну руку
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect(self, frame):
        """
        Обнаруживает руку на кадре и извлекает координаты суставов.
        
        Args:
            frame: Кадр изображения (BGR формат для OpenCV)
            
        Returns:
            Кортеж (landmarks, annotated_frame), где:
            - landmarks: Массив координат суставов [63] или None если рука не найдена
            - annotated_frame: Кадр с нарисованной рукой
        """
        # Конвертируем BGR в RGB (Mediapipe работает с RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Обрабатываем кадр
        results = self.hands.process(rgb_frame)
        
        annotated_frame = frame.copy()
        landmarks = None
        
        if results.multi_hand_landmarks:
            # Берем первую обнаруженную руку
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Рисуем скелет руки на кадре
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            # Извлекаем координаты всех 21 точки
            landmarks = []
            h, w = frame.shape[:2]
            
            for landmark in hand_landmarks.landmark:
                # Mediapipe возвращает нормализованные координаты (0-1)
                # Сохраняем их в массиве
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)
            
            # Преобразуем в numpy массив
            landmarks = np.array(landmarks)
            
            # Должно быть 21 точка × 3 координаты = 63 значения
            assert len(landmarks) == 63, f"Ожидается 63 значения, получено {len(landmarks)}"
        
        return landmarks, annotated_frame
    
    def close(self):
        """
        Закрывает детектор жестов (освобождает ресурсы).
        """
        self.hands.close()
    
    def __enter__(self):
        """Контекстный менеджер"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер"""
        self.close()

