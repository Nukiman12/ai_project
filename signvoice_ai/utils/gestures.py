"""
Модуль для обработки жестов через Mediapipe Hands.

Поддерживает:
- Обнаружение одной или двух рук
- Динамические жесты (движения)
- Статические жесты (позы)
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque


class GestureDetector:
    """
    Класс для обнаружения и извлечения координат рук через Mediapipe.
    Поддерживает одну или две руки одновременно.
    """
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, 
                 max_num_hands=2, detect_both_hands=True):
        """
        Инициализация детектора жестов.
        
        Args:
            min_detection_confidence: Минимальная уверенность для обнаружения руки
            min_tracking_confidence: Минимальная уверенность для отслеживания руки
            max_num_hands: Максимальное количество рук (1 или 2)
            detect_both_hands: Обнаруживать обе руки
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.detect_both_hands = detect_both_hands
        self.max_num_hands = max_num_hands if detect_both_hands else 1
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect(self, frame):
        """
        Обнаруживает руки на кадре и извлекает координаты суставов.
        
        Args:
            frame: Кадр изображения (BGR формат для OpenCV)
            
        Returns:
            Кортеж (hands_data, annotated_frame), где:
            - hands_data: Словарь с данными рук {
                'left': landmarks_array или None,
                'right': landmarks_array или None,
                'count': количество обнаруженных рук
              }
            - annotated_frame: Кадр с нарисованными руками
        """
        # Конвертируем BGR в RGB (Mediapipe работает с RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Обрабатываем кадр
        results = self.hands.process(rgb_frame)
        
        annotated_frame = frame.copy()
        hands_data = {
            'left': None,
            'right': None,
            'count': 0
        }
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hands_data['count'] = len(results.multi_hand_landmarks)
            
            # Обрабатываем каждую обнаруженную руку
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                   results.multi_handedness):
                # Определяем какая рука (левая или правая)
                # Mediapipe определяет руку относительно владельца (как в зеркале)
                hand_label = handedness.classification[0].label.lower()  # 'left' или 'right'
                
                # Цвета для разных рук
                if hand_label == 'left':
                    color = (0, 255, 0)  # Зеленый для левой
                else:
                    color = (0, 0, 255)  # Красный для правой
                
                # Рисуем скелет руки на кадре
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=color, thickness=2)
                )
                
                # Извлекаем координаты всех 21 точки
                landmarks = []
                
                for landmark in hand_landmarks.landmark:
                    # Mediapipe возвращает нормализованные координаты (0-1)
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)
                    landmarks.append(landmark.z)
                
                # Преобразуем в numpy массив
                landmarks = np.array(landmarks, dtype=np.float32)
                
                # Сохраняем координаты для соответствующей руки
                hands_data[hand_label] = landmarks
                
                # Добавляем метку на видео
                h, w = frame.shape[:2]
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min = int(min(y_coords) * h)
                
                label = "Левая" if hand_label == 'left' else "Правая"
                cv2.putText(annotated_frame, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return hands_data, annotated_frame
    
    def detect_single_hand(self, frame):
        """
        Обнаруживает одну руку (для обратной совместимости).
        
        Args:
            frame: Кадр изображения
            
        Returns:
            Кортеж (landmarks, annotated_frame)
        """
        hands_data, annotated_frame = self.detect(frame)
        
        # Возвращаем первую найденную руку
        landmarks = hands_data.get('left') or hands_data.get('right')
        
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


class DynamicGestureRecognizer:
    """
    Класс для распознавания динамических жестов (движений).
    Сохраняет последовательность поз и распознает жесты по движениям.
    """
    
    def __init__(self, sequence_length=30, hands_mode='both'):
        """
        Инициализация распознавателя динамических жестов.
        
        Args:
            sequence_length: Длина последовательности кадров для анализа
            hands_mode: Режим рук ('single', 'both', 'left', 'right')
        """
        self.sequence_length = sequence_length
        self.hands_mode = hands_mode
        
        # Очереди для хранения последовательностей
        self.left_hand_sequence = deque(maxlen=sequence_length)
        self.right_hand_sequence = deque(maxlen=sequence_length)
        self.both_hands_sequence = deque(maxlen=sequence_length)
        
        # Статистика движения
        self.movement_history = deque(maxlen=10)
        
    def add_frame(self, hands_data):
        """
        Добавляет данные кадра в последовательность.
        
        Args:
            hands_data: Словарь с данными рук от GestureDetector
        """
        left = hands_data.get('left')
        right = hands_data.get('right')
        
        # Добавляем в соответствующие очереди
        if left is not None:
            self.left_hand_sequence.append(left)
        else:
            # Добавляем нули если рука не обнаружена
            self.left_hand_sequence.append(np.zeros(63, dtype=np.float32))
            
        if right is not None:
            self.right_hand_sequence.append(right)
        else:
            self.right_hand_sequence.append(np.zeros(63, dtype=np.float32))
        
        # Объединенная последовательность для обеих рук
        if left is not None and right is not None:
            # Конкатенируем обе руки
            both = np.concatenate([left, right])
        elif left is not None:
            # Только левая рука + нули для правой
            both = np.concatenate([left, np.zeros(63, dtype=np.float32)])
        elif right is not None:
            # Нули для левой + правая рука
            both = np.concatenate([np.zeros(63, dtype=np.float32), right])
        else:
            # Обе руки отсутствуют
            both = np.zeros(126, dtype=np.float32)
            
        self.both_hands_sequence.append(both)
    
    def get_sequence(self, mode='both'):
        """
        Получает текущую последовательность.
        
        Args:
            mode: Режим ('single', 'both', 'left', 'right')
            
        Returns:
            Массив последовательности [sequence_length, features]
        """
        if mode == 'left':
            return np.array(list(self.left_hand_sequence))
        elif mode == 'right':
            return np.array(list(self.right_hand_sequence))
        elif mode == 'both':
            return np.array(list(self.both_hands_sequence))
        else:
            # По умолчанию возвращаем обе руки
            return np.array(list(self.both_hands_sequence))
    
    def is_sequence_ready(self):
        """
        Проверяет, заполнена ли последовательность.
        
        Returns:
            True если последовательность готова для анализа
        """
        if self.hands_mode == 'left':
            return len(self.left_hand_sequence) >= self.sequence_length
        elif self.hands_mode == 'right':
            return len(self.right_hand_sequence) >= self.sequence_length
        else:
            return len(self.both_hands_sequence) >= self.sequence_length
    
    def calculate_movement(self, hands_data):
        """
        Вычисляет интенсивность движения.
        
        Args:
            hands_data: Текущие данные рук
            
        Returns:
            Значение интенсивности движения (0-1)
        """
        if len(self.both_hands_sequence) < 2:
            return 0.0
        
        # Берем текущий и предыдущий кадры
        current = self.both_hands_sequence[-1]
        previous = self.both_hands_sequence[-2]
        
        # Вычисляем разницу (эвклидово расстояние)
        diff = np.linalg.norm(current - previous)
        
        # Нормализуем (примерное значение для руки)
        movement = min(diff / 2.0, 1.0)
        
        self.movement_history.append(movement)
        
        return movement
    
    def get_average_movement(self):
        """
        Получает среднюю интенсивность движения за последние кадры.
        
        Returns:
            Средняя интенсивность движения
        """
        if len(self.movement_history) == 0:
            return 0.0
        return np.mean(list(self.movement_history))
    
    def clear(self):
        """Очищает все последовательности."""
        self.left_hand_sequence.clear()
        self.right_hand_sequence.clear()
        self.both_hands_sequence.clear()
        self.movement_history.clear()
    
    def extract_features(self):
        """
        Извлекает признаки из последовательности для классификации.
        
        Returns:
            Словарь с признаками движения
        """
        if not self.is_sequence_ready():
            return None
        
        sequence = self.get_sequence(self.hands_mode)
        
        # Базовые статистические признаки
        features = {
            'mean': np.mean(sequence, axis=0),
            'std': np.std(sequence, axis=0),
            'min': np.min(sequence, axis=0),
            'max': np.max(sequence, axis=0),
            'range': np.max(sequence, axis=0) - np.min(sequence, axis=0),
            'sequence': sequence,  # Полная последовательность
            'movement_intensity': self.get_average_movement()
        }
        
        return features

