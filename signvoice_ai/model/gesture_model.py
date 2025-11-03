"""
Модуль с PyTorch моделью для распознавания жестов.

Модель принимает координаты суставов руки (21 точка × 3 координаты = 63 значения)
и классифицирует жест в одно из слов: "Hello", "Thanks", "Yes", "No".
"""

import torch
import torch.nn as nn
import random


class GestureClassifier(nn.Module):
    """
    Простая нейросеть для классификации жестов.
    
    Архитектура:
    - Входной слой: 63 значения (21 точка × 3 координаты)
    - Скрытый слой: 128 нейронов
    - Выходной слой: 4 класса (Hello, Thanks, Yes, No)
    """
    
    def __init__(self, input_size=63, hidden_size=128, num_classes=4):
        super(GestureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Прямой проход через сеть.
        
        Args:
            x: Тензор с координатами суставов [batch_size, 63]
            
        Returns:
            Логиты для каждого класса [batch_size, 4]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Список распознаваемых жестов
GESTURE_CLASSES = ["Hello", "Thanks", "Yes", "No"]


class GestureModelWrapper:
    """
    Обертка для удобной работы с моделью распознавания жестов.
    Предоставляет методы для загрузки модели и предсказания жестов.
    """
    
    def __init__(self, model_path=None, use_dummy=True):
        """
        Инициализация модели.
        
        Args:
            model_path: Путь к файлу с обученной моделью (.pth или .pt)
            use_dummy: Если True и модель не найдена, использовать заглушку
        """
        self.model_path = model_path
        self.use_dummy = use_dummy
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_dummy = False
        
        # Попытка загрузить модель
        if model_path:
            try:
                self.load_model(model_path)
                print(f"Модель загружена из {model_path}")
            except Exception as e:
                print(f"Не удалось загрузить модель из {model_path}: {e}")
                if use_dummy:
                    print("Используется заглушка с случайными жестами")
                    self.is_dummy = True
        else:
            if use_dummy:
                print("Путь к модели не указан. Используется заглушка с случайными жестами")
                self.is_dummy = True
    
    def load_model(self, model_path):
        """
        Загружает обученную модель из файла.
        
        Args:
            model_path: Путь к файлу модели
        """
        self.model = GestureClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.is_dummy = False
    
    def predict(self, landmarks):
        """
        Предсказывает жест на основе координат суставов.
        
        Args:
            landmarks: Список или numpy массив с координатами суставов (63 значения)
            
        Returns:
            Кортеж (название жеста, уверенность)
        """
        if self.is_dummy:
            # Заглушка: возвращает случайный жест
            gesture = random.choice(GESTURE_CLASSES)
            confidence = random.uniform(0.6, 0.95)
            return gesture, confidence
        
        if self.model is None:
            raise RuntimeError("Модель не инициализирована")
        
        # Преобразуем входные данные в тензор
        if isinstance(landmarks, list):
            landmarks = torch.tensor(landmarks, dtype=torch.float32)
        else:
            landmarks = torch.from_numpy(landmarks).float()
        
        # Если landmarks - это плоский массив, преобразуем в батч
        if landmarks.dim() == 1:
            landmarks = landmarks.unsqueeze(0)
        
        # Проверяем размерность
        if landmarks.shape[1] != 63:
            raise ValueError(f"Ожидается 63 значения, получено {landmarks.shape[1]}")
        
        # Переводим на нужное устройство
        landmarks = landmarks.to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(landmarks)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            gesture = GESTURE_CLASSES[predicted_class.item()]
            confidence_value = confidence.item()
        
        return gesture, confidence_value
    
    def normalize_landmarks(self, landmarks):
        """
        Нормализует координаты суставов относительно запястья.
        Это делает модель более устойчивой к различным положениям руки.
        
        Args:
            landmarks: Массив координат (21 точка × 3 координаты)
            
        Returns:
            Нормализованные координаты
        """
        if len(landmarks) < 63:
            raise ValueError("Недостаточно координат для нормализации")
        
        # Преобразуем в numpy массив если нужно
        import numpy as np
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks)
        
        # Преобразуем в форму [21, 3]
        landmarks = landmarks.reshape(21, 3)
        
        # Берем координаты запястья (первая точка)
        wrist = landmarks[0]
        
        # Нормализуем относительно запястья
        normalized = landmarks - wrist
        
        # Флаттеним обратно в массив [63]
        return normalized.flatten()

