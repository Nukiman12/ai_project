"""
Модель для распознавания динамических жестов (движений).

Использует LSTM для анализа последовательностей движений рук.
Поддерживает как одну, так и две руки одновременно.
"""

import torch
import torch.nn as nn
import numpy as np
import os


class DynamicGestureClassifier(nn.Module):
    """
    LSTM модель для классификации динамических жестов.
    
    Архитектура:
    - LSTM слои для анализа временных последовательностей
    - Fully connected слои для классификации
    - Dropout для регуляризации
    """
    
    def __init__(self, input_size=126, hidden_size=256, num_layers=2, 
                 num_classes=10, dropout=0.3, bidirectional=True):
        """
        Инициализация модели.
        
        Args:
            input_size: Размер входных данных (63 для 1 руки, 126 для 2 рук)
            hidden_size: Размер скрытого слоя LSTM
            num_layers: Количество LSTM слоев
            num_classes: Количество классов жестов
            dropout: Процент dropout
            bidirectional: Использовать двунаправленный LSTM
        """
        super(DynamicGestureClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Размер после LSTM (с учетом bidirectional)
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected слои
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        """
        Прямой проход.
        
        Args:
            x: Входной тензор [batch_size, sequence_length, input_size]
            
        Returns:
            Логиты [batch_size, num_classes]
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Берем последний выход
        if self.bidirectional:
            # Конкатенируем forward и backward последние состояния
            out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            out = h_n[-1,:,:]
        
        # Fully connected слои
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out


class DynamicGestureModelWrapper:
    """
    Обертка для модели динамических жестов.
    Упрощает загрузку, сохранение и предсказание.
    """
    
    def __init__(self, model_path=None, input_size=126, num_classes=10, 
                 gesture_classes=None, use_dummy=False):
        """
        Инициализация обертки модели.
        
        Args:
            model_path: Путь к сохраненной модели
            input_size: Размер входных данных
            num_classes: Количество классов
            gesture_classes: Список названий жестов
            use_dummy: Использовать режим заглушки
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_dummy = use_dummy
        
        # Названия жестов (по умолчанию)
        if gesture_classes is None:
            self.gesture_classes = [
                'Махать рукой',      # Wave
                'Показать пальцем',  # Point
                'Хлопать',           # Clap
                'Круговое движение', # Circle
                'Движение вверх',    # Up
                'Движение вниз',     # Down
                'Движение влево',    # Left
                'Движение вправо',   # Right
                'Приближение',       # Come
                'Удаление'           # Go
            ]
        else:
            self.gesture_classes = gesture_classes
            self.num_classes = len(gesture_classes)
        
        # Создаем модель
        self.model = DynamicGestureClassifier(
            input_size=input_size,
            num_classes=self.num_classes
        )
        
        # Загружаем веса если путь указан
        if model_path and os.path.exists(model_path) and not use_dummy:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                print(f"✓ Модель загружена: {model_path}")
            except Exception as e:
                print(f"⚠ Ошибка загрузки модели: {e}")
                print("→ Используется режим заглушки")
                self.use_dummy = True
        else:
            if not use_dummy:
                print("⚠ Модель не найдена, используется режим заглушки")
            self.use_dummy = True
        
        # Режим eval
        self.model.eval()
        
        # Dummy счетчик
        self.dummy_counter = 0
    
    def predict(self, sequence):
        """
        Предсказывает жест по последовательности.
        
        Args:
            sequence: Последовательность поз [sequence_length, features]
            
        Returns:
            Кортеж (название_жеста, уверенность)
        """
        if self.use_dummy:
            # Режим заглушки - возвращаем случайный жест
            self.dummy_counter += 1
            if self.dummy_counter % 10 == 0:  # Меняем жест каждые 10 кадров
                gesture_idx = np.random.randint(0, self.num_classes)
                confidence = np.random.uniform(0.7, 0.95)
                return self.gesture_classes[gesture_idx], confidence
            return None, 0.0
        
        try:
            # Преобразуем в тензор
            if isinstance(sequence, np.ndarray):
                sequence_tensor = torch.FloatTensor(sequence)
            else:
                sequence_tensor = sequence
            
            # Добавляем batch dimension
            if len(sequence_tensor.shape) == 2:
                sequence_tensor = sequence_tensor.unsqueeze(0)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                gesture_idx = predicted.item()
                confidence_val = confidence.item()
                
                gesture_name = self.gesture_classes[gesture_idx]
                
                return gesture_name, confidence_val
                
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return None, 0.0
    
    def save(self, path):
        """
        Сохраняет модель.
        
        Args:
            path: Путь для сохранения
        """
        try:
            # Сохраняем словарь состояния
            state = {
                'model_state_dict': self.model.state_dict(),
                'input_size': self.input_size,
                'num_classes': self.num_classes,
                'gesture_classes': self.gesture_classes
            }
            torch.save(state, path)
            print(f"✓ Модель сохранена: {path}")
        except Exception as e:
            print(f"✗ Ошибка сохранения модели: {e}")
    
    @staticmethod
    def load(path):
        """
        Загружает модель из файла.
        
        Args:
            path: Путь к файлу модели
            
        Returns:
            Экземпляр DynamicGestureModelWrapper
        """
        try:
            state = torch.load(path, map_location='cpu')
            
            wrapper = DynamicGestureModelWrapper(
                model_path=None,
                input_size=state.get('input_size', 126),
                num_classes=state.get('num_classes', 10),
                gesture_classes=state.get('gesture_classes'),
                use_dummy=False
            )
            
            wrapper.model.load_state_dict(state['model_state_dict'])
            wrapper.model.eval()
            
            print(f"✓ Модель загружена из {path}")
            return wrapper
            
        except Exception as e:
            print(f"✗ Ошибка загрузки: {e}")
            return DynamicGestureModelWrapper(use_dummy=True)


class GestureDataCollector:
    """
    Класс для сбора данных жестов в реальном времени.
    Используется для обучения модели на ваших движениях.
    """
    
    def __init__(self):
        """Инициализация коллектора."""
        self.samples = []
        self.labels = []
        self.current_gesture_name = None
        self.is_recording = False
        
    def start_recording(self, gesture_name):
        """
        Начинает запись жеста.
        
        Args:
            gesture_name: Название жеста
        """
        self.current_gesture_name = gesture_name
        self.is_recording = True
        print(f"🔴 Запись жеста: {gesture_name}")
    
    def stop_recording(self):
        """Останавливает запись."""
        self.is_recording = False
        self.current_gesture_name = None
        print("⏸ Запись остановлена")
    
    def add_sample(self, sequence, label):
        """
        Добавляет образец.
        
        Args:
            sequence: Последовательность поз
            label: Метка жеста (индекс или название)
        """
        self.samples.append(sequence)
        self.labels.append(label)
    
    def get_dataset(self):
        """
        Получает собранный датасет.
        
        Returns:
            Кортеж (samples, labels)
        """
        return np.array(self.samples), np.array(self.labels)
    
    def save_dataset(self, path):
        """
        Сохраняет датасет.
        
        Args:
            path: Путь для сохранения
        """
        try:
            np.savez(path, 
                    samples=np.array(self.samples),
                    labels=np.array(self.labels))
            print(f"✓ Датасет сохранен: {path}")
            print(f"  Образцов: {len(self.samples)}")
        except Exception as e:
            print(f"✗ Ошибка сохранения: {e}")
    
    def load_dataset(self, path):
        """
        Загружает датасет.
        
        Args:
            path: Путь к файлу
        """
        try:
            data = np.load(path)
            self.samples = list(data['samples'])
            self.labels = list(data['labels'])
            print(f"✓ Датасет загружен: {path}")
            print(f"  Образцов: {len(self.samples)}")
        except Exception as e:
            print(f"✗ Ошибка загрузки: {e}")
    
    def clear(self):
        """Очищает собранные данные."""
        self.samples.clear()
        self.labels.clear()
        print("Данные очищены")
    
    def get_stats(self):
        """
        Получает статистику по собранным данным.
        
        Returns:
            Словарь со статистикой
        """
        if len(self.labels) == 0:
            return {'total': 0, 'by_class': {}}
        
        unique, counts = np.unique(self.labels, return_counts=True)
        by_class = dict(zip(unique, counts))
        
        return {
            'total': len(self.samples),
            'by_class': by_class,
            'classes': len(unique)
        }


# Список динамических жестов по умолчанию
DYNAMIC_GESTURE_CLASSES = [
    'Махать рукой',      # Wave - движение руки из стороны в сторону
    'Показать пальцем',  # Point - указательное движение
    'Хлопать',           # Clap - хлопки руками
    'Круговое движение', # Circle - круговое движение рукой
    'Движение вверх',    # Up - движение руки вверх
    'Движение вниз',     # Down - движение руки вниз
    'Движение влево',    # Left - движение руки влево
    'Движение вправо',   # Right - движение руки вправо
    'Приближение',       # Come - приближающее движение
    'Удаление'           # Go - отталкивающее движение
]


