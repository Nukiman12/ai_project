"""
Модуль обучения для SignVoiceAI Enterprise Edition.

Возможности:
- Сбор данных жестов в реальном времени
- Обучение продвинутой модели
- Визуализация процесса обучения
- Метрики качества
- Автоматическое сохранение лучшей модели
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from pathlib import Path
import json
from datetime import datetime
import threading
from collections import deque


class GestureDataset(Dataset):
    """Dataset для загрузки данных жестов."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Инициализация dataset.
        
        Args:
            features: [n_samples, 63]
            labels: [n_samples]
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DataCollector:
    """
    Сборщик данных для обучения.
    Накапливает примеры жестов с метаданными.
    """
    
    def __init__(self, save_dir: str = "training_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Данные: {gesture_name: [samples]}
        self.data = {}
        
        # Метаданные
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'total_samples': 0,
            'gesture_counts': {}
        }
    
    def add_sample(self, gesture_name: str, landmarks: np.ndarray):
        """
        Добавление примера жеста.
        
        Args:
            gesture_name: Название жеста
            landmarks: Landmarks [63] или [21, 3]
        """
        if landmarks.shape == (21, 3):
            landmarks = landmarks.flatten()
        
        if gesture_name not in self.data:
            self.data[gesture_name] = []
            self.metadata['gesture_counts'][gesture_name] = 0
        
        self.data[gesture_name].append(landmarks.copy())
        self.metadata['gesture_counts'][gesture_name] += 1
        self.metadata['total_samples'] += 1
    
    def get_samples_count(self, gesture_name: Optional[str] = None) -> int:
        """Получение количества примеров."""
        if gesture_name:
            return len(self.data.get(gesture_name, []))
        return self.metadata['total_samples']
    
    def get_gesture_names(self) -> List[str]:
        """Получение списка жестов."""
        return list(self.data.keys())
    
    def prepare_training_data(self, test_split: float = 0.2) -> Tuple:
        """
        Подготовка данных для обучения.
        
        Args:
            test_split: Доля тестовых данных
            
        Returns:
            (X_train, X_test, y_train, y_test, gesture_classes)
        """
        if not self.data:
            raise ValueError("Нет данных для обучения!")
        
        # Собираем все данные
        X_all = []
        y_all = []
        gesture_classes = sorted(self.data.keys())
        gesture_to_idx = {name: idx for idx, name in enumerate(gesture_classes)}
        
        for gesture_name, samples in self.data.items():
            label_idx = gesture_to_idx[gesture_name]
            for sample in samples:
                X_all.append(sample)
                y_all.append(label_idx)
        
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.int64)
        
        # Перемешиваем
        indices = np.random.permutation(len(X_all))
        X_all = X_all[indices]
        y_all = y_all[indices]
        
        # Разделяем на train/test
        n_test = int(len(X_all) * test_split)
        
        X_test = X_all[:n_test]
        y_test = y_all[:n_test]
        X_train = X_all[n_test:]
        y_train = y_all[n_test:]
        
        return X_train, X_test, y_train, y_test, gesture_classes
    
    def save(self, filename: Optional[str] = None):
        """Сохранение данных."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.npz"
        
        filepath = self.save_dir / filename
        
        # Подготавливаем данные для сохранения
        save_dict = {
            'metadata': json.dumps(self.metadata),
        }
        
        for gesture_name, samples in self.data.items():
            save_dict[gesture_name] = np.array(samples)
        
        np.savez_compressed(filepath, **save_dict)
        print(f"✓ Данные сохранены: {filepath}")
    
    def load(self, filepath: str):
        """Загрузка данных."""
        data_loaded = np.load(filepath, allow_pickle=True)
        
        self.metadata = json.loads(str(data_loaded['metadata']))
        self.data = {}
        
        for key in data_loaded.keys():
            if key != 'metadata':
                self.data[key] = list(data_loaded[key])
        
        print(f"✓ Данные загружены: {filepath}")


class ModelTrainer:
    """
    Тренер модели с визуализацией и метриками.
    """
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # История обучения
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Статус
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.best_val_acc = 0.0
        
        # Коллбэки
        self.callbacks = []
    
    def add_callback(self, callback: Callable):
        """Добавление коллбэка для уведомлений о прогрессе."""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, event: str, data: dict):
        """Уведомление коллбэков."""
        for callback in self.callbacks:
            try:
                callback(event, data)
            except Exception as e:
                print(f"⚠ Ошибка в callback: {e}")
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4,
              patience: int = 10):
        """
        Обучение модели.
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            epochs: Количество эпох
            learning_rate: Скорость обучения
            weight_decay: L2 регуляризация
            patience: Терпение для early stopping
        """
        self.is_training = True
        self.total_epochs = epochs
        
        # Оптимизатор и планировщик
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        best_val_acc = 0.0
        epochs_without_improvement = 0
        
        self._notify_callbacks('training_start', {
            'total_epochs': epochs,
            'learning_rate': learning_rate
        })
        
        for epoch in range(epochs):
            if not self.is_training:
                print("⚠ Обучение прервано пользователем")
                break
            
            self.current_epoch = epoch + 1
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            
            # Обновляем scheduler
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Сохраняем историю
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Уведомляем о прогрессе
            self._notify_callbacks('epoch_end', {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': current_lr
            })
            
            # Early stopping и сохранение лучшей модели
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_acc = best_val_acc
                epochs_without_improvement = 0
                
                self._notify_callbacks('best_model', {
                    'epoch': epoch + 1,
                    'val_acc': val_acc
                })
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"⚠ Early stopping: {patience} эпох без улучшений")
                break
        
        self.is_training = False
        
        self._notify_callbacks('training_complete', {
            'best_val_acc': best_val_acc,
            'total_epochs': epoch + 1
        })
    
    def stop_training(self):
        """Остановка обучения."""
        self.is_training = False
    
    def get_progress(self) -> dict:
        """Получение текущего прогресса."""
        return {
            'is_training': self.is_training,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }




