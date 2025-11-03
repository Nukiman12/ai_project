"""
Скрипт для обучения модели распознавания жестов.

Использование:
    python train_model.py --data data --output models/gesture_model.pth

Скрипт:
1. Загружает данные из CSV файлов
2. Разделяет на обучающую и тестовую выборки
3. Обучает модель PyTorch
4. Сохраняет обученную модель
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.gesture_model import GestureClassifier, GESTURE_CLASSES


class GestureDataset(Dataset):
    """
    Датасет для загрузки данных жестов.
    """
    
    def __init__(self, features, labels):
        """
        Инициализация датасета.
        
        Args:
            features: Массив признаков (numpy array) [n_samples, 63]
            labels: Массив меток (numpy array) [n_samples]
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data(data_dir):
    """
    Загружает данные из CSV файлов.
    
    Args:
        data_dir: Папка с CSV файлами
        
    Returns:
        Кортеж (features, labels) где:
        - features: numpy array [n_samples, 63]
        - labels: numpy array [n_samples] с индексами классов
    """
    all_features = []
    all_labels = []
    
    # Проходим по всем CSV файлам в папке
    for filename in os.listdir(data_dir):
        if filename.endswith('_samples.csv'):
            filepath = os.path.join(data_dir, filename)
            
            print(f"Загрузка данных из: {filename}")
            df = pd.read_csv(filepath)
            
            # Извлекаем название жеста из первого столбца
            gesture_name = df.iloc[0, 0]
            
            # Проверяем, что жест входит в список распознаваемых
            if gesture_name not in GESTURE_CLASSES:
                print(f"Пропуск неизвестного жеста: {gesture_name}")
                continue
            
            # Извлекаем признаки (все столбцы кроме первого)
            features = df.iloc[:, 1:].values  # [n_samples, 63]
            
            # Получаем индекс класса
            class_idx = GESTURE_CLASSES.index(gesture_name)
            
            # Создаем метки
            labels = np.full(len(features), class_idx)
            
            all_features.append(features)
            all_labels.append(labels)
            
            print(f"  Загружено {len(features)} образцов для жеста '{gesture_name}'")
    
    if len(all_features) == 0:
        raise ValueError(f"Не найдено данных в папке {data_dir}")
    
    # Объединяем все данные
    features = np.vstack(all_features)
    labels = np.hstack(all_labels)
    
    print(f"\nВсего загружено: {len(features)} образцов")
    print(f"Распределение по классам:")
    for i, gesture in enumerate(GESTURE_CLASSES):
        count = np.sum(labels == i)
        print(f"  {gesture}: {count} образцов")
    
    return features, labels


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cpu'):
    """
    Обучает модель.
    
    Args:
        model: Модель PyTorch
        train_loader: DataLoader для обучающей выборки
        val_loader: DataLoader для валидационной выборки
        num_epochs: Количество эпох
        learning_rate: Скорость обучения
        device: Устройство (cpu или cuda)
        
    Returns:
        История обучения (словарь с losses и accuracies)
    """
    model.to(device)
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # История обучения
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\n" + "=" * 60)
    print("Начало обучения")
    print("=" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Прямой проход
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Обратный проход
            loss.backward()
            optimizer.step()
            
            # Статистика
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Выводим прогресс
        print(f"Эпоха {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✓ Новая лучшая модель! (Val Acc: {best_val_acc:.2f}%)")
    
    print("\n" + "=" * 60)
    print(f"Обучение завершено! Лучшая точность валидации: {best_val_acc:.2f}%")
    print("=" * 60)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description='Обучение модели распознавания жестов'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data',
        help='Папка с данными (CSV файлы)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/gesture_model.pth',
        help='Путь для сохранения обученной модели'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Количество эпох обучения (по умолчанию 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Размер батча (по умолчанию 32)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Скорость обучения (по умолчанию 0.001)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Доля тестовой выборки (по умолчанию 0.2)'
    )
    
    args = parser.parse_args()
    
    # Проверяем наличие папки с данными
    if not os.path.exists(args.data):
        print(f"Ошибка: папка с данными не найдена: {args.data}")
        print("Сначала соберите данные с помощью train_collect_data.py")
        return
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Использование устройства: {device}")
    
    # Загружаем данные
    print("\nЗагрузка данных...")
    features, labels = load_data(args.data)
    
    # Разделяем на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    
    print(f"\nРазделение данных:")
    print(f"  Обучающая выборка: {len(X_train)} образцов")
    print(f"  Валидационная выборка: {len(X_val)} образцов")
    
    # Создаем датасеты и загрузчики
    train_dataset = GestureDataset(X_train, y_train)
    val_dataset = GestureDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Создаем модель
    print("\nСоздание модели...")
    model = GestureClassifier(input_size=63, hidden_size=128, num_classes=5)
    print(f"  Параметров модели: {sum(p.numel() for p in model.parameters())}")
    
    # Обучаем модель
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Сохраняем модель
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\nМодель сохранена в: {args.output}")
    
    # Выводим финальную статистику
    print("\n" + "=" * 60)
    print("Финальная статистика:")
    print("=" * 60)
    print(f"Точность на обучающей выборке: {history['train_acc'][-1]:.2f}%")
    print(f"Точность на валидационной выборке: {history['val_acc'][-1]:.2f}%")
    print(f"\nИспользуйте обученную модель:")
    print(f"  python main.py --model {args.output}")


if __name__ == "__main__":
    main()

