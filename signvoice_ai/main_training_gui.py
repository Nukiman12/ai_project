"""
SignVoiceAI - GUI для обучения на динамических жестах.

Интерфейс для:
- Записи ваших движений в реальном времени
- Обучения модели на собранных данных
- Тестирования обученной модели
- Поддержка двух рук одновременно
"""

import cv2
import sys
import os
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
from datetime import datetime
from collections import deque
import numpy as np
from tkinter import messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.camera import Camera
from utils.gestures import GestureDetector, DynamicGestureRecognizer
from utils.speech import TextToSpeech
from model.dynamic_gesture_model import (
    DynamicGestureModelWrapper, 
    GestureDataCollector,
    DYNAMIC_GESTURE_CLASSES
)

# Настройка темы
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class GestureDataset(Dataset):
    """Dataset для обучения модели жестов."""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class TrainingGUI:
    """
    GUI для обучения модели на ваших движениях.
    """
    
    def __init__(self, root, camera_index=0):
        """Инициализация интерфейса обучения."""
        self.root = root
        self.root.title("SignVoiceAI - Обучение на движениях")
        self.root.geometry("1600x1000")
        
        # Компоненты
        self.camera_index = camera_index
        self.camera = None
        self.gesture_detector = None
        self.dynamic_recognizer = None
        self.tts = None
        self.is_running = False
        self.video_thread = None
        
        # Данные для обучения
        self.data_collector = GestureDataCollector()
        self.gesture_classes = DYNAMIC_GESTURE_CLASSES.copy()
        self.current_gesture_idx = 0
        self.is_recording = False
        self.recorded_sequences = deque(maxlen=100)
        
        # Модель
        self.model = None
        self.is_training = False
        
        # Состояние
        self.mode = 'collect'  # 'collect', 'train', 'test'
        
        # Инициализация компонентов
        self.init_components()
        
        # Создание GUI
        self.create_gui()
        
        # Обработчик закрытия
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def init_components(self):
        """Инициализация компонентов."""
        print("=" * 60)
        print("Инициализация Training GUI...")
        print("=" * 60)
        
        # Камера
        self.camera = Camera(camera_index=self.camera_index, width=640, height=480)
        
        # Детектор жестов (2 руки)
        self.gesture_detector = GestureDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2,
            detect_both_hands=True
        )
        
        # Распознаватель динамических жестов
        self.dynamic_recognizer = DynamicGestureRecognizer(
            sequence_length=30,
            hands_mode='both'
        )
        
        # TTS
        self.tts = TextToSpeech(rate=150, volume=0.8)
        
        print("✓ Инициализация завершена!")
    
    def create_gui(self):
        """Создание графического интерфейса."""
        
        # Главный контейнер
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Левая панель (видео и управление)
        left_panel = ctk.CTkFrame(main_container, fg_color="transparent")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Правая панель (управление обучением)
        right_panel = ctk.CTkFrame(main_container, fg_color="transparent", width=500)
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Создание элементов
        self.create_video_panel(left_panel)
        self.create_mode_panel(left_panel)
        self.create_recording_panel(right_panel)
        self.create_gesture_list_panel(right_panel)
        self.create_training_panel(right_panel)
        self.create_stats_panel(right_panel)
    
    def create_video_panel(self, parent):
        """Создание панели с видео."""
        video_container = ctk.CTkFrame(parent, corner_radius=15)
        video_container.pack(fill="both", expand=True, pady=(0, 15))
        
        # Заголовок
        header_frame = ctk.CTkFrame(video_container, corner_radius=10, 
                                    fg_color=("#2b2b2b", "#1a1a1a"))
        header_frame.pack(fill="x", padx=15, pady=15)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="📹 Видео (2 руки)",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1f6aa5", "#42a5f5")
        )
        title_label.pack(side="left", padx=20, pady=12)
        
        # Индикатор записи
        self.recording_indicator = ctk.CTkLabel(
            header_frame,
            text="⚫ Готов",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#9ca3af", "#9ca3af")
        )
        self.recording_indicator.pack(side="right", padx=20, pady=12)
        
        # Видео Canvas
        video_frame = ctk.CTkFrame(video_container, corner_radius=10, fg_color="#000000")
        video_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.video_canvas = ctk.CTkCanvas(
            video_frame,
            bg="#000000",
            highlightthickness=0
        )
        self.video_canvas.pack(fill="both", expand=True, padx=3, pady=3)
        
        # Информация
        self.info_overlay = ctk.CTkLabel(
            video_frame,
            text="Нажмите 'Старт' для начала",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        self.info_overlay.place(relx=0.5, rely=0.5, anchor="center")
    
    def create_mode_panel(self, parent):
        """Создание панели выбора режима."""
        mode_frame = ctk.CTkFrame(parent, corner_radius=15, height=90)
        mode_frame.pack(fill="x")
        mode_frame.pack_propagate(False)
        
        title = ctk.CTkLabel(
            mode_frame,
            text="Режим работы:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=(15, 5))
        
        # Кнопки режимов
        buttons_frame = ctk.CTkFrame(mode_frame, fg_color="transparent")
        buttons_frame.pack(expand=True, fill="x", padx=20)
        
        self.collect_btn = ctk.CTkButton(
            buttons_frame,
            text="📝 Сбор данных",
            command=lambda: self.set_mode('collect'),
            fg_color=("#1976d2", "#1976d2"),
            hover_color=("#42a5f5", "#42a5f5")
        )
        self.collect_btn.pack(side="left", expand=True, fill="x", padx=2)
        
        self.train_btn = ctk.CTkButton(
            buttons_frame,
            text="🧠 Обучение",
            command=lambda: self.set_mode('train'),
            fg_color=("#4a5568", "#4a5568"),
            hover_color=("#5a657a", "#5a657a")
        )
        self.train_btn.pack(side="left", expand=True, fill="x", padx=2)
        
        self.test_btn = ctk.CTkButton(
            buttons_frame,
            text="🎯 Тестирование",
            command=lambda: self.set_mode('test'),
            fg_color=("#4a5568", "#4a5568"),
            hover_color=("#5a657a", "#5a657a")
        )
        self.test_btn.pack(side="left", expand=True, fill="x", padx=2)
    
    def create_recording_panel(self, parent):
        """Создание панели записи."""
        self.recording_frame = ctk.CTkFrame(parent, corner_radius=15)
        self.recording_frame.pack(fill="x", pady=(0, 15))
        
        # Заголовок
        header = ctk.CTkLabel(
            self.recording_frame,
            text="🎬 Управление записью",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Кнопки
        btn_frame = ctk.CTkFrame(self.recording_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        self.start_button = ctk.CTkButton(
            btn_frame,
            text="▶ Старт",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=50,
            fg_color=("#1976d2", "#1976d2"),
            hover_color=("#42a5f5", "#42a5f5"),
            command=self.toggle_camera
        )
        self.start_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        self.record_button = ctk.CTkButton(
            btn_frame,
            text="🔴 Записать",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=50,
            fg_color=("#c92a2a", "#c92a2a"),
            hover_color=("#ff6b6b", "#ff6b6b"),
            state="disabled",
            command=self.toggle_recording
        )
        self.record_button.pack(side="left", expand=True, fill="x", padx=(5, 0))
    
    def create_gesture_list_panel(self, parent):
        """Создание панели списка жестов."""
        self.gesture_list_frame = ctk.CTkFrame(parent, corner_radius=15)
        self.gesture_list_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        # Заголовок
        header = ctk.CTkLabel(
            self.gesture_list_frame,
            text="📋 Жесты для записи",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Scrollable список
        self.gesture_scrollable = ctk.CTkScrollableFrame(
            self.gesture_list_frame,
            corner_radius=10,
            fg_color=("#2b2b2b", "#1a1a1a")
        )
        self.gesture_scrollable.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Создаем кнопки для каждого жеста
        self.gesture_buttons = []
        for idx, gesture_name in enumerate(self.gesture_classes):
            btn = ctk.CTkButton(
                self.gesture_scrollable,
                text=f"{idx+1}. {gesture_name}",
                font=ctk.CTkFont(size=12),
                height=40,
                anchor="w",
                command=lambda i=idx: self.select_gesture(i)
            )
            btn.pack(fill="x", pady=2)
            self.gesture_buttons.append(btn)
        
        # Выделяем первый
        self.select_gesture(0)
    
    def create_training_panel(self, parent):
        """Создание панели обучения."""
        self.training_frame = ctk.CTkFrame(parent, corner_radius=15)
        self.training_frame.pack(fill="x", pady=(0, 15))
        
        # Заголовок
        header = ctk.CTkLabel(
            self.training_frame,
            text="🧠 Обучение модели",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Параметры
        params_frame = ctk.CTkFrame(self.training_frame, fg_color="transparent")
        params_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(params_frame, text="Эпох:").pack(anchor="w")
        self.epochs_entry = ctk.CTkEntry(params_frame, width=100)
        self.epochs_entry.insert(0, "50")
        self.epochs_entry.pack(fill="x", pady=(0, 5))
        
        ctk.CTkLabel(params_frame, text="Batch size:").pack(anchor="w")
        self.batch_entry = ctk.CTkEntry(params_frame, width=100)
        self.batch_entry.insert(0, "16")
        self.batch_entry.pack(fill="x")
        
        # Кнопка обучения
        self.train_model_btn = ctk.CTkButton(
            self.training_frame,
            text="🚀 Обучить модель",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=50,
            fg_color=("#1976d2", "#1976d2"),
            hover_color=("#42a5f5", "#42a5f5"),
            command=self.start_training
        )
        self.train_model_btn.pack(fill="x", padx=15, pady=(0, 15))
        
        # Прогресс
        self.training_progress = ctk.CTkProgressBar(self.training_frame)
        self.training_progress.pack(fill="x", padx=15, pady=(0, 10))
        self.training_progress.set(0)
        
        self.training_status = ctk.CTkLabel(
            self.training_frame,
            text="Готов к обучению",
            font=ctk.CTkFont(size=10)
        )
        self.training_status.pack(padx=15, pady=(0, 15))
    
    def create_stats_panel(self, parent):
        """Создание панели статистики."""
        stats_frame = ctk.CTkFrame(parent, corner_radius=15)
        stats_frame.pack(fill="x")
        
        # Заголовок
        header = ctk.CTkLabel(
            stats_frame,
            text="📊 Статистика",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Статистика
        self.stats_label = ctk.CTkLabel(
            stats_frame,
            text="Образцов собрано: 0\nГотовых жестов: 0/10",
            font=ctk.CTkFont(size=12),
            justify="left"
        )
        self.stats_label.pack(padx=15, pady=(0, 15), anchor="w")
    
    def set_mode(self, mode):
        """Установка режима работы."""
        self.mode = mode
        
        # Обновляем цвета кнопок
        colors = {
            'active': ("#1976d2", "#1976d2"),
            'inactive': ("#4a5568", "#4a5568")
        }
        
        self.collect_btn.configure(
            fg_color=colors['active'] if mode == 'collect' else colors['inactive']
        )
        self.train_btn.configure(
            fg_color=colors['active'] if mode == 'train' else colors['inactive']
        )
        self.test_btn.configure(
            fg_color=colors['active'] if mode == 'test' else colors['inactive']
        )
        
        print(f"Режим: {mode}")
    
    def select_gesture(self, idx):
        """Выбор жеста для записи."""
        self.current_gesture_idx = idx
        
        # Обновляем цвета кнопок
        for i, btn in enumerate(self.gesture_buttons):
            if i == idx:
                btn.configure(fg_color=("#1976d2", "#1976d2"))
            else:
                btn.configure(fg_color=("#4a5568", "#4a5568"))
    
    def toggle_camera(self):
        """Переключение камеры."""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Запуск камеры."""
        if not self.camera.open():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру")
            return
        
        self.is_running = True
        self.start_button.configure(text="⏸ Стоп", fg_color="#c92a2a")
        self.record_button.configure(state="normal")
        self.info_overlay.place_forget()
        
        # Запуск потока
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
    
    def stop_camera(self):
        """Остановка камеры."""
        self.is_running = False
        self.start_button.configure(text="▶ Старт", fg_color="#1976d2")
        self.record_button.configure(state="disabled")
        
        if self.camera:
            self.camera.release()
        
        self.info_overlay.place(relx=0.5, rely=0.5, anchor="center")
    
    def toggle_recording(self):
        """Переключение записи."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Начало записи жеста."""
        self.is_recording = True
        self.record_button.configure(text="⏹ Стоп запись", fg_color="#ff6b6b")
        self.recording_indicator.configure(
            text="🔴 ЗАПИСЬ", 
            text_color=("#ff6b6b", "#ff6b6b")
        )
        
        # Очищаем буфер
        self.dynamic_recognizer.clear()
        self.recorded_sequences.clear()
        
        gesture_name = self.gesture_classes[self.current_gesture_idx]
        print(f"🔴 Запись жеста: {gesture_name}")
        self.tts.speak(f"Записываем {gesture_name}")
    
    def stop_recording(self):
        """Остановка записи."""
        self.is_recording = False
        self.record_button.configure(text="🔴 Записать", fg_color="#c92a2a")
        self.recording_indicator.configure(
            text="⚫ Готов",
            text_color=("#9ca3af", "#9ca3af")
        )
        
        # Сохраняем записанные последовательности
        gesture_name = self.gesture_classes[self.current_gesture_idx]
        count = len(self.recorded_sequences)
        
        for seq in self.recorded_sequences:
            self.data_collector.add_sample(seq, self.current_gesture_idx)
        
        print(f"✓ Сохранено {count} образцов для '{gesture_name}'")
        self.tts.speak(f"Сохранено {count} образцов")
        
        # Обновляем статистику
        self.update_stats()
    
    def process_video(self):
        """Обработка видеопотока."""
        while self.is_running:
            ret, frame = self.camera.read()
            
            if not ret:
                break
            
            # Обнаружение рук
            hands_data, annotated_frame = self.gesture_detector.detect(frame)
            
            # Добавляем в динамический распознаватель
            self.dynamic_recognizer.add_frame(hands_data)
            
            # Если записываем
            if self.is_recording and self.dynamic_recognizer.is_sequence_ready():
                sequence = self.dynamic_recognizer.get_sequence('both')
                self.recorded_sequences.append(sequence.copy())
            
            # Отображение кадра
            self.display_frame(annotated_frame)
    
    def display_frame(self, frame):
        """Отображение кадра."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_canvas.delete("all")
        self.video_canvas.create_image(320, 240, image=imgtk)
        self.video_canvas.imgtk = imgtk
    
    def update_stats(self):
        """Обновление статистики."""
        stats = self.data_collector.get_stats()
        total = stats['total']
        by_class = stats['by_class']
        classes_with_data = len(by_class)
        
        text = f"Образцов собрано: {total}\n"
        text += f"Готовых жестов: {classes_with_data}/{len(self.gesture_classes)}\n\n"
        
        if by_class:
            text += "По жестам:\n"
            for label, count in by_class.items():
                gesture_name = self.gesture_classes[int(label)]
                text += f"  {gesture_name}: {count}\n"
        
        self.stats_label.configure(text=text)
    
    def start_training(self):
        """Начало обучения модели."""
        # Проверяем наличие данных
        stats = self.data_collector.get_stats()
        if stats['total'] < 10:
            messagebox.showwarning(
                "Недостаточно данных",
                "Соберите хотя бы 10 образцов для обучения"
            )
            return
        
        # Запускаем обучение в отдельном потоке
        self.is_training = True
        self.train_model_btn.configure(state="disabled", text="⏳ Обучение...")
        
        training_thread = threading.Thread(target=self.train_model, daemon=True)
        training_thread.start()
    
    def train_model(self):
        """Обучение модели (выполняется в отдельном потоке)."""
        try:
            # Получаем данные
            samples, labels = self.data_collector.get_dataset()
            
            # Создаем dataset
            dataset = GestureDataset(samples, labels)
            
            epochs = int(self.epochs_entry.get())
            batch_size = int(self.batch_entry.get())
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Создаем модель
            from model.dynamic_gesture_model import DynamicGestureClassifier
            
            model = DynamicGestureClassifier(
                input_size=126,
                num_classes=len(self.gesture_classes)
            )
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Обучение
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for sequences, labels_batch in dataloader:
                    optimizer.zero_grad()
                    outputs = model(sequences)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # Обновляем прогресс
                progress = (epoch + 1) / epochs
                self.training_progress.set(progress)
                
                avg_loss = total_loss / len(dataloader)
                status_text = f"Эпоха {epoch+1}/{epochs}, Loss: {avg_loss:.4f}"
                self.training_status.configure(text=status_text)
                
                print(status_text)
            
            # Сохраняем модель
            self.model = DynamicGestureModelWrapper(
                model_path=None,
                input_size=126,
                num_classes=len(self.gesture_classes),
                gesture_classes=self.gesture_classes,
                use_dummy=False
            )
            self.model.model = model
            self.model.model.eval()
            
            # Сохраняем на диск
            os.makedirs('models', exist_ok=True)
            self.model.save('models/dynamic_gesture_model.pth')
            
            # Уведомление
            self.training_status.configure(text="✓ Обучение завершено!")
            self.tts.speak("Обучение завершено")
            messagebox.showinfo("Успех", "Модель обучена и сохранена!")
            
        except Exception as e:
            print(f"Ошибка обучения: {e}")
            messagebox.showerror("Ошибка", f"Ошибка обучения: {e}")
        finally:
            self.is_training = False
            self.train_model_btn.configure(state="normal", text="🚀 Обучить модель")
    
    def on_closing(self):
        """Обработчик закрытия."""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        if self.gesture_detector:
            self.gesture_detector.close()
        
        if self.tts:
            self.tts.stop()
        
        self.root.destroy()


def main():
    """Точка входа."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Training GUI для SignVoiceAI')
    parser.add_argument('--camera', type=int, default=0, help='Индекс камеры')
    
    args = parser.parse_args()
    
    root = ctk.CTk()
    app = TrainingGUI(root, camera_index=args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()


