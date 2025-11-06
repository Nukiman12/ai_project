"""
Упрощенная и надежная система обучения жестам.

Особенности:
- Простое добавление своих жестов
- Улучшенная обработка данных
- Более устойчивое распознавание
- Автоматическая нормализация
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
from tkinter import messagebox, simpledialog
import pickle
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.camera import Camera
from utils.gestures import GestureDetector

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ImprovedGestureRecognizer:
    """
    Улучшенный распознаватель жестов с DTW (Dynamic Time Warping).
    Более устойчивый к вариациям скорости и длительности.
    """
    
    def __init__(self, sequence_length=40, smoothing=True):
        """
        Инициализация.
        
        Args:
            sequence_length: Длина последовательности
            smoothing: Применять сглаживание
        """
        self.sequence_length = sequence_length
        self.smoothing = smoothing
        self.sequence = deque(maxlen=sequence_length)
        self.gesture_templates = {}  # {имя_жеста: [шаблоны]}
        self.gesture_names = []
        
    def add_frame(self, hands_data):
        """Добавляет кадр в последовательность."""
        # Извлекаем координаты
        left = hands_data.get('left')
        right = hands_data.get('right')
        
        # Объединяем обе руки
        if left is not None and right is not None:
            features = np.concatenate([left, right])
        elif left is not None:
            features = np.concatenate([left, np.zeros(63)])
        elif right is not None:
            features = np.concatenate([np.zeros(63), right])
        else:
            features = np.zeros(126)
        
        # Нормализуем
        features = self._normalize_features(features)
        
        self.sequence.append(features)
    
    def _normalize_features(self, features):
        """Нормализация признаков."""
        # Разделяем на x, y, z для каждой руки
        features = features.reshape(-1, 3)
        
        # Нормализуем относительно первой точки (запястье)
        for i in range(0, len(features), 21):
            if i + 21 <= len(features):
                wrist = features[i].copy()
                for j in range(i, i + 21):
                    features[j] = features[j] - wrist
        
        return features.flatten()
    
    def get_sequence(self):
        """Получает текущую последовательность."""
        return np.array(list(self.sequence))
    
    def is_ready(self):
        """Проверяет готовность последовательности."""
        return len(self.sequence) >= self.sequence_length
    
    def add_gesture_template(self, gesture_name, sequence):
        """
        Добавляет шаблон жеста.
        
        Args:
            gesture_name: Название жеста
            sequence: Последовательность кадров
        """
        if gesture_name not in self.gesture_templates:
            self.gesture_templates[gesture_name] = []
            self.gesture_names.append(gesture_name)
        
        # Сглаживаем и нормализуем
        if self.smoothing:
            sequence = self._smooth_sequence(sequence)
        
        self.gesture_templates[gesture_name].append(sequence)
    
    def _smooth_sequence(self, sequence, window=3):
        """Сглаживание последовательности."""
        if len(sequence) < window:
            return sequence
        
        smoothed = np.copy(sequence)
        for i in range(window // 2, len(sequence) - window // 2):
            smoothed[i] = np.mean(sequence[i - window // 2:i + window // 2 + 1], axis=0)
        
        return smoothed
    
    def recognize(self, sequence):
        """
        Распознает жест используя DTW.
        
        Args:
            sequence: Текущая последовательность
            
        Returns:
            (gesture_name, confidence)
        """
        if not self.gesture_templates:
            return None, 0.0
        
        min_distance = float('inf')
        best_gesture = None
        
        # Сравниваем с каждым шаблоном
        for gesture_name, templates in self.gesture_templates.items():
            for template in templates:
                distance = self._dtw_distance(sequence, template)
                if distance < min_distance:
                    min_distance = distance
                    best_gesture = gesture_name
        
        # Преобразуем distance в confidence (0-1)
        # Меньше distance = больше confidence
        confidence = 1.0 / (1.0 + min_distance / 100.0)
        
        return best_gesture, confidence
    
    def _dtw_distance(self, seq1, seq2):
        """
        Dynamic Time Warping - вычисляет расстояние между последовательностями.
        Устойчиво к различиям в скорости выполнения.
        """
        n, m = len(seq1), len(seq2)
        
        # Создаем матрицу расстояний
        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0
        
        # Заполняем матрицу
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # вставка
                    dtw_matrix[i, j - 1],      # удаление
                    dtw_matrix[i - 1, j - 1]   # совпадение
                )
        
        return dtw_matrix[n, m]
    
    def save_templates(self, filepath):
        """Сохраняет шаблоны жестов."""
        data = {
            'templates': self.gesture_templates,
            'names': self.gesture_names,
            'sequence_length': self.sequence_length
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Шаблоны сохранены: {filepath}")
    
    def load_templates(self, filepath):
        """Загружает шаблоны жестов."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.gesture_templates = data['templates']
            self.gesture_names = data['names']
            self.sequence_length = data.get('sequence_length', 40)
            
            print(f"✓ Загружено шаблонов: {len(self.gesture_templates)}")
            return True
        except Exception as e:
            print(f"✗ Ошибка загрузки: {e}")
            return False
    
    def clear(self):
        """Очищает последовательность."""
        self.sequence.clear()
    
    def get_stats(self):
        """Получает статистику."""
        total_templates = sum(len(templates) for templates in self.gesture_templates.values())
        return {
            'gestures': len(self.gesture_names),
            'templates': total_templates,
            'templates_per_gesture': {
                name: len(templates) 
                for name, templates in self.gesture_templates.items()
            }
        }


class SimpleGestureTrainer:
    """
    Упрощенный интерфейс для обучения жестам.
    """
    
    def __init__(self, root, camera_index=0):
        """Инициализация."""
        self.root = root
        self.root.title("🎯 Обучение жестам - Упрощенная версия")
        self.root.geometry("1400x900")
        
        # Компоненты
        self.camera = Camera(camera_index=camera_index, width=640, height=480)
        self.gesture_detector = GestureDetector(
            max_num_hands=2,
            detect_both_hands=True
        )
        self.recognizer = ImprovedGestureRecognizer(sequence_length=40)
        
        # Состояние
        self.is_running = False
        self.is_recording = False
        self.video_thread = None
        self.current_gesture_name = None
        self.recorded_samples = []
        
        # Режим
        self.mode = 'record'  # 'record' или 'test'
        
        # GUI
        self.create_gui()
        
        # Загружаем шаблоны если есть
        if os.path.exists('gesture_templates.pkl'):
            self.recognizer.load_templates('gesture_templates.pkl')
            self.update_gesture_list()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_gui(self):
        """Создание GUI."""
        # Главный контейнер
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Левая панель (видео)
        left = ctk.CTkFrame(main, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Правая панель (управление)
        right = ctk.CTkFrame(main, fg_color="transparent", width=450)
        right.pack(side="right", fill="both", padx=(10, 0))
        right.pack_propagate(False)
        
        # Видео
        self.create_video_panel(left)
        
        # Управление
        self.create_control_panel(right)
        self.create_gesture_list_panel(right)
        self.create_status_panel(right)
    
    def create_video_panel(self, parent):
        """Панель видео."""
        video_frame = ctk.CTkFrame(parent, corner_radius=15)
        video_frame.pack(fill="both", expand=True)
        
        # Заголовок
        header = ctk.CTkFrame(video_frame, corner_radius=10, fg_color=("#2b2b2b", "#1a1a1a"))
        header.pack(fill="x", padx=15, pady=15)
        
        title = ctk.CTkLabel(
            header,
            text="📹 Видео",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        title.pack(side="left", padx=20, pady=12)
        
        # Статус
        self.status_label = ctk.CTkLabel(
            header,
            text="⚫ Остановлено",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#9ca3af", "#9ca3af")
        )
        self.status_label.pack(side="right", padx=20, pady=12)
        
        # Canvas
        canvas_frame = ctk.CTkFrame(video_frame, corner_radius=10, fg_color="#000000")
        canvas_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.video_canvas = ctk.CTkCanvas(canvas_frame, bg="#000000", highlightthickness=0)
        self.video_canvas.pack(fill="both", expand=True, padx=3, pady=3)
        
        # Инфо
        self.info_label = ctk.CTkLabel(
            canvas_frame,
            text="Нажмите 'Старт камеры' для начала",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        self.info_label.place(relx=0.5, rely=0.5, anchor="center")
    
    def create_control_panel(self, parent):
        """Панель управления."""
        control = ctk.CTkFrame(parent, corner_radius=15)
        control.pack(fill="x", pady=(0, 15))
        
        header = ctk.CTkLabel(
            control,
            text="🎮 Управление",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Кнопки
        btn_frame = ctk.CTkFrame(control, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        # Старт камеры
        self.start_btn = ctk.CTkButton(
            btn_frame,
            text="▶ Старт камеры",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=50,
            fg_color=("#1976d2", "#1976d2"),
            command=self.toggle_camera
        )
        self.start_btn.pack(fill="x", pady=(0, 10))
        
        # Новый жест
        self.new_gesture_btn = ctk.CTkButton(
            btn_frame,
            text="➕ Добавить новый жест",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=50,
            fg_color=("#2e7d32", "#2e7d32"),
            state="disabled",
            command=self.add_new_gesture
        )
        self.new_gesture_btn.pack(fill="x", pady=(0, 10))
        
        # Записать образец
        self.record_btn = ctk.CTkButton(
            btn_frame,
            text="🔴 Записать образец",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=50,
            fg_color=("#c92a2a", "#c92a2a"),
            state="disabled",
            command=self.toggle_recording
        )
        self.record_btn.pack(fill="x", pady=(0, 10))
        
        # Режимы
        mode_frame = ctk.CTkFrame(btn_frame, fg_color="transparent")
        mode_frame.pack(fill="x")
        
        self.record_mode_btn = ctk.CTkButton(
            mode_frame,
            text="📝 Запись",
            width=100,
            fg_color=("#1976d2", "#1976d2"),
            command=lambda: self.set_mode('record')
        )
        self.record_mode_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        self.test_mode_btn = ctk.CTkButton(
            mode_frame,
            text="🎯 Тест",
            width=100,
            fg_color=("#4a5568", "#4a5568"),
            command=lambda: self.set_mode('test')
        )
        self.test_mode_btn.pack(side="left", expand=True, fill="x", padx=(5, 0))
    
    def create_gesture_list_panel(self, parent):
        """Панель списка жестов."""
        list_frame = ctk.CTkFrame(parent, corner_radius=15)
        list_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        header = ctk.CTkLabel(
            list_frame,
            text="📋 Ваши жесты",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Scrollable список
        self.gesture_scrollable = ctk.CTkScrollableFrame(
            list_frame,
            corner_radius=10,
            fg_color=("#2b2b2b", "#1a1a1a")
        )
        self.gesture_scrollable.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Placeholder
        self.gesture_placeholder = ctk.CTkLabel(
            self.gesture_scrollable,
            text="Добавьте свои жесты\n\nНажмите '➕ Добавить новый жест'",
            font=ctk.CTkFont(size=12),
            text_color=("#6b7280", "#6b7280")
        )
        self.gesture_placeholder.pack(pady=30)
        
        self.gesture_buttons = []
    
    def create_status_panel(self, parent):
        """Панель статуса."""
        status_frame = ctk.CTkFrame(parent, corner_radius=15)
        status_frame.pack(fill="x")
        
        header = ctk.CTkLabel(
            status_frame,
            text="📊 Статистика",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        self.stats_label = ctk.CTkLabel(
            status_frame,
            text="Жестов: 0\nОбразцов: 0",
            font=ctk.CTkFont(size=12),
            justify="left"
        )
        self.stats_label.pack(padx=15, pady=(0, 15), anchor="w")
        
        # Распознанный жест
        self.recognized_label = ctk.CTkLabel(
            status_frame,
            text="Распознан: —\nУверенность: —",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#42a5f5", "#42a5f5"),
            justify="left"
        )
        self.recognized_label.pack(padx=15, pady=(0, 15), anchor="w")
    
    def toggle_camera(self):
        """Переключение камеры."""
        if not self.is_running:
            if self.camera.open():
                self.is_running = True
                self.start_btn.configure(text="⏸ Стоп камеры", fg_color="#c92a2a")
                self.new_gesture_btn.configure(state="normal")
                self.info_label.place_forget()
                
                self.video_thread = threading.Thread(target=self.process_video, daemon=True)
                self.video_thread.start()
        else:
            self.is_running = False
            self.start_btn.configure(text="▶ Старт камеры", fg_color="#1976d2")
            self.new_gesture_btn.configure(state="disabled")
            self.record_btn.configure(state="disabled")
            
            if self.camera:
                self.camera.release()
            
            self.info_label.place(relx=0.5, rely=0.5, anchor="center")
    
    def add_new_gesture(self):
        """Добавление нового жеста."""
        name = simpledialog.askstring(
            "Новый жест",
            "Введите название жеста:\n(например: Махать, Круг, Привет)",
            parent=self.root
        )
        
        if name:
            name = name.strip()
            if name:
                self.current_gesture_name = name
                if name not in self.recognizer.gesture_names:
                    self.recognizer.gesture_names.append(name)
                    self.update_gesture_list()
                
                self.record_btn.configure(state="normal")
                messagebox.showinfo(
                    "Жест добавлен",
                    f"Жест '{name}' добавлен!\n\nТеперь нажмите '🔴 Записать образец'\nи выполните движение 5-10 раз."
                )
    
    def toggle_recording(self):
        """Переключение записи."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Начало записи."""
        if not self.current_gesture_name:
            messagebox.showwarning("Внимание", "Сначала добавьте жест!")
            return
        
        self.is_recording = True
        self.record_btn.configure(text="⏹ Стоп запись", fg_color="#ff6b6b")
        self.status_label.configure(
            text=f"🔴 Запись: {self.current_gesture_name}",
            text_color=("#ff6b6b", "#ff6b6b")
        )
        self.recognizer.clear()
        self.recorded_samples = []
        
        print(f"🔴 Запись жеста: {self.current_gesture_name}")
    
    def stop_recording(self):
        """Остановка записи."""
        self.is_recording = False
        self.record_btn.configure(text="🔴 Записать образец", fg_color="#c92a2a")
        self.status_label.configure(
            text="⚫ Готов к записи",
            text_color=("#9ca3af", "#9ca3af")
        )
        
        # Сохраняем образцы
        count = len(self.recorded_samples)
        for sample in self.recorded_samples:
            self.recognizer.add_gesture_template(self.current_gesture_name, sample)
        
        # Сохраняем в файл
        self.recognizer.save_templates('gesture_templates.pkl')
        
        print(f"✓ Сохранено {count} образцов для '{self.current_gesture_name}'")
        messagebox.showinfo(
            "Готово!",
            f"Сохранено {count} образцов\n\nМожете записать еще образцы\nили добавить другой жест."
        )
        
        self.update_stats()
    
    def set_mode(self, mode):
        """Установка режима."""
        self.mode = mode
        
        if mode == 'record':
            self.record_mode_btn.configure(fg_color=("#1976d2", "#1976d2"))
            self.test_mode_btn.configure(fg_color=("#4a5568", "#4a5568"))
            self.record_btn.configure(state="normal" if self.current_gesture_name else "disabled")
        else:
            self.record_mode_btn.configure(fg_color=("#4a5568", "#4a5568"))
            self.test_mode_btn.configure(fg_color=("#1976d2", "#1976d2"))
            self.record_btn.configure(state="disabled")
    
    def process_video(self):
        """Обработка видео."""
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Обнаружение рук
            hands_data, annotated_frame = self.gesture_detector.detect(frame)
            
            # Добавляем в recognizer
            self.recognizer.add_frame(hands_data)
            
            # Если записываем
            if self.is_recording and self.recognizer.is_ready():
                sequence = self.recognizer.get_sequence()
                self.recorded_samples.append(sequence.copy())
                self.recognizer.clear()  # Очищаем для следующего образца
            
            # Если тестируем
            if self.mode == 'test' and self.recognizer.is_ready():
                sequence = self.recognizer.get_sequence()
                gesture, confidence = self.recognizer.recognize(sequence)
                
                if gesture and confidence > 0.5:
                    self.recognized_label.configure(
                        text=f"Распознан: {gesture}\nУверенность: {confidence:.1%}"
                    )
            
            # Отображение
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
    
    def update_gesture_list(self):
        """Обновление списка жестов."""
        # Удаляем placeholder
        if self.gesture_placeholder.winfo_exists():
            self.gesture_placeholder.destroy()
        
        # Очищаем старые кнопки
        for btn in self.gesture_buttons:
            btn.destroy()
        self.gesture_buttons.clear()
        
        # Создаем кнопки
        stats = self.recognizer.get_stats()
        templates_per = stats.get('templates_per_gesture', {})
        
        for name in self.recognizer.gesture_names:
            count = templates_per.get(name, 0)
            text = f"{name} ({count} образцов)"
            
            btn = ctk.CTkButton(
                self.gesture_scrollable,
                text=text,
                font=ctk.CTkFont(size=12),
                height=40,
                anchor="w",
                command=lambda n=name: self.select_gesture(n)
            )
            btn.pack(fill="x", pady=2)
            self.gesture_buttons.append(btn)
    
    def select_gesture(self, name):
        """Выбор жеста."""
        self.current_gesture_name = name
        self.record_btn.configure(state="normal")
        print(f"Выбран жест: {name}")
    
    def update_stats(self):
        """Обновление статистики."""
        stats = self.recognizer.get_stats()
        text = f"Жестов: {stats['gestures']}\n"
        text += f"Образцов: {stats['templates']}"
        self.stats_label.configure(text=text)
        
        self.update_gesture_list()
    
    def on_closing(self):
        """Закрытие."""
        self.is_running = False
        if self.camera:
            self.camera.release()
        if self.gesture_detector:
            self.gesture_detector.close()
        self.root.destroy()


def main():
    """Точка входа."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Упрощенное обучение жестам')
    parser.add_argument('--camera', type=int, default=0, help='Индекс камеры')
    
    args = parser.parse_args()
    
    root = ctk.CTk()
    app = SimpleGestureTrainer(root, camera_index=args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()


