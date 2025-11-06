"""
Упрощенный интерфейс для использования обученных жестов.

Использует DTW для надежного распознавания.
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
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.camera import Camera
from utils.gestures import GestureDetector
from utils.speech import TextToSpeech

# Импортируем recognizer из trainer
from gesture_trainer_simple import ImprovedGestureRecognizer

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class GestureRecognizerApp:
    """
    Приложение для распознавания жестов.
    """
    
    def __init__(self, root, camera_index=0):
        """Инициализация."""
        self.root = root
        self.root.title("🎯 Распознавание ваших жестов")
        self.root.geometry("1200x800")
        
        # Компоненты
        self.camera = Camera(camera_index=camera_index, width=640, height=480)
        self.gesture_detector = GestureDetector(max_num_hands=2, detect_both_hands=True)
        self.recognizer = ImprovedGestureRecognizer(sequence_length=40)
        self.tts = TextToSpeech(rate=150, volume=0.8)
        
        # Загружаем шаблоны
        if not os.path.exists('gesture_templates.pkl'):
            messagebox.showerror(
                "Ошибка",
                "Файл gesture_templates.pkl не найден!\n\n"
                "Сначала обучите жесты через:\n"
                "python gesture_trainer_simple.py"
            )
            self.root.destroy()
            return
        
        self.recognizer.load_templates('gesture_templates.pkl')
        
        # Состояние
        self.is_running = False
        self.video_thread = None
        self.last_gesture = None
        self.gesture_stable_count = 0
        self.stability_threshold = 3
        
        # История
        self.gesture_history = deque(maxlen=20)
        self.history_widgets = []
        
        # Настройки
        self.min_confidence = ctk.DoubleVar(value=0.6)
        self.stability_threshold_var = ctk.IntVar(value=3)
        
        # GUI
        self.create_gui()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_gui(self):
        """Создание GUI."""
        # Главный контейнер
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Левая панель
        left = ctk.CTkFrame(main, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Правая панель
        right = ctk.CTkFrame(main, fg_color="transparent", width=400)
        right.pack(side="right", fill="both", padx=(10, 0))
        right.pack_propagate(False)
        
        # Панели
        self.create_video_panel(left)
        self.create_control_panel(left)
        self.create_status_panel(left)
        self.create_history_panel(right)
        self.create_settings_panel(right)
    
    def create_video_panel(self, parent):
        """Панель видео."""
        video_frame = ctk.CTkFrame(parent, corner_radius=15)
        video_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        # Заголовок
        header = ctk.CTkFrame(video_frame, corner_radius=10, fg_color=("#2b2b2b", "#1a1a1a"))
        header.pack(fill="x", padx=15, pady=15)
        
        title = ctk.CTkLabel(
            header,
            text="📹 Распознавание жестов",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        title.pack(side="left", padx=20, pady=12)
        
        # Статус
        self.status_indicator = ctk.CTkLabel(
            header,
            text="⚫ Остановлено",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#9ca3af", "#9ca3af")
        )
        self.status_indicator.pack(side="right", padx=20, pady=12)
        
        # Canvas
        canvas_frame = ctk.CTkFrame(video_frame, corner_radius=10, fg_color="#000000")
        canvas_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.video_canvas = ctk.CTkCanvas(canvas_frame, bg="#000000", highlightthickness=0)
        self.video_canvas.pack(fill="both", expand=True, padx=3, pady=3)
        
        # Инфо
        self.info_label = ctk.CTkLabel(
            canvas_frame,
            text="Нажмите 'Старт' для начала",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        self.info_label.place(relx=0.5, rely=0.5, anchor="center")
    
    def create_control_panel(self, parent):
        """Панель управления."""
        control = ctk.CTkFrame(parent, corner_radius=15, height=90)
        control.pack(fill="x", pady=(0, 15))
        control.pack_propagate(False)
        
        btn_container = ctk.CTkFrame(control, fg_color="transparent")
        btn_container.pack(expand=True, fill="x", padx=20)
        
        # Старт/Стоп
        self.start_btn = ctk.CTkButton(
            btn_container,
            text="▶ Старт",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=50,
            fg_color=("#1976d2", "#1976d2"),
            command=self.toggle_recognition
        )
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        # Повтор
        self.repeat_btn = ctk.CTkButton(
            btn_container,
            text="🔊 Повторить",
            font=ctk.CTkFont(size=14),
            height=50,
            fg_color=("#4a5568", "#4a5568"),
            state="disabled",
            command=self.repeat_gesture
        )
        self.repeat_btn.pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        # Очистить
        clear_btn = ctk.CTkButton(
            btn_container,
            text="🗑️ Очистить",
            font=ctk.CTkFont(size=14),
            height=50,
            fg_color=("#4a5568", "#4a5568"),
            command=self.clear_history
        )
        clear_btn.pack(side="left", expand=True, fill="x")
    
    def create_status_panel(self, parent):
        """Панель статуса."""
        status_frame = ctk.CTkFrame(parent, corner_radius=15, height=120)
        status_frame.pack(fill="x")
        status_frame.pack_propagate(False)
        
        status_grid = ctk.CTkFrame(status_frame, fg_color="transparent")
        status_grid.pack(expand=True, fill="both", padx=25, pady=20)
        
        status_grid.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Текущий жест
        gesture_container = ctk.CTkFrame(status_grid, corner_radius=10, 
                                        fg_color=("#2b2b2b", "#1a1a1a"))
        gesture_container.grid(row=0, column=0, sticky="nsew", padx=5)
        
        ctk.CTkLabel(gesture_container, text="Жест", 
                    font=ctk.CTkFont(size=11), 
                    text_color=("#9ca3af", "#9ca3af")).pack(pady=(10, 2))
        
        self.gesture_label = ctk.CTkLabel(
            gesture_container,
            text="—",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        self.gesture_label.pack(pady=(0, 10))
        
        # Уверенность
        conf_container = ctk.CTkFrame(status_grid, corner_radius=10,
                                     fg_color=("#2b2b2b", "#1a1a1a"))
        conf_container.grid(row=0, column=1, sticky="nsew", padx=5)
        
        ctk.CTkLabel(conf_container, text="Уверенность",
                    font=ctk.CTkFont(size=11),
                    text_color=("#9ca3af", "#9ca3af")).pack(pady=(10, 2))
        
        self.confidence_label = ctk.CTkLabel(
            conf_container,
            text="—",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=("#ffffff", "#ffffff")
        )
        self.confidence_label.pack(pady=(0, 10))
        
        # Стабильность
        stable_container = ctk.CTkFrame(status_grid, corner_radius=10,
                                       fg_color=("#2b2b2b", "#1a1a1a"))
        stable_container.grid(row=0, column=2, sticky="nsew", padx=5)
        
        ctk.CTkLabel(stable_container, text="Статус",
                    font=ctk.CTkFont(size=11),
                    text_color=("#9ca3af", "#9ca3af")).pack(pady=(10, 2))
        
        self.stability_label = ctk.CTkLabel(
            stable_container,
            text="—",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#9ca3af", "#9ca3af")
        )
        self.stability_label.pack(pady=(0, 10))
    
    def create_history_panel(self, parent):
        """Панель истории."""
        history_frame = ctk.CTkFrame(parent, corner_radius=15)
        history_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        header = ctk.CTkLabel(
            history_frame,
            text="📜 История жестов",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        self.history_scrollable = ctk.CTkScrollableFrame(
            history_frame,
            corner_radius=10,
            fg_color=("#2b2b2b", "#1a1a1a")
        )
        self.history_scrollable.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.history_placeholder = ctk.CTkLabel(
            self.history_scrollable,
            text="История пуста",
            font=ctk.CTkFont(size=12),
            text_color=("#6b7280", "#6b7280")
        )
        self.history_placeholder.pack(pady=20)
    
    def create_settings_panel(self, parent):
        """Панель настроек."""
        settings_frame = ctk.CTkFrame(parent, corner_radius=15)
        settings_frame.pack(fill="x")
        
        header = ctk.CTkLabel(
            settings_frame,
            text="⚙️ Настройки",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        settings_container = ctk.CTkFrame(settings_frame, fg_color="transparent")
        settings_container.pack(fill="x", padx=15, pady=(0, 15))
        
        # Стабильность
        ctk.CTkLabel(
            settings_container,
            text="Порог стабильности:",
            font=ctk.CTkFont(size=12),
            text_color=("#9ca3af", "#9ca3af")
        ).pack(anchor="w", pady=(5, 2))
        
        stability_slider = ctk.CTkSlider(
            settings_container,
            from_=1,
            to=10,
            number_of_steps=9,
            variable=self.stability_threshold_var,
            command=self.update_stability
        )
        stability_slider.pack(fill="x", pady=(0, 15))
        
        # Уверенность
        ctk.CTkLabel(
            settings_container,
            text="Мин. уверенность:",
            font=ctk.CTkFont(size=12),
            text_color=("#9ca3af", "#9ca3af")
        ).pack(anchor="w", pady=(0, 2))
        
        conf_slider = ctk.CTkSlider(
            settings_container,
            from_=0.3,
            to=1.0,
            variable=self.min_confidence
        )
        conf_slider.pack(fill="x", pady=(0, 15))
        
        # Инфо о жестах
        info_frame = ctk.CTkFrame(settings_container, corner_radius=10,
                                 fg_color=("#2b2b2b", "#1a1a1a"))
        info_frame.pack(fill="x")
        
        ctk.CTkLabel(
            info_frame,
            text="Доступные жесты:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#9ca3af", "#9ca3af")
        ).pack(anchor="w", padx=15, pady=(10, 5))
        
        gestures_text = "\n".join([f"• {name}" for name in self.recognizer.gesture_names])
        if not gestures_text:
            gestures_text = "Нет жестов"
        
        ctk.CTkLabel(
            info_frame,
            text=gestures_text,
            font=ctk.CTkFont(size=10),
            text_color=("#6b7280", "#6b7280"),
            justify="left"
        ).pack(anchor="w", padx=15, pady=(0, 10))
    
    def toggle_recognition(self):
        """Переключение распознавания."""
        if not self.is_running:
            if self.camera.open():
                self.is_running = True
                self.start_btn.configure(text="⏸ Стоп", fg_color="#c92a2a")
                self.status_indicator.configure(text="🟢 Работает", text_color="#51cf66")
                self.repeat_btn.configure(state="normal")
                self.info_label.place_forget()
                
                self.video_thread = threading.Thread(target=self.process_video, daemon=True)
                self.video_thread.start()
            else:
                messagebox.showerror("Ошибка", "Не удалось открыть камеру")
        else:
            self.is_running = False
            self.start_btn.configure(text="▶ Старт", fg_color="#1976d2")
            self.status_indicator.configure(text="⚫ Остановлено", text_color="#9ca3af")
            
            if self.camera:
                self.camera.release()
            
            self.info_label.place(relx=0.5, rely=0.5, anchor="center")
    
    def process_video(self):
        """Обработка видео."""
        current_gesture = None
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Обнаружение рук
            hands_data, annotated_frame = self.gesture_detector.detect(frame)
            
            # Добавляем в recognizer
            self.recognizer.add_frame(hands_data)
            
            # Распознавание
            if self.recognizer.is_ready():
                sequence = self.recognizer.get_sequence()
                gesture, confidence = self.recognizer.recognize(sequence)
                
                if gesture and confidence >= self.min_confidence.get():
                    # Проверка стабильности
                    if gesture == current_gesture:
                        self.gesture_stable_count += 1
                    else:
                        current_gesture = gesture
                        self.gesture_stable_count = 1
                    
                    # Обновляем GUI
                    self.gesture_label.configure(text=gesture)
                    self.confidence_label.configure(text=f"{confidence:.1%}")
                    
                    # Статус стабильности
                    threshold = self.stability_threshold_var.get()
                    if self.gesture_stable_count >= threshold:
                        self.stability_label.configure(text="✓ Стабильно", text_color="#51cf66")
                        
                        # Озвучиваем если новый жест
                        if gesture != self.last_gesture:
                            self.last_gesture = gesture
                            self.add_to_history(gesture, confidence)
                            self.tts.speak(gesture)
                    else:
                        progress = f"{self.gesture_stable_count}/{threshold}"
                        self.stability_label.configure(text=f"⏳ {progress}", text_color="#ffd43b")
                else:
                    current_gesture = None
                    self.gesture_stable_count = 0
                    self.gesture_label.configure(text="—")
                    self.confidence_label.configure(text="—")
                    self.stability_label.configure(text="Покажите жест", text_color="#9ca3af")
            
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
    
    def add_to_history(self, gesture, confidence):
        """Добавление в историю."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{timestamp} - {gesture} ({confidence:.1%})"
        
        self.gesture_history.append(entry)
        
        if self.history_placeholder.winfo_exists():
            self.history_placeholder.destroy()
        
        entry_frame = ctk.CTkFrame(
            self.history_scrollable,
            corner_radius=8,
            fg_color=("#383838", "#252525"),
            height=40
        )
        entry_frame.pack(fill="x", pady=2)
        entry_frame.pack_propagate(False)
        
        entry_label = ctk.CTkLabel(
            entry_frame,
            text=entry,
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color=("#e5e7eb", "#e5e7eb")
        )
        entry_label.pack(pady=10, padx=15, anchor="w")
        
        self.history_widgets.insert(0, entry_frame)
        
        if len(self.history_widgets) > 20:
            old = self.history_widgets.pop()
            old.destroy()
    
    def clear_history(self):
        """Очистка истории."""
        self.gesture_history.clear()
        
        for widget in self.history_widgets:
            widget.destroy()
        self.history_widgets.clear()
        
        self.history_placeholder = ctk.CTkLabel(
            self.history_scrollable,
            text="История пуста",
            font=ctk.CTkFont(size=12),
            text_color=("#6b7280", "#6b7280")
        )
        self.history_placeholder.pack(pady=20)
    
    def repeat_gesture(self):
        """Повтор жеста."""
        if self.last_gesture:
            self.tts.speak(self.last_gesture, force=True)
    
    def update_stability(self, value):
        """Обновление порога стабильности."""
        self.stability_threshold = int(float(value))
    
    def on_closing(self):
        """Закрытие."""
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
    
    parser = argparse.ArgumentParser(description='Распознавание жестов')
    parser.add_argument('--camera', type=int, default=0, help='Индекс камеры')
    
    args = parser.parse_args()
    
    root = ctk.CTk()
    app = GestureRecognizerApp(root, camera_index=args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()


