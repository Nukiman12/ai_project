"""
SignVoiceAI Advanced GUI - Расширенный интерфейс.

Поддержка:
- Две руки одновременно
- Динамические жесты (движения)
- Статические жесты (позы)
- Переключение между режимами
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

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.camera import Camera
from utils.gestures import GestureDetector, DynamicGestureRecognizer
from utils.speech import TextToSpeech
from model.gesture_model import GestureModelWrapper, GESTURE_CLASSES
from model.dynamic_gesture_model import DynamicGestureModelWrapper, DYNAMIC_GESTURE_CLASSES

# Настройка темы
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AdvancedSignVoiceGUI:
    """
    Расширенный GUI с поддержкой двух рук и динамических жестов.
    """
    
    def __init__(self, root, static_model_path=None, dynamic_model_path=None, camera_index=0):
        """Инициализация расширенного интерфейса."""
        self.root = root
        self.root.title("SignVoiceAI Advanced - 2 руки + Движения")
        self.root.geometry("1600x950")
        
        # Компоненты
        self.camera_index = camera_index
        self.camera = None
        self.gesture_detector = None
        self.dynamic_recognizer = None
        self.static_model = None
        self.dynamic_model = None
        self.tts = None
        self.is_running = False
        self.video_thread = None
        
        # Режим распознавания
        self.recognition_mode = ctk.StringVar(value='dynamic')  # 'static', 'dynamic', 'both'
        
        # Состояние
        self.current_gesture = None
        self.last_gesture = None
        self.gesture_confidence = 0.0
        self.gesture_stable_count = 0
        self.stability_threshold = 5
        
        # История
        self.gesture_history = deque(maxlen=20)
        self.confidence_history = deque(maxlen=50)
        
        # Данные рук
        self.hands_status = {'left': False, 'right': False}
        
        # Настройки
        self.settings = {
            'stability_threshold': ctk.IntVar(value=5),
            'min_confidence': ctk.DoubleVar(value=0.5),
            'speech_rate': ctk.IntVar(value=150),
            'speech_volume': ctk.DoubleVar(value=0.8),
            'sequence_length': ctk.IntVar(value=30)
        }
        
        # Инициализация
        self.init_components(static_model_path, dynamic_model_path)
        self.create_gui()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def init_components(self, static_model_path, dynamic_model_path):
        """Инициализация компонентов."""
        print("=" * 60)
        print("Инициализация Advanced GUI...")
        print("=" * 60)
        
        # Статическая модель
        print("[1/5] Загрузка статической модели...")
        self.static_model = GestureModelWrapper(model_path=static_model_path, use_dummy=True)
        
        # Динамическая модель
        print("[2/5] Загрузка динамической модели...")
        self.dynamic_model = DynamicGestureModelWrapper(
            model_path=dynamic_model_path,
            input_size=126,
            num_classes=len(DYNAMIC_GESTURE_CLASSES),
            gesture_classes=DYNAMIC_GESTURE_CLASSES,
            use_dummy=True
        )
        
        # Детектор жестов (2 руки)
        print("[3/5] Инициализация детектора жестов...")
        self.gesture_detector = GestureDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2,
            detect_both_hands=True
        )
        
        # Динамический распознаватель
        print("[4/5] Инициализация распознавателя движений...")
        self.dynamic_recognizer = DynamicGestureRecognizer(
            sequence_length=self.settings['sequence_length'].get(),
            hands_mode='both'
        )
        
        # TTS
        print("[5/5] Инициализация синтезатора речи...")
        self.tts = TextToSpeech(
            rate=self.settings['speech_rate'].get(),
            volume=self.settings['speech_volume'].get()
        )
        
        # Камера
        self.camera = Camera(camera_index=self.camera_index, width=640, height=480)
        
        print("=" * 60)
        print("✓ Инициализация завершена!")
        print("=" * 60)
    
    def create_gui(self):
        """Создание GUI."""
        # Главный контейнер
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Левая панель
        left_panel = ctk.CTkFrame(main_container, fg_color="transparent")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Правая панель
        right_panel = ctk.CTkFrame(main_container, fg_color="transparent", width=450)
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Создание элементов
        self.create_video_panel(left_panel)
        self.create_control_panel(left_panel)
        self.create_hands_status_panel(left_panel)
        self.create_gesture_status_panel(left_panel)
        self.create_mode_panel(right_panel)
        self.create_history_panel(right_panel)
        self.create_chart_panel(right_panel)
        self.create_settings_panel(right_panel)
    
    def create_video_panel(self, parent):
        """Панель видео."""
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
        
        # Индикатор статуса
        self.status_frame = ctk.CTkFrame(header_frame, corner_radius=20, 
                                         fg_color=("#3d3d3d", "#2d2d2d"))
        self.status_frame.pack(side="right", padx=20, pady=8)
        
        self.status_indicator = ctk.CTkLabel(
            self.status_frame,
            text="⚫ Остановлено",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#ff6b6b", "#ff6b6b")
        )
        self.status_indicator.pack(padx=15, pady=5)
        
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
            text="Нажмите 'Старт' для начала работы",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        self.info_overlay.place(relx=0.5, rely=0.5, anchor="center")
    
    def create_control_panel(self, parent):
        """Панель управления."""
        control_frame = ctk.CTkFrame(parent, corner_radius=15, height=90)
        control_frame.pack(fill="x", pady=(0, 15))
        control_frame.pack_propagate(False)
        
        button_container = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_container.pack(expand=True, fill="x", padx=20)
        
        # Кнопка Старт/Стоп
        self.start_button = ctk.CTkButton(
            button_container,
            text="▶  Старт",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=50,
            corner_radius=10,
            fg_color=("#1976d2", "#1976d2"),
            hover_color=("#1e88e5", "#42a5f5"),
            command=self.toggle_recognition
        )
        self.start_button.pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        # Кнопка Повтор
        self.repeat_button = ctk.CTkButton(
            button_container,
            text="🔊 Повторить",
            font=ctk.CTkFont(size=14),
            height=50,
            corner_radius=10,
            fg_color=("#4a5568", "#4a5568"),
            hover_color=("#5a657a", "#5a657a"),
            state="disabled",
            command=self.repeat_gesture
        )
        self.repeat_button.pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        # Кнопка Очистить
        clear_button = ctk.CTkButton(
            button_container,
            text="🗑️ Очистить",
            font=ctk.CTkFont(size=14),
            height=50,
            corner_radius=10,
            fg_color=("#4a5568", "#4a5568"),
            hover_color=("#5a657a", "#5a657a"),
            command=self.clear_history
        )
        clear_button.pack(side="left", expand=True, fill="x")
    
    def create_hands_status_panel(self, parent):
        """Панель статуса рук."""
        hands_frame = ctk.CTkFrame(parent, corner_radius=15, height=90)
        hands_frame.pack(fill="x", pady=(0, 15))
        hands_frame.pack_propagate(False)
        
        title = ctk.CTkLabel(
            hands_frame,
            text="🖐️ Статус рук",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=(10, 5))
        
        # Статус рук
        status_container = ctk.CTkFrame(hands_frame, fg_color="transparent")
        status_container.pack(expand=True, fill="x", padx=20)
        
        # Левая рука
        left_frame = ctk.CTkFrame(status_container, corner_radius=10, 
                                  fg_color=("#2b2b2b", "#1a1a1a"))
        left_frame.pack(side="left", expand=True, fill="both", padx=(0, 5))
        
        self.left_hand_label = ctk.CTkLabel(
            left_frame,
            text="Левая ❌",
            font=ctk.CTkFont(size=13),
            text_color=("#ff6b6b", "#ff6b6b")
        )
        self.left_hand_label.pack(pady=8)
        
        # Правая рука
        right_frame = ctk.CTkFrame(status_container, corner_radius=10,
                                   fg_color=("#2b2b2b", "#1a1a1a"))
        right_frame.pack(side="right", expand=True, fill="both", padx=(5, 0))
        
        self.right_hand_label = ctk.CTkLabel(
            right_frame,
            text="Правая ❌",
            font=ctk.CTkFont(size=13),
            text_color=("#ff6b6b", "#ff6b6b")
        )
        self.right_hand_label.pack(pady=8)
    
    def create_gesture_status_panel(self, parent):
        """Панель статуса жеста."""
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
        
        ctk.CTkLabel(
            gesture_container,
            text="Текущий жест",
            font=ctk.CTkFont(size=11),
            text_color=("#9ca3af", "#9ca3af")
        ).pack(pady=(10, 2))
        
        self.gesture_value = ctk.CTkLabel(
            gesture_container,
            text="—",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        self.gesture_value.pack(pady=(0, 10))
        
        # Уверенность
        conf_container = ctk.CTkFrame(status_grid, corner_radius=10, 
                                     fg_color=("#2b2b2b", "#1a1a1a"))
        conf_container.grid(row=0, column=1, sticky="nsew", padx=5)
        
        ctk.CTkLabel(
            conf_container,
            text="Уверенность",
            font=ctk.CTkFont(size=11),
            text_color=("#9ca3af", "#9ca3af")
        ).pack(pady=(10, 2))
        
        self.confidence_value = ctk.CTkLabel(
            conf_container,
            text="—",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#ffffff", "#ffffff")
        )
        self.confidence_value.pack(pady=(0, 10))
        
        # Статус
        stable_container = ctk.CTkFrame(status_grid, corner_radius=10, 
                                       fg_color=("#2b2b2b", "#1a1a1a"))
        stable_container.grid(row=0, column=2, sticky="nsew", padx=5)
        
        ctk.CTkLabel(
            stable_container,
            text="Статус",
            font=ctk.CTkFont(size=11),
            text_color=("#9ca3af", "#9ca3af")
        ).pack(pady=(10, 2))
        
        self.stability_value = ctk.CTkLabel(
            stable_container,
            text="—",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#9ca3af", "#9ca3af")
        )
        self.stability_value.pack(pady=(0, 10))
    
    def create_mode_panel(self, parent):
        """Панель выбора режима."""
        mode_frame = ctk.CTkFrame(parent, corner_radius=15)
        mode_frame.pack(fill="x", pady=(0, 15))
        
        header = ctk.CTkLabel(
            mode_frame,
            text="🎯 Режим распознавания",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Радио кнопки
        radio_container = ctk.CTkFrame(mode_frame, fg_color="transparent")
        radio_container.pack(fill="x", padx=15, pady=(0, 15))
        
        ctk.CTkRadioButton(
            radio_container,
            text="📊 Статические жесты (позы)",
            variable=self.recognition_mode,
            value='static'
        ).pack(anchor="w", pady=2)
        
        ctk.CTkRadioButton(
            radio_container,
            text="🔄 Динамические жесты (движения)",
            variable=self.recognition_mode,
            value='dynamic'
        ).pack(anchor="w", pady=2)
        
        ctk.CTkRadioButton(
            radio_container,
            text="🎭 Оба режима",
            variable=self.recognition_mode,
            value='both'
        ).pack(anchor="w", pady=2)
    
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
        
        self.history_widgets = []
    
    def create_chart_panel(self, parent):
        """Панель графика."""
        chart_frame = ctk.CTkFrame(parent, corner_radius=15, height=220)
        chart_frame.pack(fill="x", pady=(0, 15))
        chart_frame.pack_propagate(False)
        
        header = ctk.CTkLabel(
            chart_frame,
            text="📊 График уверенности",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        canvas_frame = ctk.CTkFrame(chart_frame, corner_radius=10, 
                                    fg_color=("#2b2b2b", "#1a1a1a"))
        canvas_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.chart_canvas = ctk.CTkCanvas(
            canvas_frame,
            bg="#1a1a1a",
            highlightthickness=0
        )
        self.chart_canvas.pack(fill="both", expand=True, padx=5, pady=5)
    
    def create_settings_panel(self, parent):
        """Панель настроек."""
        settings_frame = ctk.CTkFrame(parent, corner_radius=15)
        settings_frame.pack(fill="both", expand=True)
        
        header = ctk.CTkLabel(
            settings_frame,
            text="⚙️ Настройки",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        settings_container = ctk.CTkFrame(settings_frame, fg_color="transparent")
        settings_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Порог стабильности
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
            variable=self.settings['stability_threshold'],
            command=self.update_stability_threshold
        )
        stability_slider.pack(fill="x", pady=(0, 15))
        
        # Минимальная уверенность
        ctk.CTkLabel(
            settings_container,
            text="Мин. уверенность:",
            font=ctk.CTkFont(size=12),
            text_color=("#9ca3af", "#9ca3af")
        ).pack(anchor="w", pady=(0, 2))
        
        conf_slider = ctk.CTkSlider(
            settings_container,
            from_=0.1,
            to=1.0,
            variable=self.settings['min_confidence']
        )
        conf_slider.pack(fill="x", pady=(0, 15))
        
        # Скорость речи
        ctk.CTkLabel(
            settings_container,
            text="Скорость речи:",
            font=ctk.CTkFont(size=12),
            text_color=("#9ca3af", "#9ca3af")
        ).pack(anchor="w", pady=(0, 2))
        
        rate_slider = ctk.CTkSlider(
            settings_container,
            from_=100,
            to=200,
            variable=self.settings['speech_rate'],
            command=self.update_speech_rate
        )
        rate_slider.pack(fill="x", pady=(0, 15))
        
        # Громкость речи
        ctk.CTkLabel(
            settings_container,
            text="Громкость речи:",
            font=ctk.CTkFont(size=12),
            text_color=("#9ca3af", "#9ca3af")
        ).pack(anchor="w", pady=(0, 2))
        
        volume_slider = ctk.CTkSlider(
            settings_container,
            from_=0.0,
            to=1.0,
            variable=self.settings['speech_volume'],
            command=self.update_speech_volume
        )
        volume_slider.pack(fill="x")
    
    # Остальные методы аналогичны Modern GUI...
    # (toggle_recognition, process_video, display_frame, update_chart, etc.)
    
    def toggle_recognition(self):
        """Переключение распознавания."""
        if not self.is_running:
            self.start_recognition()
        else:
            self.stop_recognition()
    
    def start_recognition(self):
        """Запуск распознавания."""
        if not self.camera.open():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру")
            return
        
        self.is_running = True
        self.start_button.configure(text="⏸  Стоп", fg_color="#c92a2a", hover_color="#a61f1f")
        self.status_indicator.configure(text="🟢 Работает", text_color="#51cf66")
        self.repeat_button.configure(state="normal")
        self.info_overlay.place_forget()
        
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
    
    def stop_recognition(self):
        """Остановка распознавания."""
        self.is_running = False
        self.start_button.configure(text="▶  Старт", fg_color="#1976d2", hover_color="#42a5f5")
        self.status_indicator.configure(text="⚫ Остановлено", text_color="#ff6b6b")
        
        if self.camera:
            self.camera.release()
        
        self.info_overlay.configure(text="Нажмите 'Старт' для начала работы")
        self.info_overlay.place(relx=0.5, rely=0.5, anchor="center")
    
    def process_video(self):
        """Обработка видео."""
        while self.is_running:
            ret, frame = self.camera.read()
            
            if not ret:
                break
            
            # Обнаружение рук
            hands_data, annotated_frame = self.gesture_detector.detect(frame)
            
            # Обновляем статус рук
            self.update_hands_status(hands_data)
            
            # Добавляем в динамический распознаватель
            self.dynamic_recognizer.add_frame(hands_data)
            
            # Распознавание в зависимости от режима
            mode = self.recognition_mode.get()
            gesture = None
            confidence = 0.0
            
            if mode == 'dynamic' or mode == 'both':
                # Динамические жесты
                if self.dynamic_recognizer.is_sequence_ready():
                    sequence = self.dynamic_recognizer.get_sequence('both')
                    gesture, confidence = self.dynamic_model.predict(sequence)
            
            if (mode == 'static' or mode == 'both') and not gesture:
                # Статические жесты
                if hands_data['count'] > 0:
                    # Берем первую руку для совместимости
                    landmarks = hands_data.get('left') or hands_data.get('right')
                    if landmarks is not None:
                        try:
                            normalized_landmarks = self.static_model.normalize_landmarks(landmarks)
                            gesture, confidence = self.static_model.predict(normalized_landmarks)
                        except:
                            pass
            
            # Обновление GUI
            if gesture:
                # Проверка стабильности
                if gesture == self.current_gesture:
                    self.gesture_stable_count += 1
                else:
                    self.current_gesture = gesture
                    self.gesture_stable_count = 1
                
                self.gesture_confidence = confidence
                
                # Озвучивание
                if (self.gesture_stable_count >= self.stability_threshold and 
                    gesture != self.last_gesture and
                    confidence >= self.settings['min_confidence'].get()):
                    self.last_gesture = gesture
                    self.add_to_history(gesture, confidence, mode)
                    self.tts.speak(gesture)
                
                self.update_status_display(gesture, confidence)
                self.confidence_history.append(confidence)
                self.update_chart()
            else:
                self.current_gesture = None
                self.gesture_stable_count = 0
                self.update_status_display(None, 0.0)
            
            # Отображение
            self.display_frame(annotated_frame)
    
    def update_hands_status(self, hands_data):
        """Обновление статуса рук."""
        left = hands_data.get('left') is not None
        right = hands_data.get('right') is not None
        
        self.hands_status['left'] = left
        self.hands_status['right'] = right
        
        # Обновляем метки
        if left:
            self.left_hand_label.configure(text="Левая ✅", text_color="#51cf66")
        else:
            self.left_hand_label.configure(text="Левая ❌", text_color="#ff6b6b")
        
        if right:
            self.right_hand_label.configure(text="Правая ✅", text_color="#51cf66")
        else:
            self.right_hand_label.configure(text="Правая ❌", text_color="#ff6b6b")
    
    def display_frame(self, frame):
        """Отображение кадра."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_canvas.delete("all")
        self.video_canvas.create_image(320, 240, image=imgtk)
        self.video_canvas.imgtk = imgtk
    
    def update_status_display(self, gesture, confidence):
        """Обновление статуса."""
        if gesture:
            self.gesture_value.configure(text=gesture)
            self.confidence_value.configure(text=f"{confidence:.1%}")
            
            if self.gesture_stable_count >= self.stability_threshold:
                self.stability_value.configure(text="✓ Стабильно", text_color="#51cf66")
            else:
                progress = f"{self.gesture_stable_count}/{self.stability_threshold}"
                self.stability_value.configure(text=f"⏳ {progress}", text_color="#ffd43b")
        else:
            self.gesture_value.configure(text="—")
            self.confidence_value.configure(text="—")
            self.stability_value.configure(text="Покажите жест", text_color="#9ca3af")
    
    def update_chart(self):
        """Обновление графика."""
        self.chart_canvas.delete("all")
        
        if len(self.confidence_history) < 2:
            return
        
        width = self.chart_canvas.winfo_width()
        height = self.chart_canvas.winfo_height()
        
        if width < 10 or height < 10:
            return
        
        padding = 25
        chart_width = width - 2 * padding
        chart_height = height - 2 * padding
        
        # Сетка
        for i in range(5):
            y = padding + (chart_height * i / 4)
            self.chart_canvas.create_line(
                padding, y, width - padding, y,
                fill='#3d3d3d', dash=(2, 2), width=1
            )
            
            label = f"{1.0 - i * 0.25:.1f}"
            self.chart_canvas.create_text(
                padding - 8, y,
                text=label, anchor='e',
                fill='#6b7280', font=('Segoe UI', 8)
            )
        
        # График
        points = list(self.confidence_history)
        step = chart_width / (len(points) - 1) if len(points) > 1 else 0
        
        coords = []
        for i, conf in enumerate(points):
            x = padding + i * step
            y = padding + chart_height * (1 - conf)
            coords.extend([x, y])
        
        if len(coords) >= 4:
            self.chart_canvas.create_line(
                coords, fill='#42a5f5', width=3, smooth=True
            )
            
            if len(coords) >= 2:
                last_x, last_y = coords[-2], coords[-1]
                self.chart_canvas.create_oval(
                    last_x - 5, last_y - 5,
                    last_x + 5, last_y + 5,
                    fill='#42a5f5', outline='#1976d2', width=2
                )
    
    def add_to_history(self, gesture, confidence, mode):
        """Добавление в историю."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        mode_icon = "📊" if mode == 'static' else "🔄"
        entry = f"{timestamp} {mode_icon} {gesture} ({confidence:.1%})"
        
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
            old_widget = self.history_widgets.pop()
            old_widget.destroy()
    
    def clear_history(self):
        """Очистка истории."""
        self.gesture_history.clear()
        self.confidence_history.clear()
        
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
        
        self.update_chart()
    
    def repeat_gesture(self):
        """Повторное озвучивание."""
        if self.last_gesture:
            self.tts.speak(self.last_gesture, force=True)
    
    def update_stability_threshold(self, value):
        """Обновление порога стабильности."""
        self.stability_threshold = int(float(value))
    
    def update_speech_rate(self, value):
        """Обновление скорости речи."""
        if self.tts:
            self.tts.set_rate(int(float(value)))
    
    def update_speech_volume(self, value):
        """Обновление громкости речи."""
        if self.tts:
            self.tts.set_volume(float(value))
    
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
    
    parser = argparse.ArgumentParser(
        description='SignVoiceAI Advanced GUI - 2 руки + Движения'
    )
    parser.add_argument('--static-model', type=str, default=None,
                       help='Путь к статической модели')
    parser.add_argument('--dynamic-model', type=str, default=None,
                       help='Путь к динамической модели')
    parser.add_argument('--camera', type=int, default=0,
                       help='Индекс камеры')
    
    args = parser.parse_args()
    
    root = ctk.CTk()
    app = AdvancedSignVoiceGUI(root, 
                               static_model_path=args.static_model,
                               dynamic_model_path=args.dynamic_model,
                               camera_index=args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()


