"""
SignVoiceAI - Современный графический интерфейс для распознавания жестов.

Приложение с современным GUI, включающим:
- Видео с камеры в реальном времени
- Панель управления с кнопками
- История распознанных жестов
- График уверенности распознавания
- Настройки приложения
- Темная/светлая тема
"""

import cv2
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import queue
from datetime import datetime
from collections import deque
import numpy as np

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.camera import Camera
from utils.gestures import GestureDetector
from utils.speech import TextToSpeech
from utils.sentence_builder import SentenceBuilder
from utils.gesture_store import GestureStore
from utils.ollama_client import OllamaClient
from model.gesture_model import GestureModelWrapper, GESTURE_CLASSES


class ModernSignVoiceAI:
    """
    Главный класс приложения с современным GUI.
    """
    
    def __init__(self, root, model_path=None, camera_index=0):
        """
        Инициализация приложения.
        
        Args:
            root: Корневое окно Tkinter
            model_path: Путь к обученной модели
            camera_index: Индекс камеры
        """
        self.root = root
        self.root.title("SignVoiceAI - Распознавание жестов")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Переменные приложения
        self.camera_index = camera_index
        self.camera = None
        self.gesture_detector = None
        self.gesture_model = None
        self.tts = None
        self.is_running = False
        self.video_thread = None
        
        # Состояние распознавания
        self.current_gesture = None
        self.last_gesture = None
        self.gesture_confidence = 0.0
        self.gesture_stable_count = 0
        self.stability_threshold = 5
        
        # История жестов
        self.gesture_history = deque(maxlen=20)
        self.confidence_history = deque(maxlen=50)

        # Предложение из жестов
        self.sentence_builder = SentenceBuilder(max_tokens=40, dedupe_window_s=0.7)
        self.sentence_var = tk.StringVar(value="")

        self.session_events = []
        self.gesture_store = GestureStore()
        self.ollama_url_var = tk.StringVar(value="http://localhost:11434")
        self.ollama_model_var = tk.StringVar(value="llama3.1")
        self.ollama_temperature_var = tk.DoubleVar(value=0.2)
        self._sending_to_ai = False
        
        # Настройки
        self.settings = {
            'stability_threshold': tk.IntVar(value=5),
            'min_confidence': tk.DoubleVar(value=0.5),
            'speech_rate': tk.IntVar(value=150),
            'speech_volume': tk.DoubleVar(value=0.8),
            'theme': tk.StringVar(value='dark')
        }
        
        # Инициализация компонентов
        self.init_components(model_path)
        
        # Создание GUI
        self.create_gui()
        
        # Обработчик закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def init_components(self, model_path):
        """Инициализация компонентов распознавания."""
        print("=" * 60)
        print("Инициализация SignVoiceAI...")
        print("=" * 60)
        
        # Модель
        print("[1/4] Загрузка модели...")
        self.gesture_model = GestureModelWrapper(model_path=model_path, use_dummy=True)
        
        # Детектор жестов
        print("[2/4] Инициализация детектора жестов...")
        self.gesture_detector = GestureDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Синтезатор речи
        print("[3/4] Инициализация синтезатора речи...")
        self.tts = TextToSpeech(
            rate=self.settings['speech_rate'].get(),
            volume=self.settings['speech_volume'].get()
        )
        
        # Камера
        print("[4/4] Инициализация камеры...")
        self.camera = Camera(camera_index=self.camera_index, width=640, height=480)
        
        print("=" * 60)
        print("Инициализация завершена!")
        print("=" * 60)
    
    def create_gui(self):
        """Создание графического интерфейса."""
        
        # Стиль
        self.setup_styles()
        
        # Главный контейнер
        main_container = tk.Frame(self.root, bg='#1e1e1e')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель (видео и управление)
        left_panel = tk.Frame(main_container, bg='#1e1e1e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Правая панель (история и настройки)
        right_panel = tk.Frame(main_container, bg='#1e1e1e', width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Создание элементов интерфейса
        self.create_video_panel(left_panel)
        self.create_control_panel(left_panel)
        self.create_status_panel(left_panel)
        self.create_history_panel(right_panel)
        self.create_chart_panel(right_panel)
        self.create_settings_panel(right_panel)
    
    def setup_styles(self):
        """Настройка стилей."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Цвета
        bg_dark = '#1e1e1e'
        bg_card = '#2d2d2d'
        fg_light = '#ffffff'
        accent = '#0d7377'
        accent_hover = '#14ffec'
        
        # Кнопки
        style.configure('Accent.TButton',
                       background=accent,
                       foreground=fg_light,
                       borderwidth=0,
                       focuscolor='none',
                       padding=10,
                       font=('Segoe UI', 10, 'bold'))
        
        style.map('Accent.TButton',
                 background=[('active', accent_hover)])
    
    def create_video_panel(self, parent):
        """Создание панели с видео."""
        # Заголовок
        header = tk.Frame(parent, bg='#2d2d2d', height=60)
        header.pack(fill=tk.X, pady=(0, 10))
        
        title = tk.Label(header, text="📹 Видеопоток",
                        font=('Segoe UI', 16, 'bold'),
                        bg='#2d2d2d', fg='#14ffec')
        title.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Индикатор статуса
        self.status_indicator = tk.Label(header, text="⚫ Остановлено",
                                        font=('Segoe UI', 10),
                                        bg='#2d2d2d', fg='#ff6b6b')
        self.status_indicator.pack(side=tk.RIGHT, padx=20, pady=15)
        
        # Видео фрейм
        video_frame = tk.Frame(parent, bg='#000000', relief=tk.SOLID, borderwidth=2)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg='#000000')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Информационная панель поверх видео
        self.info_overlay = tk.Label(video_frame,
                                     text="Нажмите 'Старт' для начала",
                                     font=('Segoe UI', 14),
                                     bg='#000000', fg='#14ffec')
        self.info_overlay.place(relx=0.5, rely=0.5, anchor='center')
    
    def create_control_panel(self, parent):
        """Создание панели управления."""
        control_frame = tk.Frame(parent, bg='#2d2d2d', height=80)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        control_frame.pack_propagate(False)
        
        # Контейнер для кнопок
        button_container = tk.Frame(control_frame, bg='#2d2d2d')
        button_container.pack(expand=True)
        
        # Кнопка Старт/Стоп
        self.start_button = tk.Button(button_container,
                                      text="▶ Старт",
                                      font=('Segoe UI', 12, 'bold'),
                                      bg='#0d7377', fg='white',
                                      activebackground='#14ffec',
                                      activeforeground='#1e1e1e',
                                      relief=tk.FLAT,
                                      padx=30, pady=15,
                                      cursor='hand2',
                                      command=self.toggle_recognition)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Кнопка Повтор
        self.repeat_button = tk.Button(button_container,
                                       text="🔊 Повторить",
                                       font=('Segoe UI', 11),
                                       bg='#495057', fg='white',
                                       activebackground='#6c757d',
                                       relief=tk.FLAT,
                                       padx=20, pady=15,
                                       cursor='hand2',
                                       state=tk.DISABLED,
                                       command=self.repeat_gesture)
        self.repeat_button.pack(side=tk.LEFT, padx=5)
        
        # Кнопка Очистить историю
        clear_button = tk.Button(button_container,
                                text="🗑️ Очистить",
                                font=('Segoe UI', 11),
                                bg='#495057', fg='white',
                                activebackground='#6c757d',
                                relief=tk.FLAT,
                                padx=20, pady=15,
                                cursor='hand2',
                                command=self.clear_history)
        clear_button.pack(side=tk.LEFT, padx=5)
    
    def create_status_panel(self, parent):
        """Создание панели статуса."""
        status_frame = tk.Frame(parent, bg='#2d2d2d', height=100)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        status_frame.pack_propagate(False)
        
        # Сетка для статуса
        status_grid = tk.Frame(status_frame, bg='#2d2d2d')
        status_grid.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Текущий жест
        gesture_label = tk.Label(status_grid, text="Текущий жест:",
                                font=('Segoe UI', 10),
                                bg='#2d2d2d', fg='#a0a0a0')
        gesture_label.grid(row=0, column=0, sticky='w', padx=(0, 20))
        
        self.gesture_value = tk.Label(status_grid, text="—",
                                      font=('Segoe UI', 16, 'bold'),
                                      bg='#2d2d2d', fg='#14ffec')
        self.gesture_value.grid(row=0, column=1, sticky='w')
        
        # Уверенность
        conf_label = tk.Label(status_grid, text="Уверенность:",
                             font=('Segoe UI', 10),
                             bg='#2d2d2d', fg='#a0a0a0')
        conf_label.grid(row=0, column=2, sticky='w', padx=(40, 20))
        
        self.confidence_value = tk.Label(status_grid, text="—",
                                        font=('Segoe UI', 16, 'bold'),
                                        bg='#2d2d2d', fg='#ffffff')
        self.confidence_value.grid(row=0, column=3, sticky='w')
        
        # Статус стабильности
        stable_label = tk.Label(status_grid, text="Статус:",
                               font=('Segoe UI', 10),
                               bg='#2d2d2d', fg='#a0a0a0')
        stable_label.grid(row=1, column=0, sticky='w', padx=(0, 20), pady=(10, 0))
        
        self.stability_value = tk.Label(status_grid, text="—",
                                       font=('Segoe UI', 12),
                                       bg='#2d2d2d', fg='#a0a0a0')
        self.stability_value.grid(row=1, column=1, columnspan=3, sticky='w', pady=(10, 0))
    
    def create_history_panel(self, parent):
        """Создание панели истории жестов."""
        history_frame = tk.LabelFrame(parent, text="📜 История жестов",
                                     font=('Segoe UI', 12, 'bold'),
                                     bg='#2d2d2d', fg='#14ffec',
                                     relief=tk.FLAT)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Панель предложения
        sentence_frame = tk.Frame(history_frame, bg='#2d2d2d')
        sentence_frame.pack(fill=tk.X, padx=8, pady=(8, 0))

        tk.Label(sentence_frame, text="✍️ Предложение:",
                 font=('Segoe UI', 10, 'bold'),
                 bg='#2d2d2d', fg='#a0a0a0').pack(side=tk.LEFT)

        clear_sentence_btn = tk.Button(
            sentence_frame,
            text="Очистить",
            font=('Segoe UI', 8),
            bg='#1e1e1e',
            fg='#ffffff',
            command=self.clear_sentence
        )
        clear_sentence_btn.pack(side=tk.RIGHT)

        self.sentence_label = tk.Label(
            history_frame,
            textvariable=self.sentence_var,
            font=('Segoe UI', 10),
            bg='#1e1e1e',
            fg='#ffffff',
            wraplength=360,
            justify=tk.LEFT,
            anchor='w'
        )
        self.sentence_label.pack(fill=tk.X, padx=8, pady=(6, 8))

        actions = tk.Frame(history_frame, bg='#2d2d2d')
        actions.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.save_gestures_btn = tk.Button(
            actions,
            text="💾 Сохранить",
            font=('Segoe UI', 9, 'bold'),
            bg='#0d7377',
            fg='#ffffff',
            relief=tk.FLAT,
            command=self.save_current_gestures,
        )
        self.save_gestures_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        self.send_to_ai_btn = tk.Button(
            actions,
            text="🤖 Отправить в ИИ",
            font=('Segoe UI', 9, 'bold'),
            bg='#0d7377',
            fg='#ffffff',
            relief=tk.FLAT,
            command=self.send_saved_to_ai,
        )
        self.send_to_ai_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        self.clear_saved_btn = tk.Button(
            actions,
            text="🧹 Очистить сохранённые",
            font=('Segoe UI', 9),
            bg='#1e1e1e',
            fg='#ffffff',
            relief=tk.FLAT,
            command=self.clear_saved_gestures,
        )
        self.clear_saved_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Скроллбар
        scrollbar = tk.Scrollbar(history_frame, bg='#2d2d2d')
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        
        # Список истории
        self.history_listbox = tk.Listbox(history_frame,
                                          font=('Consolas', 10),
                                          bg='#1e1e1e', fg='#ffffff',
                                          selectbackground='#0d7377',
                                          relief=tk.FLAT,
                                          yscrollcommand=scrollbar.set)
        self.history_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.history_listbox.yview)
    
    def create_chart_panel(self, parent):
        """Создание панели с графиком уверенности."""
        chart_frame = tk.LabelFrame(parent, text="📊 График уверенности",
                                   font=('Segoe UI', 12, 'bold'),
                                   bg='#2d2d2d', fg='#14ffec',
                                   relief=tk.FLAT, height=200)
        chart_frame.pack(fill=tk.X, pady=(0, 10))
        chart_frame.pack_propagate(False)
        
        # Canvas для графика
        self.chart_canvas = tk.Canvas(chart_frame, bg='#1e1e1e',
                                      relief=tk.FLAT, highlightthickness=0)
        self.chart_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_settings_panel(self, parent):
        """Создание панели настроек."""
        settings_frame = tk.LabelFrame(parent, text="⚙️ Настройки",
                                      font=('Segoe UI', 12, 'bold'),
                                      bg='#2d2d2d', fg='#14ffec',
                                      relief=tk.FLAT)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Контейнер для настроек
        settings_container = tk.Frame(settings_frame, bg='#2d2d2d')
        settings_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Порог стабильности
        row = 0
        tk.Label(settings_container, text="Порог стабильности:",
                font=('Segoe UI', 9),
                bg='#2d2d2d', fg='#a0a0a0').grid(row=row, column=0, sticky='w', pady=5)
        
        stability_scale = tk.Scale(settings_container,
                                  from_=1, to=10,
                                  orient=tk.HORIZONTAL,
                                  variable=self.settings['stability_threshold'],
                                  bg='#2d2d2d', fg='#ffffff',
                                  troughcolor='#1e1e1e',
                                  highlightthickness=0,
                                  command=self.update_stability_threshold)
        stability_scale.grid(row=row, column=1, sticky='ew', pady=5)
        
        # Минимальная уверенность
        row += 1
        tk.Label(settings_container, text="Мин. уверенность:",
                font=('Segoe UI', 9),
                bg='#2d2d2d', fg='#a0a0a0').grid(row=row, column=0, sticky='w', pady=5)
        
        conf_scale = tk.Scale(settings_container,
                             from_=0.1, to=1.0,
                             resolution=0.1,
                             orient=tk.HORIZONTAL,
                             variable=self.settings['min_confidence'],
                             bg='#2d2d2d', fg='#ffffff',
                             troughcolor='#1e1e1e',
                             highlightthickness=0)
        conf_scale.grid(row=row, column=1, sticky='ew', pady=5)
        
        # Скорость речи
        row += 1
        tk.Label(settings_container, text="Скорость речи:",
                font=('Segoe UI', 9),
                bg='#2d2d2d', fg='#a0a0a0').grid(row=row, column=0, sticky='w', pady=5)
        
        rate_scale = tk.Scale(settings_container,
                             from_=100, to=200,
                             orient=tk.HORIZONTAL,
                             variable=self.settings['speech_rate'],
                             bg='#2d2d2d', fg='#ffffff',
                             troughcolor='#1e1e1e',
                             highlightthickness=0,
                             command=self.update_speech_rate)
        rate_scale.grid(row=row, column=1, sticky='ew', pady=5)
        
        # Громкость речи
        row += 1
        tk.Label(settings_container, text="Громкость речи:",
                font=('Segoe UI', 9),
                bg='#2d2d2d', fg='#a0a0a0').grid(row=row, column=0, sticky='w', pady=5)
        
        volume_scale = tk.Scale(settings_container,
                               from_=0.0, to=1.0,
                               resolution=0.1,
                               orient=tk.HORIZONTAL,
                               variable=self.settings['speech_volume'],
                               bg='#2d2d2d', fg='#ffffff',
                               troughcolor='#1e1e1e',
                               highlightthickness=0,
                               command=self.update_speech_volume)
        volume_scale.grid(row=row, column=1, sticky='ew', pady=5)
        
        # Информация о жестах
        row += 1
        gestures_label = tk.Label(settings_container,
                                 text=f"Распознаваемые жесты:\n{', '.join(GESTURE_CLASSES)}",
                                 font=('Segoe UI', 8),
                                 bg='#2d2d2d', fg='#a0a0a0',
                                 justify=tk.LEFT,
                                 wraplength=350)
        gestures_label.grid(row=row, column=0, columnspan=2, sticky='w', pady=(15, 5))

        row += 1
        tk.Label(settings_container, text="Ollama URL:",
                 font=('Segoe UI', 9),
                 bg='#2d2d2d', fg='#a0a0a0').grid(row=row, column=0, sticky='w', pady=5)

        tk.Entry(settings_container,
                  textvariable=self.ollama_url_var,
                  bg='#1e1e1e',
                  fg='#ffffff',
                  relief=tk.FLAT).grid(row=row, column=1, sticky='ew', pady=5)

        row += 1
        tk.Label(settings_container, text="Ollama модель:",
                 font=('Segoe UI', 9),
                 bg='#2d2d2d', fg='#a0a0a0').grid(row=row, column=0, sticky='w', pady=5)

        tk.Entry(settings_container,
                  textvariable=self.ollama_model_var,
                  bg='#1e1e1e',
                  fg='#ffffff',
                  relief=tk.FLAT).grid(row=row, column=1, sticky='ew', pady=5)

        row += 1
        tk.Label(settings_container, text="Ollama темп.:",
                 font=('Segoe UI', 9),
                 bg='#2d2d2d', fg='#a0a0a0').grid(row=row, column=0, sticky='w', pady=5)

        tk.Scale(settings_container,
                  from_=0.0,
                  to=1.0,
                  resolution=0.05,
                  orient=tk.HORIZONTAL,
                  variable=self.ollama_temperature_var,
                  bg='#2d2d2d',
                  fg='#ffffff',
                  troughcolor='#1e1e1e',
                  highlightthickness=0).grid(row=row, column=1, sticky='ew', pady=5)
        
        settings_container.columnconfigure(1, weight=1)
    
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
        self.start_button.config(text="⏸ Стоп", bg='#c92a2a')
        self.status_indicator.config(text="🟢 Работает", fg='#51cf66')
        self.repeat_button.config(state=tk.NORMAL)
        self.info_overlay.place_forget()
        
        # Запуск потока обработки видео
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
    
    def stop_recognition(self):
        """Остановка распознавания."""
        self.is_running = False
        self.start_button.config(text="▶ Старт", bg='#0d7377')
        self.status_indicator.config(text="⚫ Остановлено", fg='#ff6b6b')
        
        if self.camera:
            self.camera.release()
        
        # Показываем сообщение
        self.info_overlay.config(text="Нажмите 'Старт' для начала")
        self.info_overlay.place(relx=0.5, rely=0.5, anchor='center')
    
    def process_video(self):
        """Обработка видеопотока."""
        while self.is_running:
            ret, frame = self.camera.read()
            
            if not ret:
                break
            
            # Обнаружение жестов
            landmarks, annotated_frame = self.gesture_detector.detect(frame)
            
            if landmarks is not None:
                try:
                    # Нормализация
                    normalized_landmarks = self.gesture_model.normalize_landmarks(landmarks)
                    
                    # Классификация
                    gesture, confidence = self.gesture_model.predict(normalized_landmarks)
                    
                    # Обновление состояния
                    if gesture == self.current_gesture:
                        self.gesture_stable_count += 1
                    else:
                        self.current_gesture = gesture
                        self.gesture_stable_count = 1
                    
                    self.gesture_confidence = confidence
                    
                    # Проверка стабильности и озвучивание
                    if (self.gesture_stable_count >= self.stability_threshold and 
                        gesture != self.last_gesture and
                        confidence >= self.settings['min_confidence'].get()):
                        self.last_gesture = gesture
                        self.add_to_history(gesture, confidence)
                        self.tts.speak(gesture)
                    
                    # Обновление GUI
                    self.update_status_display(gesture, confidence)
                    self.confidence_history.append(confidence)
                    self.update_chart()
                    
                except Exception as e:
                    print(f"Ошибка обработки: {e}")
            else:
                self.current_gesture = None
                self.gesture_stable_count = 0
                self.update_status_display(None, 0.0)
            
            # Отображение кадра
            self.display_frame(annotated_frame)
    
    def display_frame(self, frame):
        """Отображение кадра в GUI."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def update_status_display(self, gesture, confidence):
        """Обновление отображения статуса."""
        if gesture:
            self.gesture_value.config(text=gesture)
            self.confidence_value.config(text=f"{confidence:.2%}")
            
            if self.gesture_stable_count >= self.stability_threshold:
                self.stability_value.config(text="✓ Стабильно", fg='#51cf66')
            else:
                progress = f"{self.gesture_stable_count}/{self.stability_threshold}"
                self.stability_value.config(text=f"⏳ Определение... ({progress})", fg='#ffd43b')
        else:
            self.gesture_value.config(text="—")
            self.confidence_value.config(text="—")
            self.stability_value.config(text="Покажите жест", fg='#a0a0a0')
    
    def update_chart(self):
        """Обновление графика уверенности."""
        self.chart_canvas.delete("all")
        
        if len(self.confidence_history) < 2:
            return
        
        width = self.chart_canvas.winfo_width()
        height = self.chart_canvas.winfo_height()
        
        if width < 10 or height < 10:
            return
        
        padding = 20
        chart_width = width - 2 * padding
        chart_height = height - 2 * padding
        
        # Сетка
        for i in range(5):
            y = padding + (chart_height * i / 4)
            self.chart_canvas.create_line(padding, y, width - padding, y,
                                         fill='#3d3d3d', dash=(2, 2))
            
            label = f"{1.0 - i * 0.25:.1f}"
            self.chart_canvas.create_text(padding - 5, y,
                                         text=label, anchor='e',
                                         fill='#a0a0a0', font=('Segoe UI', 7))
        
        # График
        points = list(self.confidence_history)
        step = chart_width / (len(points) - 1) if len(points) > 1 else 0
        
        coords = []
        for i, conf in enumerate(points):
            x = padding + i * step
            y = padding + chart_height * (1 - conf)
            coords.extend([x, y])
        
        if len(coords) >= 4:
            self.chart_canvas.create_line(coords, fill='#14ffec', width=2, smooth=True)
            
            # Последняя точка
            if len(coords) >= 2:
                last_x, last_y = coords[-2], coords[-1]
                self.chart_canvas.create_oval(last_x - 4, last_y - 4,
                                             last_x + 4, last_y + 4,
                                             fill='#14ffec', outline='#0d7377', width=2)
    
    def add_to_history(self, gesture, confidence):
        """Добавление жеста в историю."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{timestamp} - {gesture} ({confidence:.2%})"
        
        self.gesture_history.append(entry)
        self.history_listbox.insert(0, entry)
        self.add_to_sentence(gesture)
        self.session_events.append({"ts": timestamp, "gesture": gesture, "confidence": float(confidence)})

    def save_current_gestures(self):
        if not self.session_events:
            messagebox.showwarning("Сохранение", "Нет распознанных жестов для сохранения")
            return

        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sentence": self.sentence_var.get(),
            "events": list(self.session_events),
        }
        self.gesture_store.append_record(record)
        messagebox.showinfo("Сохранение", f"Сохранено записей: {len(self.gesture_store.list_records())}")
        self.session_events.clear()
        self.clear_sentence()

    def clear_saved_gestures(self):
        self.gesture_store.clear()
        messagebox.showinfo("Сохранение", "Сохранённые жесты очищены")

    def send_saved_to_ai(self):
        if self._sending_to_ai:
            return

        records = self.gesture_store.list_records()
        if not records:
            messagebox.showwarning("ИИ", "Нет сохранённых записей. Сначала нажмите 'Сохранить'.")
            return

        prompt = self.gesture_store.build_prompt(records)
        self._sending_to_ai = True
        if self.send_to_ai_btn:
            self.send_to_ai_btn.config(state=tk.DISABLED)

        def worker():
            try:
                client = OllamaClient(base_url=self.ollama_url_var.get(), model=self.ollama_model_var.get())
                text = client.generate(prompt=prompt, temperature=float(self.ollama_temperature_var.get()))
                self.root.after(0, lambda: self._on_ai_response(text))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("ИИ", str(e)))
            finally:
                self.root.after(0, self._ai_send_finished)

        threading.Thread(target=worker, daemon=True).start()

    def _ai_send_finished(self):
        self._sending_to_ai = False
        if self.send_to_ai_btn:
            self.send_to_ai_btn.config(state=tk.NORMAL)

    def _on_ai_response(self, text: str):
        win = tk.Toplevel(self.root)
        win.title("Ответ ИИ (Ollama)")
        win.geometry("900x600")

        scrollbar = tk.Scrollbar(win)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(win, wrap="word", yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.DISABLED)

        try:
            if self.tts:
                self.tts.speak(text, force=True)
        except Exception:
            pass

    def clear_history(self):
        """Очистка истории."""
        self.gesture_history.clear()
        self.history_listbox.delete(0, tk.END)
        self.confidence_history.clear()
        self.update_chart()
        self.session_events.clear()

    def add_to_sentence(self, gesture):
        """Добавляет жест в предложение."""
        if self.sentence_builder.add_gesture(gesture):
            self.sentence_var.set(self.sentence_builder.get_sentence())
    def clear_sentence(self):
        """Очистка предложения."""
        self.sentence_builder.reset()
        self.sentence_var.set("")
    
    def repeat_gesture(self):
        """Повторное озвучивание жеста."""
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
        """Обработчик закрытия окна."""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        if self.gesture_detector:
            self.gesture_detector.close()
        
        if self.tts:
            self.tts.stop()
        
        self.root.destroy()


def main():
    """Точка входа в приложение."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SignVoiceAI - Современный GUI для распознавания жестов'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Путь к файлу обученной модели (.pth или .pt)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Индекс камеры (по умолчанию 0)'
    )
    
    args = parser.parse_args()
    
    # Создание приложения
    root = tk.Tk()
    app = ModernSignVoiceAI(root, model_path=args.model, camera_index=args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()


