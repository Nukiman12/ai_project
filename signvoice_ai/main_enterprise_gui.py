"""
SignVoiceAI Enterprise - Полноценный enterprise GUI.

Возможности:
- Система авторизации и регистрации
- Профили пользователей
- Продвинутая аналитика
- Множество тем
- Мультиязычность
- Достижения и прогресс
- Экспорт/импорт данных
"""

import cv2
import sys
import os
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import threading
from datetime import datetime
from collections import deque
import numpy as np
from tkinter import messagebox, filedialog
import uuid

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.camera import Camera
from utils.gestures import GestureDetector
from utils.speech import TextToSpeech
from model.gesture_model import GestureModelWrapper, GESTURE_CLASSES
from database.db_manager import DatabaseManager
from analytics.analytics_engine import AnalyticsEngine
from config.themes import ThemeManager, UserPreferences, LocalizationManager

# Модули обучения
import time
try:
    from training.training_module import DataCollector, ModelTrainer, GestureDataset
    from model.advanced_gesture_model import AdvancedGestureClassifier
    from torch.utils.data import DataLoader
    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Модули обучения недоступны: {e}")
    TRAINING_AVAILABLE = False

# Настройка темы CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class LoginWindow:
    """
    Окно авторизации и регистрации.
    """
    
    def __init__(self, db_manager: DatabaseManager, on_login_success):
        """
        Инициализация окна авторизации.
        
        Args:
            db_manager: Менеджер базы данных
            on_login_success: Callback при успешной авторизации
        """
        print("  → Создание окна CustomTkinter...")
        self.db = db_manager
        self.on_login_success = on_login_success
        
        self.window = ctk.CTk()
        print("  → Настройка окна...")
        self.window.title("SignVoiceAI - Вход")
        self.window.geometry("500x600")
        
        print("  → Создание UI элементов...")
        self.create_ui()
        print("  → UI готов")
        
    def create_ui(self):
        """Создает интерфейс окна авторизации."""
        # Главный контейнер
        main_container = ctk.CTkFrame(self.window, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=40, pady=40)
        
        # Логотип и заголовок
        title_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        title_frame.pack(pady=(0, 30))
        
        title = ctk.CTkLabel(
            title_frame,
            text="🎤 SignVoiceAI",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color=("#1976d2", "#42a5f5")
        )
        title.pack()
        
        subtitle = ctk.CTkLabel(
            title_frame,
            text="Enterprise Edition",
            font=ctk.CTkFont(size=14),
            text_color=("#666666", "#999999")
        )
        subtitle.pack()
        
        # Табы для входа и регистрации
        self.tabview = ctk.CTkTabview(main_container, width=400, height=400)
        self.tabview.pack(pady=20, fill="both", expand=True)
        
        self.tabview.add("Вход")
        self.tabview.add("Регистрация")
        
        # Вкладка входа
        self.create_login_tab()
        
        # Вкладка регистрации
        self.create_register_tab()
        
        # Кнопка гостевого входа
        guest_btn = ctk.CTkButton(
            main_container,
            text="Продолжить как гость",
            font=ctk.CTkFont(size=12),
            fg_color="transparent",
            hover_color=("#3d3d3d", "#2d2d2d"),
            command=self.guest_login
        )
        guest_btn.pack(pady=10)
    
    def create_login_tab(self):
        """Создает вкладку входа."""
        tab = self.tabview.tab("Вход")
        
        # Поля ввода
        fields_frame = ctk.CTkFrame(tab, fg_color="transparent")
        fields_frame.pack(pady=30, padx=20)
        
        # Имя пользователя
        ctk.CTkLabel(
            fields_frame,
            text="Имя пользователя",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 5))
        
        self.login_username = ctk.CTkEntry(
            fields_frame,
            width=300,
            height=40,
            placeholder_text="Введите имя пользователя"
        )
        self.login_username.pack(pady=(0, 20))
        
        # Пароль
        ctk.CTkLabel(
            fields_frame,
            text="Пароль",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 5))
        
        self.login_password = ctk.CTkEntry(
            fields_frame,
            width=300,
            height=40,
            placeholder_text="Введите пароль",
            show="●"
        )
        self.login_password.pack(pady=(0, 30))
        
        # Кнопка входа
        login_btn = ctk.CTkButton(
            fields_frame,
            text="Войти",
            width=300,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.do_login
        )
        login_btn.pack()
    
    def create_register_tab(self):
        """Создает вкладку регистрации."""
        tab = self.tabview.tab("Регистрация")
        
        # Поля ввода
        fields_frame = ctk.CTkFrame(tab, fg_color="transparent")
        fields_frame.pack(pady=20, padx=20)
        
        # Имя пользователя
        ctk.CTkLabel(
            fields_frame,
            text="Имя пользователя",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 5))
        
        self.reg_username = ctk.CTkEntry(
            fields_frame,
            width=300,
            height=40,
            placeholder_text="Выберите имя пользователя"
        )
        self.reg_username.pack(pady=(0, 15))
        
        # Email
        ctk.CTkLabel(
            fields_frame,
            text="Email (опционально)",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 5))
        
        self.reg_email = ctk.CTkEntry(
            fields_frame,
            width=300,
            height=40,
            placeholder_text="email@example.com"
        )
        self.reg_email.pack(pady=(0, 15))
        
        # Полное имя
        ctk.CTkLabel(
            fields_frame,
            text="Полное имя (опционально)",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 5))
        
        self.reg_fullname = ctk.CTkEntry(
            fields_frame,
            width=300,
            height=40,
            placeholder_text="Иван Иванов"
        )
        self.reg_fullname.pack(pady=(0, 15))
        
        # Пароль
        ctk.CTkLabel(
            fields_frame,
            text="Пароль",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 5))
        
        self.reg_password = ctk.CTkEntry(
            fields_frame,
            width=300,
            height=40,
            placeholder_text="Введите пароль",
            show="●"
        )
        self.reg_password.pack(pady=(0, 20))
        
        # Кнопка регистрации
        register_btn = ctk.CTkButton(
            fields_frame,
            text="Зарегистрироваться",
            width=300,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.do_register
        )
        register_btn.pack()
    
    def do_login(self):
        """Выполняет вход."""
        username = self.login_username.get().strip()
        password = self.login_password.get()
        
        if not username or not password:
            messagebox.showerror("Ошибка", "Заполните все поля")
            return
        
        user_id = self.db.authenticate_user(username, password)
        
        if user_id:
            self.window.destroy()
            self.on_login_success(user_id)
        else:
            messagebox.showerror("Ошибка", "Неверное имя пользователя или пароль")
    
    def do_register(self):
        """Выполняет регистрацию."""
        username = self.reg_username.get().strip()
        password = self.reg_password.get()
        email = self.reg_email.get().strip() or None
        fullname = self.reg_fullname.get().strip() or None
        
        if not username or not password:
            messagebox.showerror("Ошибка", "Заполните обязательные поля")
            return
        
        if len(password) < 6:
            messagebox.showerror("Ошибка", "Пароль должен быть не менее 6 символов")
            return
        
        user_id = self.db.create_user(username, password, email, fullname)
        
        if user_id:
            messagebox.showinfo("Успех", "Регистрация успешна! Теперь вы можете войти.")
            self.tabview.set("Вход")
            self.login_username.delete(0, 'end')
            self.login_username.insert(0, username)
        else:
            messagebox.showerror("Ошибка", "Пользователь с таким именем уже существует")
    
    def guest_login(self):
        """Вход как гость."""
        # Создаем временного пользователя-гостя
        guest_username = f"guest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_id = self.db.create_user(guest_username, "guest_password", None, "Гость")
        
        if user_id:
            self.window.destroy()
            self.on_login_success(user_id)
    
    def run(self):
        """Запускает окно авторизации."""
        print("  → Запуск mainloop()...")
        print("  → Окно должно появиться СЕЙЧАС!")
        self.window.mainloop()
        print("  → mainloop() завершен")


class EnterpriseSignVoiceGUI:
    """
    Главный класс Enterprise GUI с полным функционалом.
    """
    
    def __init__(self, user_id: int, db_manager: DatabaseManager):
        """
        Инициализация приложения.
        
        Args:
            user_id: ID пользователя
            db_manager: Менеджер базы данных
        """
        self.user_id = user_id
        self.db = db_manager
        self.analytics = AnalyticsEngine(db_manager)
        
        # Загружаем информацию о пользователе
        self.user_info = self.db.get_user_info(user_id)
        
        # Инициализация менеджеров
        self.theme_manager = ThemeManager(default_theme='dark')
        self.preferences = UserPreferences(db_manager, user_id)
        self.localization = LocalizationManager(self.preferences.get('language', 'ru'))
        
        # Применяем сохраненную тему
        saved_theme = self.preferences.get('theme', 'dark')
        self.theme_manager.set_theme(saved_theme)
        
        # Создаем главное окно
        self.root = ctk.CTk()
        self.root.title(self.localization['app_title'])
        self.root.geometry("1600x1000")
        
        # Компоненты распознавания
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
        self.stability_threshold = self.preferences.get('stability_threshold', 5)
        
        # История и сессия
        self.gesture_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=50)
        self.session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
        self.gesture_count_since_achievement_check = 0
        
        # Компоненты обучения
        if TRAINING_AVAILABLE:
            self.data_collector = DataCollector(save_dir="training_data")
            self.model_trainer = None
            self.is_recording_for_training = False
            self.current_training_gesture = ""
            self.recording_buffer = []
            self.recording_start_time = None
            self.training_history_widgets = []
        else:
            self.data_collector = None
        
        # Инициализация компонентов
        self.init_components()
        
        # Создание GUI
        self.create_gui()
        
        # Обработчик закрытия
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Начинаем сессию
        self.db.start_session(self.user_id, self.session_id)
    
    def init_components(self):
        """Инициализация компонентов распознавания."""
        print("=" * 60)
        print("Инициализация SignVoiceAI Enterprise...")
        print(f"Пользователь: {self.user_info['username']}")
        print("=" * 60)
        
        # Модель
        print("[1/4] Загрузка модели...")
        model_path = self.preferences.get('model_path')
        
        # Если путь не указан, используем модель по умолчанию
        if not model_path:
            # Проверяем наличие обученных моделей по приоритету
            import os
            possible_models = [
                'models/trained_advanced_model.pth',  # Ваша обученная модель (высший приоритет)
                'models/gesture_model.pth',           # Базовая модель
            ]
            for model_file in possible_models:
                if os.path.exists(model_file):
                    model_path = model_file
                    break
        
        # Пытаемся загрузить реальную модель, если не получится - используем заглушку
        self.gesture_model = GestureModelWrapper(model_path=model_path, use_dummy=False)
        
        # Детектор жестов
        print("[2/4] Инициализация детектора...")
        max_hands = 2 if self.preferences.get('detect_both_hands', True) else 1
        self.gesture_detector = GestureDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=max_hands
        )
        
        # Синтезатор речи
        print("[3/4] Инициализация речи...")
        if self.preferences.get('speech_enabled', True):
            self.tts = TextToSpeech(
                rate=self.preferences.get('speech_rate', 150),
                volume=self.preferences.get('speech_volume', 0.8)
            )
        
        # Камера
        print("[4/4] Инициализация камеры...")
        camera_index = self.preferences.get('camera_index', 0)
        self.camera = Camera(camera_index=camera_index, width=640, height=480)
        
        print("=" * 60)
        print("Инициализация завершена!")
        print("=" * 60)
    
    def create_gui(self):
        """Создает графический интерфейс."""
        # Главный контейнер
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Верхняя панель (профиль и навигация)
        self.create_top_panel(main_container)
        
        # Основной контент
        content_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, pady=10)
        
        # Создаем TabView для различных разделов
        self.tabview = ctk.CTkTabview(content_frame)
        self.tabview.pack(fill="both", expand=True)
        
        # Добавляем вкладки
        self.tabview.add("🎥 Распознавание")
        self.tabview.add("📊 Статистика")
        self.tabview.add("🏆 Достижения")
        self.tabview.add("⚙️ Настройки")
        self.tabview.add("👤 Профиль")
        if TRAINING_AVAILABLE:
            self.tabview.add("🎓 Обучение")
        
        # Создаем содержимое вкладок
        self.create_recognition_tab()
        self.create_statistics_tab()
        self.create_achievements_tab()
        self.create_settings_tab()
        self.create_profile_tab()
        if TRAINING_AVAILABLE:
            self.create_training_tab()
    
    def create_top_panel(self, parent):
        """Создает верхнюю панель с профилем."""
        top_panel = ctk.CTkFrame(parent, height=70)
        top_panel.pack(fill="x", pady=(0, 10))
        top_panel.pack_propagate(False)
        
        # Информация о пользователе
        user_frame = ctk.CTkFrame(top_panel, fg_color="transparent")
        user_frame.pack(side="left", padx=20, pady=15)
        
        greeting = ctk.CTkLabel(
            user_frame,
            text=f"👋 Привет, {self.user_info.get('full_name') or self.user_info['username']}!",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        greeting.pack(anchor="w")
        
        level = ctk.CTkLabel(
            user_frame,
            text=f"Уровень {self.user_info['user_level']} • {self.user_info['total_gestures']} жестов",
            font=ctk.CTkFont(size=12),
            text_color=("#666666", "#999999")
        )
        level.pack(anchor="w")
        
        # Кнопки действий
        actions_frame = ctk.CTkFrame(top_panel, fg_color="transparent")
        actions_frame.pack(side="right", padx=20, pady=15)
        
        export_btn = ctk.CTkButton(
            actions_frame,
            text="📥 Экспорт",
            width=100,
            height=40,
            command=self.export_data
        )
        export_btn.pack(side="left", padx=5)
        
        logout_btn = ctk.CTkButton(
            actions_frame,
            text="🚪 Выход",
            width=100,
            height=40,
            fg_color=("#c92a2a", "#a61f1f"),
            hover_color=("#a61f1f", "#8b1818"),
            command=self.logout
        )
        logout_btn.pack(side="left", padx=5)
    
    def create_recognition_tab(self):
        """Создает вкладку распознавания (аналогично ModernSignVoiceGUI)."""
        tab = self.tabview.tab("🎥 Распознавание")
        
        # Используем ту же структуру что и в Modern GUI
        container = ctk.CTkFrame(tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Левая панель (видео)
        left_panel = ctk.CTkFrame(container, fg_color="transparent")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Правая панель (история и график)
        right_panel = ctk.CTkFrame(container, fg_color="transparent", width=400)
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)
        
        self.create_video_panel(left_panel)
        self.create_control_panel(left_panel)
        self.create_status_panel(left_panel)
        self.create_history_panel(right_panel)
        self.create_chart_panel(right_panel)
    
    def create_statistics_tab(self):
        """Создает вкладку статистики."""
        tab = self.tabview.tab("📊 Статистика")
        
        # Получаем данные dashboard
        dashboard = self.analytics.get_user_dashboard(self.user_id)
        
        # Scrollable frame для статистики
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Обзор
        self.create_stats_section(scroll_frame, "📈 Обзор", dashboard['overview'])
        
        # Производительность
        self.create_stats_section(scroll_frame, "🎯 Производительность", dashboard['performance'])
        
        # Топ жестов
        top_gestures_data = {
            'gestures': ', '.join([g['gesture_name'] for g in dashboard['top_gestures'][:5]])
        }
        self.create_stats_section(scroll_frame, "🌟 Топ жестов", top_gestures_data)
        
        # Рекомендации
        recommendations_frame = ctk.CTkFrame(scroll_frame, corner_radius=10)
        recommendations_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            recommendations_frame,
            text="💡 Рекомендации",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        for rec in dashboard['recommendations']:
            ctk.CTkLabel(
                recommendations_frame,
                text=rec,
                font=ctk.CTkFont(size=12),
                anchor="w",
                wraplength=500
            ).pack(anchor="w", padx=20, pady=5)
        
        ctk.CTkLabel(recommendations_frame, text="").pack(pady=5)
    
    def create_achievements_tab(self):
        """Создает вкладку достижений."""
        tab = self.tabview.tab("🏆 Достижения")
        
        achievements = self.db.get_user_achievements(self.user_id)
        
        # Статистика достижений
        completed = len([a for a in achievements if a.get('is_completed')])
        total = len(achievements)
        points = sum([a['points'] for a in achievements if a.get('is_completed')])
        
        stats_frame = ctk.CTkFrame(tab, height=80)
        stats_frame.pack(fill="x", padx=10, pady=10)
        stats_frame.pack_propagate(False)
        
        stats_label = ctk.CTkLabel(
            stats_frame,
            text=f"🏆 Достижения: {completed}/{total} • 💎 Очки: {points}",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        stats_label.pack(expand=True)
        
        # Список достижений
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        for achievement in achievements:
            self.create_achievement_card(scroll_frame, achievement)
    
    def create_settings_tab(self):
        """Создает вкладку настроек."""
        tab = self.tabview.tab("⚙️ Настройки")
        
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Тема
        theme_frame = ctk.CTkFrame(scroll_frame, corner_radius=10)
        theme_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            theme_frame,
            text="🎨 Тема приложения",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        themes = self.theme_manager.get_available_themes()
        self.theme_var = ctk.StringVar(value=self.preferences.get('theme', 'dark'))
        
        theme_menu = ctk.CTkOptionMenu(
            theme_frame,
            values=themes,
            variable=self.theme_var,
            command=self.change_theme,
            width=200
        )
        theme_menu.pack(padx=15, pady=(0, 15), anchor="w")
        
        # Язык
        lang_frame = ctk.CTkFrame(scroll_frame, corner_radius=10)
        lang_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            lang_frame,
            text="🌐 Язык",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        self.lang_var = ctk.StringVar(value=self.preferences.get('language', 'ru'))
        
        lang_menu = ctk.CTkOptionMenu(
            lang_frame,
            values=['ru', 'en'],
            variable=self.lang_var,
            command=self.change_language,
            width=200
        )
        lang_menu.pack(padx=15, pady=(0, 15), anchor="w")
        
        # Распознавание
        recog_frame = ctk.CTkFrame(scroll_frame, corner_radius=10)
        recog_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            recog_frame,
            text="🎯 Распознавание",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        # Автоозвучивание
        auto_speak = ctk.CTkSwitch(
            recog_frame,
            text="Автоматическое озвучивание",
            variable=ctk.BooleanVar(value=self.preferences.get('auto_speak', True))
        )
        auto_speak.pack(anchor="w", padx=15, pady=5)
        
        # Обе руки
        both_hands = ctk.CTkSwitch(
            recog_frame,
            text="Распознавание обеих рук",
            variable=ctk.BooleanVar(value=self.preferences.get('detect_both_hands', True))
        )
        both_hands.pack(anchor="w", padx=15, pady=5)
        
        ctk.CTkLabel(recog_frame, text="").pack(pady=5)
        
        # Кнопка сброса настроек
        reset_btn = ctk.CTkButton(
            scroll_frame,
            text="🔄 Сбросить настройки",
            command=self.reset_preferences,
            fg_color=("#c92a2a", "#a61f1f")
        )
        reset_btn.pack(pady=10)
    
    def create_profile_tab(self):
        """Создает вкладку профиля."""
        tab = self.tabview.tab("👤 Профиль")
        
        profile_frame = ctk.CTkFrame(tab)
        profile_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Информация о профиле
        info_frame = ctk.CTkFrame(profile_frame, corner_radius=10)
        info_frame.pack(fill="x", padx=15, pady=15)
        
        ctk.CTkLabel(
            info_frame,
            text=f"👤 {self.user_info.get('full_name') or self.user_info['username']}",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=(20, 5))
        
        ctk.CTkLabel(
            info_frame,
            text=f"@{self.user_info['username']}",
            font=ctk.CTkFont(size=14),
            text_color=("#666666", "#999999")
        ).pack(pady=(0, 20))
        
        # Статистика профиля
        stats_grid = ctk.CTkFrame(profile_frame, fg_color="transparent")
        stats_grid.pack(fill="x", padx=15, pady=10)
        
        stats_grid.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.create_profile_stat(stats_grid, "Жесты", self.user_info['total_gestures'], 0)
        self.create_profile_stat(stats_grid, "Сессии", self.user_info['total_sessions'], 1)
        self.create_profile_stat(stats_grid, "Уровень", self.user_info['user_level'], 2)
    
    def create_training_tab(self):
        """Создаёт вкладку обучения."""
        tab = self.tabview.tab("🎓 Обучение")
        
        # Основной контейнер
        main_container = ctk.CTkFrame(tab)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Левая панель: камера и управление
        left_panel = ctk.CTkFrame(main_container, width=700)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Заголовок
        ctk.CTkLabel(
            left_panel,
            text="📹 Запись жестов для обучения",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=10)
        
        # Видео (заглушка)
        video_info = ctk.CTkFrame(left_panel, height=400)
        video_info.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            video_info,
            text="📹\n\nИспользуйте вкладку '🎥 Распознавание'\nи нажмите '▶ Старт'\n\nЗатем возвращайтесь сюда для записи",
            font=ctk.CTkFont(size=14),
            text_color=("#666666", "#888888")
        ).pack(expand=True, pady=150)
        
        # Управление записью
        control_frame = ctk.CTkFrame(left_panel)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Поле ввода имени жеста
        input_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        input_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            input_frame,
            text="Название жеста:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        self.gesture_name_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Peace, OK, ThumbsUp...",
            width=200
        )
        self.gesture_name_entry.pack(side="left", padx=5)
        
        # Кнопки управления
        buttons_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", pady=10)
        
        self.record_training_button = ctk.CTkButton(
            buttons_frame,
            text="📹 Записать образцы (5 сек)",
            command=self.start_recording_training_samples,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        self.record_training_button.pack(side="left", padx=5, expand=True, fill="x")
        
        # Статус записи
        self.recording_status_label = ctk.CTkLabel(
            control_frame,
            text="Готов к записи",
            font=ctk.CTkFont(size=12),
            text_color=("#666666", "#888888")
        )
        self.recording_status_label.pack(pady=5)
        
        # Правая панель: список жестов и обучение
        right_panel = ctk.CTkFrame(main_container, width=500)
        right_panel.pack(side="right", fill="both", padx=(5, 0))
        
        # Заголовок
        ctk.CTkLabel(
            right_panel,
            text="📊 Собранные данные",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Статистика
        stats_frame = ctk.CTkFrame(right_panel)
        stats_frame.pack(fill="x", padx=10, pady=5)
        
        stats_grid = ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_grid.pack(fill="x", padx=10, pady=10)
        
        self.training_stats_labels = {}
        
        stats_items = [
            ("Жестов:", "total_gestures", "0"),
            ("Образцов:", "total_samples", "0"),
            ("Статус:", "status", "Нет данных")
        ]
        
        for i, (label, key, default) in enumerate(stats_items):
            ctk.CTkLabel(
                stats_grid,
                text=label,
                font=ctk.CTkFont(size=11)
            ).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            
            value_label = ctk.CTkLabel(
                stats_grid,
                text=default,
                font=ctk.CTkFont(size=11, weight="bold")
            )
            value_label.grid(row=i, column=1, sticky="e", padx=5, pady=2)
            self.training_stats_labels[key] = value_label
        
        # Список жестов
        ctk.CTkLabel(
            right_panel,
            text="Список жестов:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(padx=10, pady=(10, 5), anchor="w")
        
        self.gestures_list_frame = ctk.CTkScrollableFrame(
            right_panel,
            height=300
        )
        self.gestures_list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Обучение
        training_control_frame = ctk.CTkFrame(right_panel)
        training_control_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            training_control_frame,
            text="⚙️ Обучение модели",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Прогресс
        self.training_progress_label = ctk.CTkLabel(
            training_control_frame,
            text="Готово к обучению",
            font=ctk.CTkFont(size=11)
        )
        self.training_progress_label.pack(pady=5)
        
        self.training_progress_bar = ctk.CTkProgressBar(training_control_frame)
        self.training_progress_bar.pack(fill="x", padx=10, pady=5)
        self.training_progress_bar.set(0)
        
        # Кнопки обучения
        train_buttons_frame = ctk.CTkFrame(training_control_frame, fg_color="transparent")
        train_buttons_frame.pack(fill="x", pady=10)
        
        # Проверяем доступность обучения перед созданием кнопки
        if not TRAINING_AVAILABLE:
            print("⚠️ ВНИМАНИЕ: TRAINING_AVAILABLE = False, кнопка обучения не будет работать!")
        
        # Создаём тестовую функцию для проверки клика
        def test_button_click():
            print("🔘 ТЕСТ: Кнопка нажата! Метод вызывается!")
            messagebox.showinfo("Тест", "Кнопка работает! Метод вызывается!")
        
        # Создаём кнопку с проверкой
        self.train_model_button = ctk.CTkButton(
            train_buttons_frame,
            text="🚀 Начать обучение",
            command=self.start_model_training,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            fg_color="#1a73e8",
            state="normal"  # Явно устанавливаем нормальное состояние
        )
        self.train_model_button.pack(side="left", padx=5, expand=True, fill="x")
        
        # ДОПОЛНИТЕЛЬНАЯ привязка через bind (на случай если command не работает)
        def on_button_click(event=None):
            print("🔘 КНОПКА НАЖАТА ЧЕРЕЗ BIND!")
            self.start_model_training()
        
        # Привязываем через bind тоже
        self.train_model_button.bind("<Button-1>", on_button_click)
        
        # Тестовая кнопка для проверки (временно)
        test_btn = ctk.CTkButton(
            train_buttons_frame,
            text="🔍 Тест",
            command=test_button_click,
            font=ctk.CTkFont(size=12),
            height=40,
            fg_color="#ff6b6b"
        )
        test_btn.pack(side="left", padx=5)
        
        print("✅ Кнопка 'Начать обучение' создана и привязана к методу start_model_training")
        print(f"   - Command: {self.train_model_button.cget('command')}")
        print(f"   - State: {self.train_model_button.cget('state')}")
        
        save_button = ctk.CTkButton(
            train_buttons_frame,
            text="💾 Сохранить",
            command=self.save_training_data,
            font=ctk.CTkFont(size=12),
            height=40
        )
        save_button.pack(side="left", padx=5)
        
        # Обновляем статистику при открытии
        self.update_training_statistics()
    
    # ===== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ GUI =====
    
    def create_video_panel(self, parent):
        """Создает панель с видео."""
        video_frame = ctk.CTkFrame(parent, corner_radius=10)
        video_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.video_canvas = ctk.CTkCanvas(
            video_frame,
            bg="#000000",
            highlightthickness=0
        )
        self.video_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.info_overlay = ctk.CTkLabel(
            video_frame,
            text="Нажмите 'Старт' для начала",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.info_overlay.place(relx=0.5, rely=0.5, anchor="center")
    
    def create_control_panel(self, parent):
        """Создает панель управления."""
        control_frame = ctk.CTkFrame(parent, height=70)
        control_frame.pack(fill="x", pady=(0, 10))
        control_frame.pack_propagate(False)
        
        button_container = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_container.pack(expand=True, fill="x", padx=20)
        
        self.start_button = ctk.CTkButton(
            button_container,
            text="▶ Старт",
            height=50,
            command=self.toggle_recognition
        )
        self.start_button.pack(side="left", expand=True, fill="x", padx=5)
        
        self.repeat_button = ctk.CTkButton(
            button_container,
            text="🔊 Повторить",
            height=50,
            state="disabled",
            command=self.repeat_gesture
        )
        self.repeat_button.pack(side="left", expand=True, fill="x", padx=5)
    
    def create_status_panel(self, parent):
        """Создает панель статуса."""
        status_frame = ctk.CTkFrame(parent, height=100)
        status_frame.pack(fill="x")
        status_frame.pack_propagate(False)
        
        status_grid = ctk.CTkFrame(status_frame, fg_color="transparent")
        status_grid.pack(expand=True, fill="both", padx=20, pady=15)
        status_grid.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Жест
        gesture_frame = ctk.CTkFrame(status_grid, corner_radius=10)
        gesture_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        
        ctk.CTkLabel(gesture_frame, text="Жест", font=ctk.CTkFont(size=11)).pack(pady=(10, 2))
        self.gesture_value = ctk.CTkLabel(gesture_frame, text="—", font=ctk.CTkFont(size=18, weight="bold"))
        self.gesture_value.pack(pady=(0, 10))
        
        # Уверенность
        conf_frame = ctk.CTkFrame(status_grid, corner_radius=10)
        conf_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        ctk.CTkLabel(conf_frame, text="Уверенность", font=ctk.CTkFont(size=11)).pack(pady=(10, 2))
        self.confidence_value = ctk.CTkLabel(conf_frame, text="—", font=ctk.CTkFont(size=18, weight="bold"))
        self.confidence_value.pack(pady=(0, 10))
        
        # Статус
        status_frame_inner = ctk.CTkFrame(status_grid, corner_radius=10)
        status_frame_inner.grid(row=0, column=2, sticky="nsew", padx=5)
        
        ctk.CTkLabel(status_frame_inner, text="Статус", font=ctk.CTkFont(size=11)).pack(pady=(10, 2))
        self.stability_value = ctk.CTkLabel(status_frame_inner, text="—", font=ctk.CTkFont(size=14, weight="bold"))
        self.stability_value.pack(pady=(0, 10))
    
    def create_history_panel(self, parent):
        """Создает панель истории."""
        history_frame = ctk.CTkFrame(parent, corner_radius=10)
        history_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        ctk.CTkLabel(
            history_frame,
            text="📜 История",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5), padx=10, anchor="w")
        
        self.history_scrollable = ctk.CTkScrollableFrame(history_frame)
        self.history_scrollable.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.history_widgets = []
    
    def create_chart_panel(self, parent):
        """Создает панель с графиком."""
        chart_frame = ctk.CTkFrame(parent, corner_radius=10, height=200)
        chart_frame.pack(fill="x")
        chart_frame.pack_propagate(False)
        
        ctk.CTkLabel(
            chart_frame,
            text="📊 График уверенности",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5), padx=10, anchor="w")
        
        self.chart_canvas = ctk.CTkCanvas(chart_frame, bg="#1a1a1a", highlightthickness=0)
        self.chart_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def create_stats_section(self, parent, title: str, data: dict):
        """Создает секцию статистики."""
        section_frame = ctk.CTkFrame(parent, corner_radius=10)
        section_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            section_frame,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        for key, value in data.items():
            row_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=15, pady=2)
            
            ctk.CTkLabel(
                row_frame,
                text=f"{key}:",
                font=ctk.CTkFont(size=12)
            ).pack(side="left")
            
            ctk.CTkLabel(
                row_frame,
                text=str(value),
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=("#42a5f5", "#42a5f5")
            ).pack(side="right")
        
        ctk.CTkLabel(section_frame, text="").pack(pady=5)
    
    def create_achievement_card(self, parent, achievement: dict):
        """Создает карточку достижения."""
        card = ctk.CTkFrame(parent, corner_radius=10)
        card.pack(fill="x", pady=5)
        
        is_completed = achievement.get('is_completed', False)
        
        # Иконка и название
        header_frame = ctk.CTkFrame(card, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            header_frame,
            text=f"{achievement['icon']} {achievement['name']}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#4caf50" if is_completed else "#666666")
        ).pack(side="left")
        
        ctk.CTkLabel(
            header_frame,
            text=f"{achievement['points']} 💎",
            font=ctk.CTkFont(size=12)
        ).pack(side="right")
        
        # Описание
        ctk.CTkLabel(
            card,
            text=achievement['description'],
            font=ctk.CTkFont(size=11),
            text_color=("#999999", "#666666")
        ).pack(anchor="w", padx=15, pady=(0, 10))
        
        # Прогресс
        progress = achievement.get('progress', 0) or 0  # Обработка None
        requirement = achievement['requirement_value']
        
        progress_frame = ctk.CTkFrame(card, fg_color="transparent")
        progress_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        progress_bar = ctk.CTkProgressBar(progress_frame)
        progress_bar.pack(fill="x")
        progress_bar.set(min(progress / requirement, 1.0) if requirement > 0 else 0)
        
        ctk.CTkLabel(
            progress_frame,
            text=f"{progress}/{requirement}",
            font=ctk.CTkFont(size=10)
        ).pack(anchor="e", pady=(2, 0))
    
    def create_profile_stat(self, parent, label: str, value: int, column: int):
        """Создает статистику профиля."""
        stat_frame = ctk.CTkFrame(parent, corner_radius=10)
        stat_frame.grid(row=0, column=column, sticky="nsew", padx=5)
        
        ctk.CTkLabel(
            stat_frame,
            text=str(value),
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=("#42a5f5", "#42a5f5")
        ).pack(pady=(20, 5))
        
        ctk.CTkLabel(
            stat_frame,
            text=label,
            font=ctk.CTkFont(size=12),
            text_color=("#666666", "#999999")
        ).pack(pady=(0, 20))
    
    # ===== МЕТОДЫ УПРАВЛЕНИЯ =====
    
    def toggle_recognition(self):
        """Переключает распознавание."""
        if not self.is_running:
            self.start_recognition()
        else:
            self.stop_recognition()
    
    def start_recognition(self):
        """Запускает распознавание."""
        if not self.camera.open():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру")
            return
        
        self.is_running = True
        self.start_button.configure(text="⏸ Стоп")
        self.repeat_button.configure(state="normal")
        self.info_overlay.place_forget()
        
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
    
    def stop_recognition(self):
        """Останавливает распознавание."""
        self.is_running = False
        self.start_button.configure(text="▶ Старт")
        
        if self.camera:
            self.camera.release()
        
        self.info_overlay.configure(text="Нажмите 'Старт' для начала")
        self.info_overlay.place(relx=0.5, rely=0.5, anchor="center")
    
    def process_video(self):
        """Обработка видеопотока."""
        import time
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Обнаружение
            hands_data, annotated_frame = self.gesture_detector.detect(frame)
            
            # Берем первую найденную руку (безопасно для numpy массивов)
            landmarks = hands_data.get('left')
            if landmarks is None:
                landmarks = hands_data.get('right')
            
            if landmarks is not None:
                try:
                    normalized_landmarks = self.gesture_model.normalize_landmarks(landmarks)
                    
                    # Запись для обучения
                    if TRAINING_AVAILABLE and self.is_recording_for_training:
                        self.recording_buffer.append(normalized_landmarks.copy())
                    
                    gesture, confidence = self.gesture_model.predict(normalized_landmarks)
                    
                    if gesture == self.current_gesture:
                        self.gesture_stable_count += 1
                    else:
                        self.current_gesture = gesture
                        self.gesture_stable_count = 1
                    
                    self.gesture_confidence = confidence
                    
                    if (self.gesture_stable_count >= self.stability_threshold and
                        gesture != self.last_gesture and
                        confidence >= self.preferences.get('min_confidence', 0.5)):
                        self.last_gesture = gesture
                        
                        # GUI обновление через главный поток
                        self.root.after(0, lambda g=gesture, c=confidence: self.add_to_history(g, c))
                        
                        # Сохраняем в БД (в фоновом потоке - БД thread-safe)
                        self.db.add_gesture(self.user_id, self.session_id, gesture, confidence)
                        
                        # Проверяем достижения каждые 10 жестов для производительности
                        self.gesture_count_since_achievement_check += 1
                        if self.gesture_count_since_achievement_check >= 10:
                            self.gesture_count_since_achievement_check = 0
                            # Проверка в отдельном потоке чтобы не блокировать
                            threading.Thread(
                                target=self.db.check_achievements,
                                args=(self.user_id,),
                                daemon=True
                            ).start()
                        
                        if self.tts and self.preferences.get('auto_speak', True):
                            # TTS в отдельном потоке чтобы не блокировать
                            threading.Thread(target=self.tts.speak, args=(gesture,), daemon=True).start()
                    
                    # Обновления GUI через главный поток
                    self.root.after(0, lambda g=gesture, c=confidence: self.update_status_display(g, c))
                    self.confidence_history.append(confidence)
                    self.root.after(0, self.update_chart)
                    
                except Exception as e:
                    print(f"Ошибка обработки: {e}")
            else:
                self.current_gesture = None
                self.gesture_stable_count = 0
                self.root.after(0, lambda: self.update_status_display(None, 0.0))
            
            # Обновление кадра через главный поток
            self.root.after(0, lambda f=annotated_frame: self.display_frame(f))
            
            # Небольшая задержка для разгрузки CPU
            time.sleep(0.01)
    
    def display_frame(self, frame):
        """Отображает кадр."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_canvas.delete("all")
        self.video_canvas.create_image(320, 240, image=imgtk)
        self.video_canvas.imgtk = imgtk
    
    def update_status_display(self, gesture, confidence):
        """Обновляет отображение статуса."""
        if gesture:
            self.gesture_value.configure(text=gesture)
            self.confidence_value.configure(text=f"{confidence:.1%}")
            
            if self.gesture_stable_count >= self.stability_threshold:
                self.stability_value.configure(text="✓ Стабильно")
            else:
                self.stability_value.configure(text=f"⏳ {self.gesture_stable_count}/{self.stability_threshold}")
        else:
            self.gesture_value.configure(text="—")
            self.confidence_value.configure(text="—")
            self.stability_value.configure(text="Покажите жест")
    
    def update_chart(self):
        """Обновляет график."""
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
        
        points = list(self.confidence_history)
        step = chart_width / (len(points) - 1) if len(points) > 1 else 0
        
        coords = []
        for i, conf in enumerate(points):
            x = padding + i * step
            y = padding + chart_height * (1 - conf)
            coords.extend([x, y])
        
        if len(coords) >= 4:
            self.chart_canvas.create_line(coords, fill='#42a5f5', width=2, smooth=True)
    
    def add_to_history(self, gesture, confidence):
        """Добавляет в историю."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{timestamp} - {gesture} ({confidence:.1%})"
        
        self.gesture_history.append(entry)
        
        entry_frame = ctk.CTkFrame(self.history_scrollable, corner_radius=5, height=30)
        entry_frame.pack(fill="x", pady=2)
        entry_frame.pack_propagate(False)
        
        ctk.CTkLabel(
            entry_frame,
            text=entry,
            font=ctk.CTkFont(size=10)
        ).pack(pady=5, padx=10, anchor="w")
        
        self.history_widgets.insert(0, entry_frame)
        
        if len(self.history_widgets) > 50:
            old_widget = self.history_widgets.pop()
            old_widget.destroy()
    
    def repeat_gesture(self):
        """Повторяет последний жест."""
        if self.last_gesture and self.tts:
            self.tts.speak(self.last_gesture, force=True)
    
    # ===== МЕТОДЫ ОБУЧЕНИЯ =====
    
    def start_recording_training_samples(self):
        """Начинает запись образцов для обучения."""
        if not TRAINING_AVAILABLE:
            messagebox.showwarning("Ошибка", "Модули обучения недоступны")
            return
        
        gesture_name = self.gesture_name_entry.get().strip()
        
        if not gesture_name:
            messagebox.showwarning("Ошибка", "Введите название жеста!")
            return
        
        if not self.camera or not self.is_running:
            messagebox.showwarning(
                "Ошибка", 
                "Сначала запустите распознавание на вкладке '🎥 Распознавание'!"
            )
            return
        
        self.current_training_gesture = gesture_name
        self.is_recording_for_training = True
        self.recording_buffer = []
        self.recording_start_time = time.time()
        
        self.record_training_button.configure(
            text="⏺️ Запись... (держите жест!)",
            state="disabled"
        )
        self.recording_status_label.configure(
            text=f"Записываем '{gesture_name}'...",
            text_color=("#ff0000", "#ff0000")
        )
        
        # Автоматическая остановка через 5 секунд
        self.root.after(5000, self.stop_recording_training_samples)
    
    def stop_recording_training_samples(self):
        """Останавливает запись образцов."""
        if not self.is_recording_for_training:
            return
        
        self.is_recording_for_training = False
        
        # Сохраняем все записанные кадры
        saved_count = 0
        for landmarks in self.recording_buffer:
            try:
                self.data_collector.add_sample(
                    self.current_training_gesture,
                    landmarks
                )
                saved_count += 1
            except Exception as e:
                print(f"Ошибка сохранения: {e}")
        
        self.record_training_button.configure(
            text="📹 Записать образцы (5 сек)",
            state="normal"
        )
        self.recording_status_label.configure(
            text=f"✓ Сохранено {saved_count} образцов для '{self.current_training_gesture}'",
            text_color=("#00ff00", "#00ff00")
        )
        
        # Обновляем статистику
        self.update_training_statistics()
        
        # Очищаем буфер
        self.recording_buffer = []
        self.current_training_gesture = ""
    
    def update_training_statistics(self):
        """Обновляет статистику обучения."""
        if not TRAINING_AVAILABLE:
            return
        
        gestures = self.data_collector.get_gesture_names()
        total_samples = self.data_collector.get_samples_count()
        
        self.training_stats_labels['total_gestures'].configure(text=str(len(gestures)))
        self.training_stats_labels['total_samples'].configure(text=str(total_samples))
        
        if total_samples >= 30:
            status = "✓ Готово к обучению"
            color = ("#00ff00", "#00ff00")
        elif total_samples > 0:
            status = f"⚠️ Нужно ещё {30 - total_samples} образцов"
            color = ("#ffaa00", "#ffaa00")
        else:
            status = "Нет данных"
            color = ("#666666", "#888888")
        
        self.training_stats_labels['status'].configure(text=status, text_color=color)
        
        # Обновляем список жестов
        for widget in self.gestures_list_frame.winfo_children():
            widget.destroy()
        
        for gesture_name in gestures:
            count = self.data_collector.get_samples_count(gesture_name)
            
            gesture_frame = ctk.CTkFrame(self.gestures_list_frame, height=40)
            gesture_frame.pack(fill="x", pady=2)
            
            icon = "✅" if count >= 10 else "⚠️"
            
            ctk.CTkLabel(
                gesture_frame,
                text=f"{icon} {gesture_name}",
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(side="left", padx=10, pady=8)
            
            ctk.CTkLabel(
                gesture_frame,
                text=f"{count} образцов",
                font=ctk.CTkFont(size=11),
                text_color=("#666666", "#888888")
            ).pack(side="right", padx=10, pady=8)
    
    def start_model_training(self):
        """Начинает обучение модели."""
        print("\n" + "=" * 60)
        print("КНОПКА ОБУЧЕНИЯ НАЖАТА!")
        print("=" * 60)
        
        # Сразу показываем сообщение для проверки
        print("🔘 МЕТОД ВЫЗВАН! Кнопка работает!")
        
        if not TRAINING_AVAILABLE:
            print("❌ TRAINING_AVAILABLE = False")
            messagebox.showwarning("Ошибка", "Модули обучения недоступны")
            return
        
        print("✅ TRAINING_AVAILABLE = True")
        
        # Проверяем, что data_collector инициализирован
        if not hasattr(self, 'data_collector') or self.data_collector is None:
            print("❌ data_collector не инициализирован!")
            messagebox.showerror("Ошибка", "DataCollector не инициализирован. Перезапустите приложение.")
            return
        
        print("✅ data_collector инициализирован")
        
        try:
            total_samples = self.data_collector.get_samples_count()
            print(f"📊 Собрано образцов: {total_samples}")
        except Exception as e:
            print(f"❌ Ошибка при получении количества образцов: {e}")
            messagebox.showerror("Ошибка", f"Не удалось получить количество образцов:\n{e}")
            return
        
        if total_samples < 30:
            print(f"❌ Недостаточно данных: {total_samples} < 30")
            messagebox.showwarning(
                "Недостаточно данных",
                f"Для обучения нужно минимум 30 образцов.\nСобрано: {total_samples}\n\n"
                "Рекомендация: 3+ жеста × 10+ образцов"
            )
            return
        
        print("✅ Данных достаточно, начинаем обучение...")
        
        try:
            print("📦 Подготовка данных...")
            # Подготовка данных
            X_train, X_test, y_train, y_test, gesture_classes = \
                self.data_collector.prepare_training_data()
            
            print(f"✅ Данные подготовлены:")
            print(f"   - Тренировочных примеров: {len(X_train)}")
            print(f"   - Тестовых примеров: {len(X_test)}")
            print(f"   - Классов жестов: {len(gesture_classes)}")
            print(f"   - Жесты: {gesture_classes}")
            
            print("🔧 Отключаем кнопку...")
            self.train_model_button.configure(state="disabled")
            self.training_progress_label.configure(text="Подготовка...")
            
            print("🚀 Запускаем поток обучения...")
            # Запуск обучения в отдельном потоке
            training_thread = threading.Thread(
                target=self._train_model_thread,
                args=(X_train, X_test, y_train, y_test, gesture_classes),
                daemon=True
            )
            training_thread.start()
            print("✅ Поток обучения запущен!")
            
        except Exception as e:
            print(f"❌ ОШИБКА при подготовке: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Ошибка", f"Не удалось начать обучение:\n{e}")
            self.train_model_button.configure(state="normal")
    
    def _train_model_thread(self, X_train, X_test, y_train, y_test, gesture_classes):
        """Обучение модели в отдельном потоке."""
        print("\n" + "=" * 60)
        print("🎓 ПОТОК ОБУЧЕНИЯ ЗАПУЩЕН")
        print("=" * 60)
        
        try:
            print("🧠 Создаём модель...")
            # Создаём модель
            model = AdvancedGestureClassifier(
                input_size=63,
                num_classes=len(gesture_classes)
            )
            print(f"✅ Модель создана: {len(gesture_classes)} классов")
            
            print("🎯 Создаём trainer...")
            # Создаём trainer
            self.model_trainer = ModelTrainer(model, device='cpu')
            print("✅ Trainer создан")
            
            print("📡 Добавляем callback...")
            # Добавляем callback
            self.model_trainer.add_callback(self._on_training_update)
            print("✅ Callback добавлен")
            
            print("📊 Создаём DataLoaders...")
            # Создаём DataLoaders
            train_dataset = GestureDataset(X_train, y_train)
            test_dataset = GestureDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16)
            print(f"✅ DataLoaders созданы")
            print(f"   - Train batches: {len(train_loader)}")
            print(f"   - Test batches: {len(test_loader)}")
            
            print("\n🚀 НАЧИНАЕМ ОБУЧЕНИЕ...")
            print("=" * 60)
            # Обучаем
            self.model_trainer.train(
                train_loader,
                test_loader,
                epochs=50,
                learning_rate=0.001,
                patience=10
            )
            
            print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print("💾 Сохраняем модель...")
            # Сохраняем модель
            self.root.after(0, lambda: self._on_training_complete(gesture_classes))
            
        except Exception as e:
            print(f"\n❌ ОШИБКА В ПОТОКЕ ОБУЧЕНИЯ: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror(
                "Ошибка",
                f"Ошибка при обучении:\n{e}"
            ))
            self.root.after(0, lambda: self.train_model_button.configure(state="normal"))
    
    def _on_training_update(self, event, data):
        """Callback обновления прогресса обучения."""
        def update_gui():
            if event == 'epoch_end':
                epoch = data['epoch']
                total = self.model_trainer.total_epochs
                val_acc = data['val_acc']
                
                progress = epoch / total
                self.training_progress_bar.set(progress)
                self.training_progress_label.configure(
                    text=f"Эпоха {epoch}/{total} | Точность: {val_acc:.1f}%"
                )
        
        self.root.after(0, update_gui)
    
    def _on_training_complete(self, gesture_classes):
        """Завершение обучения."""
        self.training_progress_bar.set(1.0)
        self.training_progress_label.configure(
            text=f"✓ Обучение завершено! Точность: {self.model_trainer.best_val_acc:.1f}%"
        )
        
        # Сохраняем модель
        try:
            from model.advanced_gesture_model import AdvancedGestureRecognizer
            
            recognizer = AdvancedGestureRecognizer()
            recognizer.model = self.model_trainer.model
            recognizer.gesture_classes = gesture_classes
            recognizer.save_model("models/trained_advanced_model.pth")
            
            messagebox.showinfo(
                "Успех!",
                f"Модель обучена!\n\n"
                f"Точность: {self.model_trainer.best_val_acc:.1f}%\n"
                f"Жестов: {len(gesture_classes)}\n\n"
                f"Сохранено: models/trained_advanced_model.pth\n\n"
                f"Перезапустите приложение для использования новой модели."
            )
        except Exception as e:
            print(f"Ошибка сохранения: {e}")
            import traceback
            traceback.print_exc()
        
        self.train_model_button.configure(state="normal")
    
    def save_training_data(self):
        """Сохраняет собранные данные."""
        if not TRAINING_AVAILABLE:
            return
        
        try:
            self.data_collector.save()
            messagebox.showinfo("Успех", "Данные сохранены!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")
    
    # ===== КОНЕЦ МЕТОДОВ ОБУЧЕНИЯ =====
    
    def change_theme(self, theme_name):
        """Изменяет тему."""
        self.theme_manager.set_theme(theme_name)
        self.preferences.set('theme', theme_name)
        messagebox.showinfo("Тема", "Тема будет применена при следующем запуске")
    
    def change_language(self, lang):
        """Изменяет язык."""
        self.localization.set_language(lang)
        self.preferences.set('language', lang)
        messagebox.showinfo("Язык", "Язык будет применен при следующем запуске")
    
    def reset_preferences(self):
        """Сбрасывает настройки."""
        if messagebox.askyesno("Подтверждение", "Сбросить все настройки к значениям по умолчанию?"):
            self.preferences.reset_to_defaults()
            messagebox.showinfo("Успех", "Настройки сброшены")
    
    def export_data(self):
        """Экспортирует данные пользователя."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            export_type = 'json' if filepath.endswith('.json') else 'csv'
            result_path = self.db.export_user_data(self.user_id, export_type)
            
            # Копируем файл в выбранное место
            import shutil
            shutil.copy(result_path, filepath)
            
            messagebox.showinfo("Успех", f"Данные экспортированы в {filepath}")
    
    def logout(self):
        """Выход из системы."""
        if messagebox.askyesno("Выход", "Вы уверены что хотите выйти?"):
            self.on_closing()
    
    def on_closing(self):
        """Обработчик закрытия."""
        self.is_running = False
        
        # Завершаем сессию
        self.db.end_session(self.session_id)
        
        if self.camera:
            self.camera.release()
        
        if self.gesture_detector:
            self.gesture_detector.close()
        
        if self.tts:
            self.tts.stop()
        
        self.root.destroy()
        
        # Перезапускаем окно авторизации
        login_window = LoginWindow(self.db, lambda uid: start_main_app(uid, self.db))
        login_window.run()


def start_main_app(user_id: int, db_manager: DatabaseManager):
    """Запускает главное приложение."""
    app = EnterpriseSignVoiceGUI(user_id, db_manager)
    app.root.mainloop()


def main():
    """Точка входа в приложение."""
    print("\n" + "=" * 70)
    print("SignVoiceAI Enterprise Edition - Запуск")
    print("=" * 70)
    
    print("\n[1/3] Инициализация базы данных...")
    # Инициализация БД
    db = DatabaseManager("signvoice_enterprise.db")
    print("✓ База данных готова")
    
    print("\n[2/3] Создание окна авторизации...")
    # Показываем окно авторизации
    login_window = LoginWindow(db, lambda user_id: start_main_app(user_id, db))
    print("✓ Окно создано")
    
    print("\n[3/3] Запуск GUI...")
    print("\n" + "=" * 70)
    print("ПРОВЕРЬТЕ ЭКРАН - окно должно появиться!")
    print("Если не видите окно, проверьте Alt+Tab или панель задач")
    print("=" * 70 + "\n")
    
    login_window.run()


if __name__ == "__main__":
    main()


