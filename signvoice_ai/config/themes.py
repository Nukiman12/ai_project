"""
Модуль тем и стилей для SignVoiceAI Enterprise.

Предоставляет различные цветовые схемы и настройки интерфейса.
"""

from typing import Dict, Any
from dataclasses import dataclass
import json


@dataclass
class ColorScheme:
    """Цветовая схема приложения."""
    primary: str
    secondary: str
    background: str
    surface: str
    text_primary: str
    text_secondary: str
    success: str
    warning: str
    error: str
    info: str
    accent: str
    
    def to_dict(self) -> Dict[str, str]:
        """Преобразует в словарь."""
        return {
            'primary': self.primary,
            'secondary': self.secondary,
            'background': self.background,
            'surface': self.surface,
            'text_primary': self.text_primary,
            'text_secondary': self.text_secondary,
            'success': self.success,
            'warning': self.warning,
            'error': self.error,
            'info': self.info,
            'accent': self.accent
        }


# ===== ПРЕДУСТАНОВЛЕННЫЕ ТЕМЫ =====

DARK_THEME = ColorScheme(
    primary='#1976d2',
    secondary='#424242',
    background='#121212',
    surface='#1e1e1e',
    text_primary='#ffffff',
    text_secondary='#b0b0b0',
    success='#4caf50',
    warning='#ff9800',
    error='#f44336',
    info='#2196f3',
    accent='#42a5f5'
)

LIGHT_THEME = ColorScheme(
    primary='#1976d2',
    secondary='#f5f5f5',
    background='#fafafa',
    surface='#ffffff',
    text_primary='#212121',
    text_secondary='#757575',
    success='#4caf50',
    warning='#ff9800',
    error='#f44336',
    info='#2196f3',
    accent='#1976d2'
)

BLUE_THEME = ColorScheme(
    primary='#0d47a1',
    secondary='#1565c0',
    background='#0a1929',
    surface='#132f4c',
    text_primary='#e3f2fd',
    text_secondary='#90caf9',
    success='#66bb6a',
    warning='#ffa726',
    error='#ef5350',
    info='#42a5f5',
    accent='#64b5f6'
)

GREEN_THEME = ColorScheme(
    primary='#2e7d32',
    secondary='#388e3c',
    background='#1b5e20',
    surface='#2e7d32',
    text_primary='#e8f5e9',
    text_secondary='#a5d6a7',
    success='#66bb6a',
    warning='#ffb74d',
    error='#e57373',
    info='#4fc3f7',
    accent='#81c784'
)

PURPLE_THEME = ColorScheme(
    primary='#6a1b9a',
    secondary='#7b1fa2',
    background='#4a148c',
    surface='#6a1b9a',
    text_primary='#f3e5f5',
    text_secondary='#ce93d8',
    success='#66bb6a',
    warning='#ffb74d',
    error='#e57373',
    info='#ba68c8',
    accent='#ab47bc'
)

ORANGE_THEME = ColorScheme(
    primary='#e65100',
    secondary='#ef6c00',
    background='#bf360c',
    surface='#d84315',
    text_primary='#fff3e0',
    text_secondary='#ffcc80',
    success='#66bb6a',
    warning='#ffa726',
    error='#e57373',
    info='#42a5f5',
    accent='#ff9800'
)

CYBER_THEME = ColorScheme(
    primary='#00bcd4',
    secondary='#00acc1',
    background='#000000',
    surface='#0d1117',
    text_primary='#00ffff',
    text_secondary='#00e5ff',
    success='#00ff41',
    warning='#ffea00',
    error='#ff1744',
    info='#00e5ff',
    accent='#00ffff'
)

FOREST_THEME = ColorScheme(
    primary='#33691e',
    secondary='#558b2f',
    background='#1b5e20',
    surface='#2e7d32',
    text_primary='#f1f8e9',
    text_secondary='#c5e1a5',
    success='#8bc34a',
    warning='#ffc107',
    error='#e57373',
    info='#4dd0e1',
    accent='#9ccc65'
)

SUNSET_THEME = ColorScheme(
    primary='#d84315',
    secondary='#ff5722',
    background='#bf360c',
    surface='#d84315',
    text_primary='#fff8e1',
    text_secondary='#ffcc80',
    success='#66bb6a',
    warning='#ffb74d',
    error='#ef5350',
    info='#42a5f5',
    accent='#ff7043'
)

OCEAN_THEME = ColorScheme(
    primary='#006064',
    secondary='#00838f',
    background='#004d40',
    surface='#00695c',
    text_primary='#e0f7fa',
    text_secondary='#80deea',
    success='#4db6ac',
    warning='#ffb74d',
    error='#e57373',
    info='#4dd0e1',
    accent='#26c6da'
)

# Словарь всех тем
THEMES = {
    'dark': DARK_THEME,
    'light': LIGHT_THEME,
    'blue': BLUE_THEME,
    'green': GREEN_THEME,
    'purple': PURPLE_THEME,
    'orange': ORANGE_THEME,
    'cyber': CYBER_THEME,
    'forest': FOREST_THEME,
    'sunset': SUNSET_THEME,
    'ocean': OCEAN_THEME
}


class ThemeManager:
    """
    Менеджер тем приложения.
    """
    
    def __init__(self, default_theme: str = 'dark'):
        """
        Инициализация менеджера тем.
        
        Args:
            default_theme: Тема по умолчанию
        """
        self.current_theme_name = default_theme
        self.current_theme = THEMES.get(default_theme, DARK_THEME)
        self.custom_themes = {}
    
    def get_theme(self, theme_name: str = None) -> ColorScheme:
        """
        Получает тему по имени.
        
        Args:
            theme_name: Название темы
            
        Returns:
            Цветовая схема
        """
        if theme_name is None:
            return self.current_theme
        
        # Проверяем в предустановленных темах
        if theme_name in THEMES:
            return THEMES[theme_name]
        
        # Проверяем в пользовательских темах
        if theme_name in self.custom_themes:
            return self.custom_themes[theme_name]
        
        return DARK_THEME
    
    def set_theme(self, theme_name: str):
        """
        Устанавливает активную тему.
        
        Args:
            theme_name: Название темы
        """
        theme = self.get_theme(theme_name)
        if theme:
            self.current_theme_name = theme_name
            self.current_theme = theme
    
    def get_available_themes(self) -> list:
        """
        Получает список доступных тем.
        
        Returns:
            Список названий тем
        """
        return list(THEMES.keys()) + list(self.custom_themes.keys())
    
    def create_custom_theme(self, name: str, colors: Dict[str, str]) -> ColorScheme:
        """
        Создает пользовательскую тему.
        
        Args:
            name: Название темы
            colors: Словарь с цветами
            
        Returns:
            Созданная цветовая схема
        """
        theme = ColorScheme(**colors)
        self.custom_themes[name] = theme
        return theme
    
    def export_theme(self, theme_name: str, filepath: str):
        """
        Экспортирует тему в файл.
        
        Args:
            theme_name: Название темы
            filepath: Путь к файлу
        """
        theme = self.get_theme(theme_name)
        if theme:
            with open(filepath, 'w') as f:
                json.dump(theme.to_dict(), f, indent=2)
    
    def import_theme(self, filepath: str, theme_name: str = None) -> str:
        """
        Импортирует тему из файла.
        
        Args:
            filepath: Путь к файлу
            theme_name: Название для темы (опционально)
            
        Returns:
            Название импортированной темы
        """
        with open(filepath, 'r') as f:
            colors = json.load(f)
        
        if theme_name is None:
            import os
            theme_name = os.path.splitext(os.path.basename(filepath))[0]
        
        self.create_custom_theme(theme_name, colors)
        return theme_name


class UserPreferences:
    """
    Класс для управления пользовательскими настройками.
    """
    
    DEFAULT_PREFERENCES = {
        # Интерфейс
        'theme': 'dark',
        'language': 'ru',
        'font_size': 12,
        'animations_enabled': True,
        'show_tooltips': True,
        'compact_mode': False,
        
        # Распознавание
        'stability_threshold': 5,
        'min_confidence': 0.5,
        'auto_speak': True,
        'hand_preference': 'auto',  # auto, left, right
        'detect_both_hands': True,
        
        # Звук и речь
        'speech_enabled': True,
        'speech_rate': 150,
        'speech_volume': 0.8,
        'speech_voice': 'default',
        'sound_effects': True,
        'notification_sounds': True,
        
        # Камера
        'camera_index': 0,
        'camera_resolution': '640x480',
        'mirror_video': False,
        'show_landmarks': True,
        'show_fps': False,
        
        # История и данные
        'save_history': True,
        'history_limit': 1000,
        'auto_export': False,
        'export_format': 'json',
        
        # Уведомления
        'notifications_enabled': True,
        'achievement_notifications': True,
        'session_reminders': False,
        'daily_goal': 50,
        
        # Продвинутые
        'debug_mode': False,
        'log_level': 'INFO',
        'auto_update': True,
        'telemetry_enabled': False,
        
        # Доступность
        'high_contrast': False,
        'large_ui': False,
        'screen_reader': False,
        'keyboard_shortcuts': True
    }
    
    def __init__(self, db_manager=None, user_id: int = None):
        """
        Инициализация пользовательских настроек.
        
        Args:
            db_manager: Менеджер базы данных
            user_id: ID пользователя
        """
        self.db = db_manager
        self.user_id = user_id
        self.preferences = self.DEFAULT_PREFERENCES.copy()
        
        if self.db and self.user_id:
            self.load_from_db()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получает значение настройки.
        
        Args:
            key: Ключ настройки
            default: Значение по умолчанию
            
        Returns:
            Значение настройки
        """
        if self.db and self.user_id:
            # Загружаем из БД
            value = self.db.get_setting(self.user_id, key)
            if value is not None:
                return value
        
        return self.preferences.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Устанавливает значение настройки.
        
        Args:
            key: Ключ настройки
            value: Значение
        """
        self.preferences[key] = value
        
        if self.db and self.user_id:
            # Сохраняем в БД
            self.db.set_setting(self.user_id, key, value)
    
    def update(self, **kwargs):
        """
        Обновляет несколько настроек сразу.
        
        Args:
            **kwargs: Пары ключ-значение
        """
        for key, value in kwargs.items():
            self.set(key, value)
    
    def load_from_db(self):
        """Загружает все настройки из БД."""
        if not self.db or not self.user_id:
            return
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT setting_key, setting_value FROM user_settings
        WHERE user_id = ?
        ''', (self.user_id,))
        
        for row in cursor.fetchall():
            key = row['setting_key']
            value = row['setting_value']
            try:
                self.preferences[key] = json.loads(value)
            except:
                self.preferences[key] = value
        
        conn.close()
    
    def reset_to_defaults(self):
        """Сбрасывает настройки к значениям по умолчанию."""
        self.preferences = self.DEFAULT_PREFERENCES.copy()
        
        if self.db and self.user_id:
            # Удаляем все настройки из БД
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
            DELETE FROM user_settings WHERE user_id = ?
            ''', (self.user_id,))
            conn.commit()
            conn.close()
    
    def export_preferences(self, filepath: str):
        """
        Экспортирует настройки в файл.
        
        Args:
            filepath: Путь к файлу
        """
        with open(filepath, 'w') as f:
            json.dump(self.preferences, f, indent=2)
    
    def import_preferences(self, filepath: str):
        """
        Импортирует настройки из файла.
        
        Args:
            filepath: Путь к файлу
        """
        with open(filepath, 'r') as f:
            imported = json.load(f)
        
        self.update(**imported)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует настройки в словарь.
        
        Returns:
            Словарь настроек
        """
        return self.preferences.copy()


# ===== ЛОКАЛИЗАЦИЯ =====

TRANSLATIONS = {
    'ru': {
        'app_title': 'SignVoiceAI - Распознавание жестов',
        'start': 'Старт',
        'stop': 'Стоп',
        'repeat': 'Повторить',
        'clear': 'Очистить',
        'settings': 'Настройки',
        'profile': 'Профиль',
        'statistics': 'Статистика',
        'achievements': 'Достижения',
        'history': 'История',
        'export': 'Экспорт',
        'import': 'Импорт',
        'logout': 'Выход',
        'current_gesture': 'Текущий жест',
        'confidence': 'Уверенность',
        'status': 'Статус',
        'running': 'Работает',
        'stopped': 'Остановлено',
        'show_gesture': 'Покажите жест',
        'stable': 'Стабильно',
        'detecting': 'Определение...',
    },
    'en': {
        'app_title': 'SignVoiceAI - Gesture Recognition',
        'start': 'Start',
        'stop': 'Stop',
        'repeat': 'Repeat',
        'clear': 'Clear',
        'settings': 'Settings',
        'profile': 'Profile',
        'statistics': 'Statistics',
        'achievements': 'Achievements',
        'history': 'History',
        'export': 'Export',
        'import': 'Import',
        'logout': 'Logout',
        'current_gesture': 'Current Gesture',
        'confidence': 'Confidence',
        'status': 'Status',
        'running': 'Running',
        'stopped': 'Stopped',
        'show_gesture': 'Show gesture',
        'stable': 'Stable',
        'detecting': 'Detecting...',
    }
}


class LocalizationManager:
    """
    Менеджер локализации приложения.
    """
    
    def __init__(self, language: str = 'ru'):
        """
        Инициализация менеджера локализации.
        
        Args:
            language: Код языка
        """
        self.language = language
        self.translations = TRANSLATIONS.get(language, TRANSLATIONS['ru'])
    
    def set_language(self, language: str):
        """
        Устанавливает язык.
        
        Args:
            language: Код языка
        """
        self.language = language
        self.translations = TRANSLATIONS.get(language, TRANSLATIONS['ru'])
    
    def get(self, key: str, default: str = None) -> str:
        """
        Получает перевод по ключу.
        
        Args:
            key: Ключ перевода
            default: Значение по умолчанию
            
        Returns:
            Переведенная строка
        """
        return self.translations.get(key, default or key)
    
    def __getitem__(self, key: str) -> str:
        """Позволяет использовать локализацию как словарь."""
        return self.get(key)




