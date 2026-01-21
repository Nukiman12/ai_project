# 📚 SignVoiceAI API Reference

> Полная документация API для разработчиков SignVoiceAI Enterprise

---

## 🗂️ Содержание

- [Database API](#database-api)
- [Analytics API](#analytics-api)
- [Configuration API](#configuration-api)
- [Logger API](#logger-api)
- [Model API](#model-api)
- [Utils API](#utils-api)

---

## Database API

### `DatabaseManager`

Класс для управления базой данных SQLite.

#### Инициализация

```python
from database.db_manager import DatabaseManager

db = DatabaseManager(db_path="signvoice_data.db")
```

**Параметры:**
- `db_path` (str): Путь к файлу базы данных

#### Методы управления пользователями

##### `create_user(username, password, email=None, full_name=None)`

Создает нового пользователя.

**Параметры:**
- `username` (str): Имя пользователя (уникальное)
- `password` (str): Пароль (будет захеширован)
- `email` (str, optional): Email пользователя
- `full_name` (str, optional): Полное имя

**Возвращает:**
- `int | None`: ID созданного пользователя или None при ошибке

**Пример:**
```python
user_id = db.create_user("john_doe", "secure_password", "john@example.com", "John Doe")
if user_id:
    print(f"User created with ID: {user_id}")
```

##### `authenticate_user(username, password)`

Аутентифицирует пользователя.

**Параметры:**
- `username` (str): Имя пользователя
- `password` (str): Пароль

**Возвращает:**
- `int | None`: ID пользователя или None если аутентификация не удалась

**Пример:**
```python
user_id = db.authenticate_user("john_doe", "secure_password")
if user_id:
    print("Authentication successful!")
```

##### `get_user_info(user_id)`

Получает информацию о пользователе.

**Параметры:**
- `user_id` (int): ID пользователя

**Возвращает:**
- `dict | None`: Словарь с информацией о пользователе

**Структура ответа:**
```python
{
    'user_id': 1,
    'username': 'john_doe',
    'email': 'john@example.com',
    'full_name': 'John Doe',
    'created_at': '2024-01-01 12:00:00',
    'last_login': '2024-01-15 10:30:00',
    'is_active': True,
    'user_level': 5,
    'total_gestures': 150,
    'total_sessions': 10,
    'total_time_seconds': 3600,
    'bio': 'Learning sign language',
    'language': 'ru',
    'theme': 'dark',
    'preferred_voice': 'default',
    'notifications_enabled': True,
    'auto_speak': True
}
```

##### `update_user_profile(user_id, **kwargs)`

Обновляет профиль пользователя.

**Параметры:**
- `user_id` (int): ID пользователя
- `**kwargs`: Поля для обновления

**Доступные поля:**
- `full_name`, `email`, `avatar_path` (таблица users)
- `bio`, `language`, `theme`, `preferred_voice`, `notifications_enabled`, `auto_speak` (таблица profiles)

**Пример:**
```python
db.update_user_profile(
    user_id,
    full_name="John Smith",
    bio="Passionate about sign language",
    theme="blue",
    language="en"
)
```

#### Методы управления жестами

##### `add_gesture(user_id, session_id, gesture_name, confidence, hand_type="right", duration_ms=None)`

Добавляет распознанный жест в историю.

**Параметры:**
- `user_id` (int): ID пользователя
- `session_id` (str): ID сессии
- `gesture_name` (str): Название жеста
- `confidence` (float): Уверенность распознавания (0.0-1.0)
- `hand_type` (str): Тип руки ("left" или "right")
- `duration_ms` (int, optional): Длительность в миллисекундах

**Пример:**
```python
db.add_gesture(
    user_id=1,
    session_id="session_123",
    gesture_name="Hello",
    confidence=0.95,
    hand_type="right",
    duration_ms=1500
)
```

##### `get_user_statistics(user_id)`

Получает статистику пользователя.

**Параметры:**
- `user_id` (int): ID пользователя

**Возвращает:**
- `dict`: Словарь со статистикой

**Структура ответа:**
```python
{
    'user': {
        'total_gestures': 150,
        'total_sessions': 10,
        'total_time_seconds': 3600,
        'user_level': 5
    },
    'gestures': [
        {
            'gesture_name': 'Hello',
            'total_count': 50,
            'avg_confidence': 0.92,
            'best_confidence': 0.98
        },
        # ... другие жесты
    ],
    'recent_sessions': [
        {
            'session_id': 'session_123',
            'start_time': '2024-01-15 10:00:00',
            'end_time': '2024-01-15 10:15:00',
            'total_gestures': 20,
            'avg_confidence': 0.90,
            'duration_seconds': 900
        },
        # ... другие сессии
    ]
}
```

#### Методы управления сессиями

##### `start_session(user_id, session_id)`

Начинает новую сессию.

**Пример:**
```python
import uuid

session_id = str(uuid.uuid4())
db.start_session(user_id, session_id)
```

##### `end_session(session_id, notes=None)`

Завершает сессию.

**Параметры:**
- `session_id` (str): ID сессии
- `notes` (str, optional): Заметки о сессии

**Пример:**
```python
db.end_session(session_id, notes="Good practice session")
```

#### Методы работы с достижениями

##### `check_achievements(user_id)`

Проверяет и обновляет достижения пользователя.

**Параметры:**
- `user_id` (int): ID пользователя

**Пример:**
```python
db.check_achievements(user_id)
```

##### `get_user_achievements(user_id)`

Получает достижения пользователя.

**Параметры:**
- `user_id` (int): ID пользователя

**Возвращает:**
- `list[dict]`: Список достижений

**Структура ответа:**
```python
[
    {
        'achievement_id': 1,
        'name': 'Первые шаги',
        'description': 'Распознать первый жест',
        'icon': '🎯',
        'category': 'beginner',
        'requirement_type': 'gesture_count',
        'requirement_value': 1,
        'points': 10,
        'progress': 15,
        'is_completed': True,
        'unlocked_at': '2024-01-10 15:30:00'
    },
    # ... другие достижения
]
```

#### Методы работы с настройками

##### `get_setting(user_id, key, default=None)`

Получает настройку пользователя.

**Пример:**
```python
theme = db.get_setting(user_id, 'theme', 'dark')
```

##### `set_setting(user_id, key, value)`

Устанавливает настройку пользователя.

**Пример:**
```python
db.set_setting(user_id, 'theme', 'blue')
db.set_setting(user_id, 'speech_rate', 180)
```

#### Методы экспорта

##### `export_user_data(user_id, export_type='json')`

Экспортирует данные пользователя.

**Параметры:**
- `user_id` (int): ID пользователя
- `export_type` (str): Тип экспорта ('json' или 'csv')

**Возвращает:**
- `str`: Путь к файлу экспорта

**Пример:**
```python
filepath = db.export_user_data(user_id, 'json')
print(f"Data exported to: {filepath}")
```

---

## Analytics API

### `AnalyticsEngine`

Класс для анализа данных пользователя.

#### Инициализация

```python
from analytics.analytics_engine import AnalyticsEngine

analytics = AnalyticsEngine(db_manager)
```

**Параметры:**
- `db_manager`: Экземпляр DatabaseManager

#### Методы

##### `get_user_dashboard(user_id)`

Получает данные для панели управления.

**Возвращает:**
```python
{
    'overview': {
        'total_gestures': 150,
        'total_sessions': 10,
        'total_time': '1ч 0м',
        'level': 5,
        'avg_gestures_per_session': 15.0
    },
    'performance': {
        'avg_confidence': 0.92,
        'best_confidence': 0.98,
        'unique_gestures': 8,
        'avg_session_duration': 900,
        'consistency_score': 0.85,
        'progress_trend': 'improving',
        'performance_rating': 'excellent'
    },
    'trends': { ... },
    'top_gestures': [ ... ],
    'recent_activity': { ... },
    'recommendations': [
        '🏆 Отличная работа! Ваша точность распознавания очень высокая',
        '📈 Вы на правильном пути! Продолжайте регулярные занятия'
    ]
}
```

##### `analyze_performance(user_id)`

Анализирует производительность пользователя.

**Возвращает:**
```python
{
    'avg_confidence': 0.92,
    'best_confidence': 0.98,
    'unique_gestures': 8,
    'avg_session_duration': 900,
    'consistency_score': 0.85,
    'progress_trend': 'improving',  # 'improving', 'declining', 'stable'
    'performance_rating': 'excellent'  # 'excellent', 'very_good', 'good', 'fair', 'needs_improvement'
}
```

##### `analyze_trends(user_id, days=30)`

Анализирует тренды активности.

**Параметры:**
- `user_id` (int): ID пользователя
- `days` (int): Количество дней для анализа

**Возвращает:**
```python
{
    'daily_activity': [
        {'date': '2024-01-15', 'count': 20, 'avg_conf': 0.92},
        # ...
    ],
    'trend_direction': 'increasing',  # 'increasing', 'decreasing', 'stable'
    'peak_hours': [10, 14, 18],
    'most_active_day': '2024-01-15',
    'activity_consistency': 0.75
}
```

##### `get_top_gestures(user_id, limit=10)`

Получает топ жестов пользователя.

**Возвращает:**
```python
[
    {
        'gesture_name': 'Hello',
        'total_count': 50,
        'avg_confidence': 0.92,
        'best_confidence': 0.98,
        'last_used': '2024-01-15 10:30:00'
    },
    # ...
]
```

##### `generate_recommendations(user_id)`

Генерирует рекомендации для пользователя.

**Возвращает:**
```python
[
    '💡 Улучшите освещение и фон для повышения точности распознавания',
    '🏆 Отличная работа! Ваша точность распознавания очень высокая',
    # ...
]
```

##### `generate_report(user_id, period_days=30)`

Генерирует полный отчет.

**Параметры:**
- `user_id` (int): ID пользователя
- `period_days` (int): Период для отчета в днях

**Возвращает:**
```python
{
    'generated_at': '2024-01-15T10:30:00',
    'period_days': 30,
    'user_info': { ... },
    'summary': { ... },
    'performance': { ... },
    'trends': { ... },
    'achievements': { ... },
    'recommendations': [ ... ],
    'top_gestures': [ ... ]
}
```

---

## Configuration API

### `ThemeManager`

Класс для управления темами приложения.

#### Инициализация

```python
from config.themes import ThemeManager, THEMES

theme_manager = ThemeManager(default_theme='dark')
```

#### Методы

##### `get_theme(theme_name=None)`

Получает тему по имени.

**Возвращает:** `ColorScheme`

**Пример:**
```python
theme = theme_manager.get_theme('blue')
print(theme.primary)  # '#0d47a1'
```

##### `set_theme(theme_name)`

Устанавливает активную тему.

**Пример:**
```python
theme_manager.set_theme('cyber')
```

##### `get_available_themes()`

Получает список доступных тем.

**Возвращает:**
```python
['dark', 'light', 'blue', 'green', 'purple', 'orange', 'cyber', 'forest', 'sunset', 'ocean']
```

##### `create_custom_theme(name, colors)`

Создает пользовательскую тему.

**Параметры:**
- `name` (str): Название темы
- `colors` (dict): Словарь с цветами

**Пример:**
```python
custom_colors = {
    'primary': '#ff0000',
    'secondary': '#00ff00',
    'background': '#000000',
    'surface': '#1a1a1a',
    'text_primary': '#ffffff',
    'text_secondary': '#cccccc',
    'success': '#00ff00',
    'warning': '#ffff00',
    'error': '#ff0000',
    'info': '#00ffff',
    'accent': '#ff00ff'
}

theme_manager.create_custom_theme('my_theme', custom_colors)
```

##### `export_theme(theme_name, filepath)`

Экспортирует тему в файл.

**Пример:**
```python
theme_manager.export_theme('cyber', 'my_theme.json')
```

##### `import_theme(filepath, theme_name=None)`

Импортирует тему из файла.

**Пример:**
```python
theme_name = theme_manager.import_theme('my_theme.json')
```

### `UserPreferences`

Класс для управления пользовательскими настройками.

#### Инициализация

```python
from config.themes import UserPreferences

prefs = UserPreferences(db_manager, user_id)
```

#### Методы

##### `get(key, default=None)`

Получает значение настройки.

**Пример:**
```python
theme = prefs.get('theme', 'dark')
speech_rate = prefs.get('speech_rate', 150)
```

##### `set(key, value)`

Устанавливает значение настройки.

**Пример:**
```python
prefs.set('theme', 'blue')
prefs.set('speech_rate', 180)
```

##### `update(**kwargs)`

Обновляет несколько настроек сразу.

**Пример:**
```python
prefs.update(
    theme='blue',
    language='en',
    speech_rate=180,
    auto_speak=True
)
```

##### `reset_to_defaults()`

Сбрасывает настройки к значениям по умолчанию.

**Пример:**
```python
prefs.reset_to_defaults()
```

##### `export_preferences(filepath)`

Экспортирует настройки в файл.

**Пример:**
```python
prefs.export_preferences('my_settings.json')
```

##### `import_preferences(filepath)`

Импортирует настройки из файла.

**Пример:**
```python
prefs.import_preferences('my_settings.json')
```

##### `to_dict()`

Преобразует настройки в словарь.

**Возвращает:**
```python
{
    'theme': 'dark',
    'language': 'ru',
    'font_size': 12,
    'animations_enabled': True,
    # ... все настройки
}
```

#### Доступные настройки

**Интерфейс:**
- `theme` (str): Тема приложения
- `language` (str): Язык интерфейса
- `font_size` (int): Размер шрифта
- `animations_enabled` (bool): Включить анимации
- `show_tooltips` (bool): Показывать подсказки
- `compact_mode` (bool): Компактный режим

**Распознавание:**
- `stability_threshold` (int): Порог стабильности (1-10)
- `min_confidence` (float): Минимальная уверенность (0.0-1.0)
- `auto_speak` (bool): Автоматическое озвучивание
- `hand_preference` (str): Предпочтение руки ('auto', 'left', 'right')
- `detect_both_hands` (bool): Распознавание обеих рук

**Звук и речь:**
- `speech_enabled` (bool): Включить речь
- `speech_rate` (int): Скорость речи (100-200)
- `speech_volume` (float): Громкость (0.0-1.0)
- `speech_voice` (str): Голос
- `sound_effects` (bool): Звуковые эффекты
- `notification_sounds` (bool): Звуки уведомлений

**Камера:**
- `camera_index` (int): Индекс камеры
- `camera_resolution` (str): Разрешение ('640x480', '1280x720', etc.)
- `mirror_video` (bool): Зеркалирование видео
- `show_landmarks` (bool): Показывать точки
- `show_fps` (bool): Показывать FPS

### `LocalizationManager`

Класс для управления локализацией.

#### Инициализация

```python
from config.themes import LocalizationManager

loc = LocalizationManager(language='ru')
```

#### Методы

##### `set_language(language)`

Устанавливает язык.

**Пример:**
```python
loc.set_language('en')
```

##### `get(key, default=None)`

Получает перевод по ключу.

**Пример:**
```python
title = loc.get('app_title')  # 'SignVoiceAI - Распознавание жестов'
start = loc.get('start')      # 'Старт'
```

---

## Logger API

### `SignVoiceLogger`

Класс для логирования событий приложения.

#### Инициализация

```python
from utils.logger import SignVoiceLogger, get_logger

# Создание логгера
logger = SignVoiceLogger(
    name="MyApp",
    log_dir="logs",
    log_level="INFO",
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5
)

# Или использование глобального логгера
logger = get_logger()
```

#### Основные методы

##### `debug(message, **kwargs)`

Логирует debug сообщение.

**Пример:**
```python
logger.debug("Processing frame", frame_number=100)
```

##### `info(message, **kwargs)`

Логирует info сообщение.

**Пример:**
```python
logger.info("User logged in", user_id=1, username="john_doe")
```

##### `warning(message, **kwargs)`

Логирует warning сообщение.

**Пример:**
```python
logger.warning("Low confidence detected", confidence=0.3)
```

##### `error(message, exc_info=None, **kwargs)`

Логирует error сообщение.

**Пример:**
```python
try:
    # код
except Exception as e:
    logger.error("Failed to process gesture", exc_info=e)
```

##### `critical(message, exc_info=None, **kwargs)`

Логирует critical сообщение.

**Пример:**
```python
logger.critical("Database connection lost", exc_info=e)
```

#### Специализированные методы

##### `log_exception(exc, context="")`

Логирует исключение с полным трейсбеком.

**Пример:**
```python
try:
    # код
except Exception as e:
    logger.log_exception(e, "gesture_processing")
```

##### `log_user_action(user_id, action, details=None)`

Логирует действие пользователя.

**Пример:**
```python
logger.log_user_action(
    user_id=1,
    action="changed_theme",
    details={'old_theme': 'dark', 'new_theme': 'blue'}
)
```

##### `log_gesture(user_id, gesture, confidence, duration_ms=None)`

Логирует распознанный жест.

**Пример:**
```python
logger.log_gesture(
    user_id=1,
    gesture="Hello",
    confidence=0.95,
    duration_ms=1500
)
```

##### `log_performance(operation, duration, success=True, details=None)`

Логирует производительность операции.

**Пример:**
```python
import time

start = time.time()
# выполнение операции
duration = time.time() - start

logger.log_performance(
    operation="gesture_detection",
    duration=duration,
    success=True
)
```

#### Декораторы

##### `@measure_time(operation_name=None)`

Декоратор для измерения времени выполнения функции.

**Пример:**
```python
@logger.measure_time("process_frame")
def process_frame(frame):
    # обработка кадра
    return result
```

##### `@log_calls(level=logging.DEBUG)`

Декоратор для логирования вызовов функций.

**Пример:**
```python
@logger.log_calls(level=logging.INFO)
def important_function(arg1, arg2):
    # код
    return result
```

#### Утилиты

##### `get_performance_stats()`

Получает статистику производительности.

**Возвращает:**
```python
{
    'process_frame': {
        'count': 100,
        'total': 5.234,
        'avg': 0.052,
        'min': 0.045,
        'max': 0.089,
        'last': 0.051
    },
    # ...
}
```

##### `generate_report(output_file=None)`

Генерирует отчет о производительности.

**Пример:**
```python
report = logger.generate_report('performance_report.txt')
print(report)
```

---

## Model API

### `GestureModelWrapper`

Обертка для работы с моделью распознавания жестов.

#### Инициализация

```python
from model.gesture_model import GestureModelWrapper

model = GestureModelWrapper(
    model_path="models/gesture_model.pth",
    use_dummy=False
)
```

**Параметры:**
- `model_path` (str, optional): Путь к модели
- `use_dummy` (bool): Использовать заглушку если модель не найдена

#### Методы

##### `predict(landmarks)`

Предсказывает жест на основе координат.

**Параметры:**
- `landmarks`: Массив с 63 значениями (21 точка × 3 координаты)

**Возвращает:**
- `tuple`: (название жеста, уверенность)

**Пример:**
```python
gesture, confidence = model.predict(normalized_landmarks)
print(f"Gesture: {gesture}, Confidence: {confidence:.2f}")
```

##### `normalize_landmarks(landmarks)`

Нормализует координаты суставов.

**Параметры:**
- `landmarks`: Массив координат

**Возвращает:**
- Нормализованные координаты

**Пример:**
```python
normalized = model.normalize_landmarks(landmarks)
```

---

## Utils API

### Camera

```python
from utils.camera import Camera

camera = Camera(camera_index=0, width=640, height=480)
camera.open()

ret, frame = camera.read()
if ret:
    # обработка кадра
    pass

camera.release()
```

### GestureDetector

```python
from utils.gestures import GestureDetector

detector = GestureDetector(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

hands_data, annotated_frame = detector.detect(frame)

# hands_data содержит:
# {
#     'left': landmarks_array или None,
#     'right': landmarks_array или None,
#     'count': количество обнаруженных рук
# }

detector.close()
```

### TextToSpeech

```python
from utils.speech import TextToSpeech

tts = TextToSpeech(rate=150, volume=0.8)

# Озвучивание
tts.speak("Hello")

# Принудительное озвучивание
tts.speak("Hello", force=True)

# Изменение настроек
tts.set_rate(180)
tts.set_volume(0.9)

# Остановка
tts.stop()
```

---

## Примеры использования

### Полный пример приложения

```python
from database.db_manager import DatabaseManager
from analytics.analytics_engine import AnalyticsEngine
from config.themes import ThemeManager, UserPreferences
from utils.logger import get_logger
from model.gesture_model import GestureModelWrapper
from utils.gestures import GestureDetector
from utils.camera import Camera
import uuid

# Инициализация
db = DatabaseManager("my_app.db")
analytics = AnalyticsEngine(db)
theme_manager = ThemeManager()
logger = get_logger()

# Создание пользователя
user_id = db.create_user("john_doe", "password123", "john@example.com")

# Настройки
prefs = UserPreferences(db, user_id)
prefs.set('theme', 'blue')

# Инициализация компонентов
model = GestureModelWrapper(use_dummy=True)
detector = GestureDetector(max_num_hands=2)
camera = Camera()

# Начало сессии
session_id = str(uuid.uuid4())
db.start_session(user_id, session_id)

# Основной цикл
camera.open()
try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Обнаружение
        hands_data, annotated_frame = detector.detect(frame)
        landmarks = hands_data.get('right')
        
        if landmarks is not None:
            # Распознавание
            normalized = model.normalize_landmarks(landmarks)
            gesture, confidence = model.predict(normalized)
            
            # Сохранение
            db.add_gesture(user_id, session_id, gesture, confidence)
            
            # Логирование
            logger.log_gesture(user_id, gesture, confidence)
            
            print(f"Gesture: {gesture} ({confidence:.2%})")
        
        # Отображение
        cv2.imshow('Camera', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    detector.close()
    cv2.destroyAllWindows()

# Завершение сессии
db.end_session(session_id)

# Проверка достижений
db.check_achievements(user_id)

# Получение статистики
dashboard = analytics.get_user_dashboard(user_id)
print(dashboard['overview'])

# Генерация отчета
report = analytics.generate_report(user_id)
print(report)
```

### Пример с контекстными менеджерами

```python
from utils.gestures import GestureDetector
from utils.camera import Camera

with GestureDetector() as detector, Camera() as camera:
    camera.open()
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        hands_data, annotated_frame = detector.detect(frame)
        
        # обработка...
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

---

## Константы и перечисления

### Жесты

```python
from model.gesture_model import GESTURE_CLASSES

# ['Hello', 'Thanks', 'Yes', 'No', 'Love']
```

### Темы

```python
from config.themes import THEMES

# {
#     'dark': ColorScheme(...),
#     'light': ColorScheme(...),
#     # ...
# }
```

---

## Типы данных

### ColorScheme

```python
@dataclass
class ColorScheme:
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
```

---

## Обработка ошибок

### Общие исключения

```python
try:
    user_id = db.create_user("existing_user", "password")
except sqlite3.IntegrityError:
    print("User already exists")

try:
    model = GestureModelWrapper(model_path="invalid_path")
except FileNotFoundError:
    print("Model file not found")
```

### Логирование ошибок

```python
try:
    # код
except Exception as e:
    logger.log_exception(e, "operation_name")
    # обработка ошибки
```

---

## Best Practices

### 1. Управление ресурсами

```python
# Всегда закрывайте ресурсы
camera.open()
try:
    # использование
finally:
    camera.release()

# Или используйте контекстные менеджеры
with Camera() as camera:
    # использование
```

### 2. Логирование

```python
# Используйте соответствующие уровни
logger.debug("Подробная информация для отладки")
logger.info("Общая информация о работе")
logger.warning("Предупреждения")
logger.error("Ошибки")
logger.critical("Критические ошибки")

# Добавляйте контекст
logger.info("User action", user_id=1, action="login", ip="192.168.1.1")
```

### 3. Работа с БД

```python
# Проверяйте возвращаемые значения
user_id = db.create_user("username", "password")
if user_id:
    print("Success")
else:
    print("Failed: user already exists")

# Регулярно проверяйте достижения
db.check_achievements(user_id)
```

### 4. Производительность

```python
# Используйте декоратор для измерения времени
@logger.measure_time("heavy_operation")
def heavy_operation():
    # код
    
# Получайте статистику
stats = logger.get_performance_stats()
```

---

<div align="center">

## 📚 SignVoiceAI API Reference

**Полная документация для разработчиков**

[Вернуться к README](README.md) • [Enterprise README](ENTERPRISE_README.md)

</div>





