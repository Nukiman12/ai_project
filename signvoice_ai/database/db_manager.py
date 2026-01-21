"""
Модуль управления базой данных для SignVoiceAI Enterprise.

Обеспечивает:
- Управление пользователями
- Хранение истории жестов
- Статистика и аналитика
- Настройки пользователей
- Достижения и прогресс
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import threading


class DatabaseManager:
    """
    Менеджер базы данных для SignVoiceAI Enterprise.
    Реализует полноценную систему управления данными.
    """
    
    def __init__(self, db_path: str = "signvoice_data.db"):
        """
        Инициализация менеджера БД.
        
        Args:
            db_path: Путь к файлу базы данных
        """
        print(f"  → Инициализация БД: {db_path}")
        self.db_path = db_path
        self.lock = threading.RLock()  # RLock позволяет реентрабельность
        print("  → Создание таблиц...")
        self.init_database()
        print("  → БД готова")
    
    def get_connection(self) -> sqlite3.Connection:
        """Создает соединение с БД."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Инициализирует схему базы данных."""
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Таблица пользователей
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT UNIQUE,
                full_name TEXT,
                avatar_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                user_level INTEGER DEFAULT 1,
                total_gestures INTEGER DEFAULT 0,
                total_sessions INTEGER DEFAULT 0,
                total_time_seconds INTEGER DEFAULT 0
            )
            ''')
            
            # Таблица профилей
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                bio TEXT,
                language TEXT DEFAULT 'ru',
                theme TEXT DEFAULT 'dark',
                preferred_voice TEXT,
                notifications_enabled BOOLEAN DEFAULT 1,
                auto_speak BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            ''')
            
            # Таблица настроек пользователей
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                setting_key TEXT NOT NULL,
                setting_value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                UNIQUE(user_id, setting_key)
            )
            ''')
            
            # Таблица истории жестов
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS gesture_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                gesture_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_ms INTEGER,
                hand_type TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            ''')
            
            # Таблица сессий
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                total_gestures INTEGER DEFAULT 0,
                avg_confidence REAL,
                duration_seconds INTEGER,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            ''')
            
            # Таблица достижений
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS achievements (
                achievement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                icon TEXT,
                category TEXT,
                requirement_type TEXT NOT NULL,
                requirement_value INTEGER NOT NULL,
                points INTEGER DEFAULT 10
            )
            ''')
            
            # Таблица прогресса пользователей
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_achievements (
                user_achievement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                achievement_id INTEGER NOT NULL,
                unlocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                progress INTEGER DEFAULT 0,
                is_completed BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (achievement_id) REFERENCES achievements (achievement_id),
                UNIQUE(user_id, achievement_id)
            )
            ''')
            
            # Таблица статистики по жестам
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS gesture_statistics (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                gesture_name TEXT NOT NULL,
                total_count INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                best_confidence REAL DEFAULT 0.0,
                last_used TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                UNIQUE(user_id, gesture_name)
            )
            ''')
            
            # Таблица экспортированных данных
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS exports (
                export_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                export_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            ''')
            
            # Индексы для оптимизации
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_gesture_history_user 
            ON gesture_history(user_id)
            ''')
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_gesture_history_session 
            ON gesture_history(session_id)
            ''')
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sessions_user 
            ON sessions(user_id)
            ''')
            
            conn.commit()
            conn.close()
            
            # Создаем достижения по умолчанию
            self.init_default_achievements()
    
    def init_default_achievements(self):
        """Создает достижения по умолчанию."""
        default_achievements = [
            ("Первые шаги", "Распознать первый жест", "🎯", "beginner", "gesture_count", 1, 10),
            ("Новичок", "Распознать 10 жестов", "🌟", "beginner", "gesture_count", 10, 20),
            ("Практик", "Распознать 50 жестов", "⭐", "intermediate", "gesture_count", 50, 50),
            ("Эксперт", "Распознать 100 жестов", "🏆", "advanced", "gesture_count", 100, 100),
            ("Мастер", "Распознать 500 жестов", "👑", "master", "gesture_count", 500, 250),
            ("Легенда", "Распознать 1000 жестов", "💎", "legendary", "gesture_count", 1000, 500),
            
            ("Час практики", "Заниматься 1 час", "⏰", "beginner", "time_hours", 1, 15),
            ("Марафонец", "Заниматься 10 часов", "🏃", "intermediate", "time_hours", 10, 75),
            ("Преданный", "Заниматься 50 часов", "💪", "advanced", "time_hours", 50, 300),
            
            ("Точность", "Достичь 95% уверенности", "🎯", "intermediate", "high_confidence", 95, 50),
            ("Идеал", "Достичь 99% уверенности", "✨", "advanced", "high_confidence", 99, 100),
            
            ("Первая сессия", "Завершить первую сессию", "📅", "beginner", "session_count", 1, 5),
            ("Регулярность", "Провести 10 сессий", "📊", "intermediate", "session_count", 10, 30),
            ("Постоянство", "Провести 50 сессий", "🔥", "advanced", "session_count", 50, 150),
        ]
        
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            for achievement in default_achievements:
                cursor.execute('''
                INSERT OR IGNORE INTO achievements 
                (name, description, icon, category, requirement_type, requirement_value, points)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', achievement)
            
            conn.commit()
            conn.close()
    
    # ===== УПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯМИ =====
    
    def create_user(self, username: str, password: str, email: Optional[str] = None,
                   full_name: Optional[str] = None) -> Optional[int]:
        """
        Создает нового пользователя.
        
        Args:
            username: Имя пользователя
            password: Пароль
            email: Email (опционально)
            full_name: Полное имя (опционально)
            
        Returns:
            ID созданного пользователя или None при ошибке
        """
        password_hash = self._hash_password(password)
        
        with self.lock:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO users (username, password_hash, email, full_name)
                VALUES (?, ?, ?, ?)
                ''', (username, password_hash, email, full_name))
                
                user_id = cursor.lastrowid
                
                # Создаем профиль по умолчанию
                cursor.execute('''
                INSERT INTO user_profiles (user_id)
                VALUES (?)
                ''', (user_id,))
                
                conn.commit()
                conn.close()
                
                return user_id
            except sqlite3.IntegrityError:
                return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[int]:
        """
        Аутентифицирует пользователя.
        
        Args:
            username: Имя пользователя
            password: Пароль
            
        Returns:
            ID пользователя или None если аутентификация не удалась
        """
        password_hash = self._hash_password(password)
        
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT user_id FROM users
            WHERE username = ? AND password_hash = ? AND is_active = 1
            ''', (username, password_hash))
            
            result = cursor.fetchone()
            
            if result:
                user_id = result[0]
                # Обновляем время последнего входа
                cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE user_id = ?
                ''', (user_id,))
                conn.commit()
            else:
                user_id = None
            
            conn.close()
            return user_id
    
    def get_user_info(self, user_id: int) -> Optional[Dict]:
        """
        Получает информацию о пользователе.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Словарь с информацией о пользователе
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT u.*, p.bio, p.language, p.theme, p.preferred_voice,
                   p.notifications_enabled, p.auto_speak
            FROM users u
            LEFT JOIN user_profiles p ON u.user_id = p.user_id
            WHERE u.user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return dict(row)
            return None
    
    def update_user_profile(self, user_id: int, **kwargs):
        """
        Обновляет профиль пользователя.
        
        Args:
            user_id: ID пользователя
            **kwargs: Поля для обновления
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Обновляем основную таблицу пользователей
            user_fields = ['full_name', 'email', 'avatar_path']
            user_updates = {k: v for k, v in kwargs.items() if k in user_fields}
            
            if user_updates:
                set_clause = ', '.join([f"{k} = ?" for k in user_updates.keys()])
                values = list(user_updates.values()) + [user_id]
                cursor.execute(f'''
                UPDATE users SET {set_clause}
                WHERE user_id = ?
                ''', values)
            
            # Обновляем таблицу профилей
            profile_fields = ['bio', 'language', 'theme', 'preferred_voice',
                            'notifications_enabled', 'auto_speak']
            profile_updates = {k: v for k, v in kwargs.items() if k in profile_fields}
            
            if profile_updates:
                set_clause = ', '.join([f"{k} = ?" for k in profile_updates.keys()])
                values = list(profile_updates.values()) + [user_id]
                cursor.execute(f'''
                UPDATE user_profiles SET {set_clause}
                WHERE user_id = ?
                ''', values)
            
            conn.commit()
            conn.close()
    
    # ===== УПРАВЛЕНИЕ ЖЕСТАМИ И СЕССИЯМИ =====
    
    def add_gesture(self, user_id: int, session_id: str, gesture_name: str,
                   confidence: float, hand_type: str = "right",
                   duration_ms: Optional[int] = None):
        """
        Добавляет распознанный жест в историю.
        
        Args:
            user_id: ID пользователя
            session_id: ID сессии
            gesture_name: Название жеста
            confidence: Уверенность распознавания
            hand_type: Тип руки (left/right)
            duration_ms: Длительность в миллисекундах
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Добавляем в историю
            cursor.execute('''
            INSERT INTO gesture_history 
            (user_id, session_id, gesture_name, confidence, hand_type, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, session_id, gesture_name, confidence, hand_type, duration_ms))
            
            # Обновляем статистику пользователя
            cursor.execute('''
            UPDATE users 
            SET total_gestures = total_gestures + 1
            WHERE user_id = ?
            ''', (user_id,))
            
            # Обновляем статистику по жестам
            cursor.execute('''
            INSERT INTO gesture_statistics (user_id, gesture_name, total_count, 
                                          avg_confidence, best_confidence, last_used)
            VALUES (?, ?, 1, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, gesture_name) DO UPDATE SET
                total_count = total_count + 1,
                avg_confidence = ((avg_confidence * (total_count - 1)) + ?) / total_count,
                best_confidence = MAX(best_confidence, ?),
                last_used = CURRENT_TIMESTAMP
            ''', (user_id, gesture_name, confidence, confidence, confidence))
            
            conn.commit()
            conn.close()
            
            # Достижения проверяются периодически из GUI
            # (не при каждом жесте для производительности)
    
    def start_session(self, user_id: int, session_id: str):
        """
        Начинает новую сессию.
        
        Args:
            user_id: ID пользователя
            session_id: ID сессии
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO sessions (session_id, user_id)
            VALUES (?, ?)
            ''', (session_id, user_id))
            
            conn.commit()
            conn.close()
    
    def end_session(self, session_id: str, notes: Optional[str] = None):
        """
        Завершает сессию.
        
        Args:
            session_id: ID сессии
            notes: Заметки о сессии
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Получаем статистику сессии
            cursor.execute('''
            SELECT COUNT(*) as total, AVG(confidence) as avg_conf
            FROM gesture_history
            WHERE session_id = ?
            ''', (session_id,))
            
            stats = cursor.fetchone()
            
            # Получаем время начала
            cursor.execute('''
            SELECT start_time, user_id FROM sessions WHERE session_id = ?
            ''', (session_id,))
            
            session_info = cursor.fetchone()
            
            if session_info:
                start_time = datetime.fromisoformat(session_info['start_time'])
                user_id = session_info['user_id']
                duration = int((datetime.now() - start_time).total_seconds())
                
                # Обновляем сессию
                cursor.execute('''
                UPDATE sessions
                SET end_time = CURRENT_TIMESTAMP,
                    total_gestures = ?,
                    avg_confidence = ?,
                    duration_seconds = ?,
                    notes = ?
                WHERE session_id = ?
                ''', (stats['total'], stats['avg_conf'], duration, notes, session_id))
                
                # Обновляем статистику пользователя
                cursor.execute('''
                UPDATE users
                SET total_sessions = total_sessions + 1,
                    total_time_seconds = total_time_seconds + ?
                WHERE user_id = ?
                ''', (duration, user_id))
            
            conn.commit()
            conn.close()
    
    def get_user_statistics(self, user_id: int) -> Dict:
        """
        Получает статистику пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Словарь со статистикой
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Основная статистика
            cursor.execute('''
            SELECT total_gestures, total_sessions, total_time_seconds, user_level
            FROM users WHERE user_id = ?
            ''', (user_id,))
            
            user_stats = dict(cursor.fetchone())
            
            # Статистика по жестам
            cursor.execute('''
            SELECT gesture_name, total_count, avg_confidence, best_confidence
            FROM gesture_statistics
            WHERE user_id = ?
            ORDER BY total_count DESC
            ''', (user_id,))
            
            gesture_stats = [dict(row) for row in cursor.fetchall()]
            
            # Последние сессии
            cursor.execute('''
            SELECT session_id, start_time, end_time, total_gestures, 
                   avg_confidence, duration_seconds
            FROM sessions
            WHERE user_id = ?
            ORDER BY start_time DESC
            LIMIT 10
            ''', (user_id,))
            
            recent_sessions = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                'user': user_stats,
                'gestures': gesture_stats,
                'recent_sessions': recent_sessions
            }
    
    # ===== ДОСТИЖЕНИЯ =====
    
    def check_achievements(self, user_id: int):
        """
        Проверяет и обновляет достижения пользователя.
        
        Args:
            user_id: ID пользователя
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Получаем все достижения
            cursor.execute('SELECT * FROM achievements')
            achievements = cursor.fetchall()
            
            # Получаем статистику пользователя (встроенный запрос для избежания вложенных блокировок)
            cursor.execute('''
            SELECT total_gestures, total_sessions, total_time_seconds, user_level
            FROM users WHERE user_id = ?
            ''', (user_id,))
            user_stats_row = cursor.fetchone()
            
            cursor.execute('''
            SELECT gesture_name, total_count, avg_confidence, best_confidence
            FROM gesture_statistics
            WHERE user_id = ?
            ORDER BY total_count DESC
            ''', (user_id,))
            gesture_stats = cursor.fetchall()
            
            user_stats = {
                'user': dict(user_stats_row) if user_stats_row else {},
                'gestures': [dict(row) for row in gesture_stats]
            }
            
            for achievement in achievements:
                req_type = achievement['requirement_type']
                req_value = achievement['requirement_value']
                current_value = 0
                
                # Определяем текущий прогресс
                if req_type == 'gesture_count':
                    current_value = user_stats['user']['total_gestures']
                elif req_type == 'time_hours':
                    current_value = user_stats['user']['total_time_seconds'] // 3600
                elif req_type == 'session_count':
                    current_value = user_stats['user']['total_sessions']
                elif req_type == 'high_confidence':
                    # Проверяем максимальную уверенность
                    if user_stats['gestures']:
                        max_conf = max([g['best_confidence'] for g in user_stats['gestures']])
                        current_value = int(max_conf * 100)
                
                # Обновляем прогресс
                is_completed = current_value >= req_value
                
                cursor.execute('''
                INSERT INTO user_achievements (user_id, achievement_id, progress, is_completed)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, achievement_id) DO UPDATE SET
                    progress = ?,
                    is_completed = ?,
                    unlocked_at = CASE WHEN ? AND NOT is_completed 
                                       THEN CURRENT_TIMESTAMP 
                                       ELSE unlocked_at END
                ''', (user_id, achievement['achievement_id'], current_value, is_completed,
                     current_value, is_completed, is_completed))
            
            conn.commit()
            conn.close()
    
    def get_user_achievements(self, user_id: int) -> List[Dict]:
        """
        Получает достижения пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Список достижений с прогрессом
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT a.*, ua.progress, ua.is_completed, ua.unlocked_at
            FROM achievements a
            LEFT JOIN user_achievements ua 
                ON a.achievement_id = ua.achievement_id AND ua.user_id = ?
            ORDER BY a.points ASC
            ''', (user_id,))
            
            achievements = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return achievements
    
    # ===== УТИЛИТЫ =====
    
    def _hash_password(self, password: str) -> str:
        """Хеширует пароль."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def get_setting(self, user_id: int, key: str, default: Any = None) -> Any:
        """
        Получает настройку пользователя.
        
        Args:
            user_id: ID пользователя
            key: Ключ настройки
            default: Значение по умолчанию
            
        Returns:
            Значение настройки
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT setting_value FROM user_settings
            WHERE user_id = ? AND setting_key = ?
            ''', (user_id, key))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                try:
                    return json.loads(result[0])
                except:
                    return result[0]
            return default
    
    def set_setting(self, user_id: int, key: str, value: Any):
        """
        Устанавливает настройку пользователя.
        
        Args:
            user_id: ID пользователя
            key: Ключ настройки
            value: Значение настройки
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            value_str = json.dumps(value) if not isinstance(value, str) else value
            
            cursor.execute('''
            INSERT INTO user_settings (user_id, setting_key, setting_value)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, setting_key) DO UPDATE SET
                setting_value = ?,
                updated_at = CURRENT_TIMESTAMP
            ''', (user_id, key, value_str, value_str))
            
            conn.commit()
            conn.close()
    
    def export_user_data(self, user_id: int, export_type: str = 'json') -> str:
        """
        Экспортирует данные пользователя.
        
        Args:
            user_id: ID пользователя
            export_type: Тип экспорта (json/csv)
            
        Returns:
            Путь к файлу экспорта
        """
        import csv
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"signvoice_export_{user_id}_{timestamp}.{export_type}"
        filepath = Path("exports") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        # Получаем все данные
        user_info = self.get_user_info(user_id)
        statistics = self.get_user_statistics(user_id)
        achievements = self.get_user_achievements(user_id)
        
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # История жестов
            cursor.execute('''
            SELECT * FROM gesture_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            ''', (user_id,))
            
            history = [dict(row) for row in cursor.fetchall()]
            conn.close()
        
        data = {
            'user_info': user_info,
            'statistics': statistics,
            'achievements': achievements,
            'history': history,
            'exported_at': datetime.now().isoformat()
        }
        
        if export_type == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        elif export_type == 'csv':
            # Экспортируем историю в CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                if history:
                    writer = csv.DictWriter(f, fieldnames=history[0].keys())
                    writer.writeheader()
                    writer.writerows(history)
        
        # Сохраняем запись об экспорте
        file_size = filepath.stat().st_size
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO exports (user_id, export_type, file_path, file_size)
            VALUES (?, ?, ?, ?)
            ''', (user_id, export_type, str(filepath), file_size))
            conn.commit()
            conn.close()
        
        return str(filepath)


