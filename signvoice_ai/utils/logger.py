"""
Модуль логирования для SignVoiceAI Enterprise.

Обеспечивает:
- Структурированное логирование
- Ротация логов
- Различные уровни логирования
- Мониторинг производительности
- Отчеты об ошибках
"""

import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from functools import wraps


class ColoredFormatter(logging.Formatter):
    """
    Цветной форматтер для консольного вывода.
    """
    
    # ANSI цветовые коды
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Форматирует запись с цветом."""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname:8}{self.RESET}"
        return super().format(record)


class PerformanceMonitor:
    """
    Монитор производительности для отслеживания метрик.
    """
    
    def __init__(self):
        """Инициализация монитора."""
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """
        Начинает отсчет времени для метрики.
        
        Args:
            name: Название метрики
        """
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Останавливает отсчет времени и возвращает длительность.
        
        Args:
            name: Название метрики
            
        Returns:
            Длительность в секундах
        """
        if name not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(duration)
        del self.start_times[name]
        
        return duration
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Получает статистику по метрике.
        
        Args:
            name: Название метрики
            
        Returns:
            Словарь со статистикой
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        
        return {
            'count': len(values),
            'total': sum(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'last': values[-1]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Получает статистику по всем метрикам.
        
        Returns:
            Словарь со статистикой
        """
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def reset(self, name: Optional[str] = None):
        """
        Сбрасывает статистику.
        
        Args:
            name: Название метрики (если None, сбрасывает все)
        """
        if name:
            if name in self.metrics:
                self.metrics[name] = []
        else:
            self.metrics = {}
            self.start_times = {}


class SignVoiceLogger:
    """
    Главный класс логгера для SignVoiceAI.
    """
    
    def __init__(self, name: str = "SignVoiceAI", log_dir: str = "logs",
                 log_level: str = "INFO", max_bytes: int = 10*1024*1024,
                 backup_count: int = 5):
        """
        Инициализация логгера.
        
        Args:
            name: Имя логгера
            log_dir: Директория для логов
            log_level: Уровень логирования
            max_bytes: Максимальный размер файла лога (10MB)
            backup_count: Количество резервных копий
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Создаем логгер
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers = []  # Очищаем существующие обработчики
        
        # Форматы логов
        console_format = '%(levelname)s | %(name)s | %(message)s'
        file_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s'
        
        # Консольный обработчик
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColoredFormatter(console_format))
        self.logger.addHandler(console_handler)
        
        # Файловый обработчик с ротацией
        log_file = self.log_dir / f"{name.lower()}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        self.logger.addHandler(file_handler)
        
        # Обработчик ошибок (отдельный файл)
        error_file = self.log_dir / f"{name.lower()}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(file_format))
        self.logger.addHandler(error_handler)
        
        # JSON обработчик для структурированных логов
        json_file = self.log_dir / f"{name.lower()}_structured.json"
        self.json_handler = logging.FileHandler(json_file, encoding='utf-8')
        self.json_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.json_handler)
        
        # Монитор производительности
        self.performance = PerformanceMonitor()
        
        self.info(f"Logger initialized: {name}")
    
    # ===== БАЗОВЫЕ МЕТОДЫ ЛОГИРОВАНИЯ =====
    
    def debug(self, message: str, **kwargs):
        """Логирует debug сообщение."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Логирует info сообщение."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Логирует warning сообщение."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info=None, **kwargs):
        """Логирует error сообщение."""
        if exc_info:
            kwargs['exc_info'] = exc_info
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exc_info=None, **kwargs):
        """Логирует critical сообщение."""
        if exc_info:
            kwargs['exc_info'] = exc_info
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """
        Внутренний метод логирования.
        
        Args:
            level: Уровень логирования
            message: Сообщение
            **kwargs: Дополнительные данные
        """
        # Обычное логирование
        self.logger.log(level, message)
        
        # Структурированное JSON логирование
        if level >= logging.INFO:
            structured_log = {
                'timestamp': datetime.now().isoformat(),
                'level': logging.getLevelName(level),
                'logger': self.name,
                'message': message,
                **kwargs
            }
            
            self.json_handler.stream.write(json.dumps(structured_log, ensure_ascii=False) + '\n')
            self.json_handler.stream.flush()
    
    # ===== СПЕЦИАЛИЗИРОВАННЫЕ МЕТОДЫ =====
    
    def log_exception(self, exc: Exception, context: str = ""):
        """
        Логирует исключение с полным трейсбеком.
        
        Args:
            exc: Исключение
            context: Контекст ошибки
        """
        tb = traceback.format_exc()
        message = f"Exception in {context}: {exc.__class__.__name__}: {str(exc)}"
        
        self.error(message, exc_info=True, traceback=tb)
    
    def log_user_action(self, user_id: int, action: str, details: Optional[Dict] = None):
        """
        Логирует действие пользователя.
        
        Args:
            user_id: ID пользователя
            action: Название действия
            details: Дополнительные детали
        """
        self.info(
            f"User action: {action}",
            user_id=user_id,
            action=action,
            details=details or {}
        )
    
    def log_gesture(self, user_id: int, gesture: str, confidence: float,
                   duration_ms: Optional[int] = None):
        """
        Логирует распознанный жест.
        
        Args:
            user_id: ID пользователя
            gesture: Название жеста
            confidence: Уверенность
            duration_ms: Длительность
        """
        self.info(
            f"Gesture recognized: {gesture}",
            user_id=user_id,
            gesture=gesture,
            confidence=confidence,
            duration_ms=duration_ms
        )
    
    def log_session(self, user_id: int, session_id: str, action: str,
                   details: Optional[Dict] = None):
        """
        Логирует событие сессии.
        
        Args:
            user_id: ID пользователя
            session_id: ID сессии
            action: Действие (start/end)
            details: Детали сессии
        """
        self.info(
            f"Session {action}: {session_id}",
            user_id=user_id,
            session_id=session_id,
            action=action,
            details=details or {}
        )
    
    def log_performance(self, operation: str, duration: float,
                       success: bool = True, details: Optional[Dict] = None):
        """
        Логирует производительность операции.
        
        Args:
            operation: Название операции
            duration: Длительность в секундах
            success: Успешность операции
            details: Дополнительные детали
        """
        level = logging.INFO if success else logging.WARNING
        
        self._log(
            level,
            f"Performance: {operation} took {duration:.3f}s",
            operation=operation,
            duration=duration,
            success=success,
            details=details or {}
        )
    
    def log_system_info(self, info: Dict[str, Any]):
        """
        Логирует системную информацию.
        
        Args:
            info: Словарь с системной информацией
        """
        self.info("System info", system_info=info)
    
    # ===== ДЕКОРАТОРЫ =====
    
    def measure_time(self, operation_name: Optional[str] = None):
        """
        Декоратор для измерения времени выполнения функции.
        
        Args:
            operation_name: Название операции
            
        Returns:
            Декоратор
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or func.__name__
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.log_performance(name, duration, True)
                    self.performance.metrics.setdefault(name, []).append(duration)
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    
                    self.log_performance(name, duration, False, {
                        'error': str(e)
                    })
                    raise
            
            return wrapper
        return decorator
    
    def log_calls(self, level: int = logging.DEBUG):
        """
        Декоратор для логирования вызовов функций.
        
        Args:
            level: Уровень логирования
            
        Returns:
            Декоратор
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = func.__name__
                
                self._log(
                    level,
                    f"Calling {func_name}",
                    function=func_name,
                    args=str(args),
                    kwargs=str(kwargs)
                )
                
                try:
                    result = func(*args, **kwargs)
                    
                    self._log(
                        level,
                        f"Completed {func_name}",
                        function=func_name,
                        result=str(result)[:100]  # Ограничиваем длину
                    )
                    
                    return result
                except Exception as e:
                    self.log_exception(e, func_name)
                    raise
            
            return wrapper
        return decorator
    
    # ===== УТИЛИТЫ =====
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Получает статистику производительности.
        
        Returns:
            Словарь со статистикой
        """
        return self.performance.get_all_stats()
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Генерирует отчет о работе приложения.
        
        Args:
            output_file: Путь к файлу для сохранения отчета
            
        Returns:
            Текст отчета
        """
        stats = self.get_performance_stats()
        
        report_lines = [
            "=" * 70,
            f"SignVoiceAI Performance Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            ""
        ]
        
        if stats:
            report_lines.append("Performance Statistics:")
            report_lines.append("-" * 70)
            
            for name, stat in sorted(stats.items()):
                report_lines.append(f"\n{name}:")
                report_lines.append(f"  Count:   {stat['count']}")
                report_lines.append(f"  Average: {stat['avg']:.3f}s")
                report_lines.append(f"  Min:     {stat['min']:.3f}s")
                report_lines.append(f"  Max:     {stat['max']:.3f}s")
                report_lines.append(f"  Total:   {stat['total']:.3f}s")
        else:
            report_lines.append("No performance data available.")
        
        report_lines.append("\n" + "=" * 70)
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def clear_old_logs(self, days: int = 30):
        """
        Удаляет старые логи.
        
        Args:
            days: Количество дней для хранения логов
        """
        from datetime import timedelta
        
        cutoff_time = time.time() - (days * 86400)
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.info(f"Deleted old log: {log_file}")
                except Exception as e:
                    self.warning(f"Failed to delete log {log_file}: {e}")


# Глобальный логгер
_global_logger: Optional[SignVoiceLogger] = None


def get_logger(name: str = "SignVoiceAI", **kwargs) -> SignVoiceLogger:
    """
    Получает или создает логгер.
    
    Args:
        name: Имя логгера
        **kwargs: Параметры для создания логгера
        
    Returns:
        Экземпляр логгера
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = SignVoiceLogger(name, **kwargs)
    
    return _global_logger


def init_logger(**kwargs) -> SignVoiceLogger:
    """
    Инициализирует глобальный логгер.
    
    Args:
        **kwargs: Параметры для логгера
        
    Returns:
        Экземпляр логгера
    """
    global _global_logger
    _global_logger = SignVoiceLogger(**kwargs)
    return _global_logger


# Удобные функции
def debug(message: str, **kwargs):
    """Логирует debug сообщение."""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """Логирует info сообщение."""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """Логирует warning сообщение."""
    get_logger().warning(message, **kwargs)


def error(message: str, exc_info=None, **kwargs):
    """Логирует error сообщение."""
    get_logger().error(message, exc_info, **kwargs)


def critical(message: str, exc_info=None, **kwargs):
    """Логирует critical сообщение."""
    get_logger().critical(message, exc_info, **kwargs)





