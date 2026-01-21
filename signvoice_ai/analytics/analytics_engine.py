"""
Модуль аналитики для SignVoiceAI Enterprise.

Предоставляет:
- Продвинутую статистику
- Визуализацию данных
- Анализ производительности
- Рекомендации для пользователя
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import Counter
import json


class AnalyticsEngine:
    """
    Движок аналитики для анализа данных пользователя.
    """
    
    def __init__(self, db_manager):
        """
        Инициализация движка аналитики.
        
        Args:
            db_manager: Экземпляр DatabaseManager
        """
        self.db = db_manager
    
    def get_user_dashboard(self, user_id: int) -> Dict:
        """
        Получает данные для панели управления пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Словарь с данными dashboard
        """
        stats = self.db.get_user_statistics(user_id)
        
        # Базовая статистика
        dashboard = {
            'overview': {
                'total_gestures': stats['user']['total_gestures'],
                'total_sessions': stats['user']['total_sessions'],
                'total_time': self._format_duration(stats['user']['total_time_seconds']),
                'level': stats['user']['user_level'],
                'avg_gestures_per_session': self._safe_divide(
                    stats['user']['total_gestures'],
                    stats['user']['total_sessions']
                )
            },
            'performance': self.analyze_performance(user_id),
            'trends': self.analyze_trends(user_id),
            'top_gestures': self.get_top_gestures(user_id, limit=5),
            'recent_activity': self.get_recent_activity(user_id, days=7),
            'recommendations': self.generate_recommendations(user_id)
        }
        
        return dashboard
    
    def analyze_performance(self, user_id: int) -> Dict:
        """
        Анализирует производительность пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Словарь с метриками производительности
        """
        stats = self.db.get_user_statistics(user_id)
        
        # Средняя уверенность по всем жестам
        avg_confidence = 0.0
        if stats['gestures']:
            avg_confidence = np.mean([g['avg_confidence'] for g in stats['gestures']])
        
        # Лучшая уверенность
        best_confidence = 0.0
        if stats['gestures']:
            best_confidence = max([g['best_confidence'] for g in stats['gestures']])
        
        # Разнообразие жестов
        unique_gestures = len(stats['gestures'])
        
        # Средняя длительность сессии
        avg_session_duration = 0
        if stats['recent_sessions']:
            durations = [s['duration_seconds'] for s in stats['recent_sessions'] 
                        if s['duration_seconds']]
            if durations:
                avg_session_duration = int(np.mean(durations))
        
        # Консистентность (стандартное отклонение уверенности)
        consistency_score = 0.0
        if stats['gestures']:
            confidences = [g['avg_confidence'] for g in stats['gestures']]
            std_dev = np.std(confidences)
            # Чем меньше отклонение, тем выше консистентность
            consistency_score = max(0, 1 - std_dev)
        
        # Оценка прогресса (сравнение последних 5 сессий с предыдущими 5)
        progress_trend = self._calculate_progress_trend(user_id)
        
        return {
            'avg_confidence': round(avg_confidence, 3),
            'best_confidence': round(best_confidence, 3),
            'unique_gestures': unique_gestures,
            'avg_session_duration': avg_session_duration,
            'consistency_score': round(consistency_score, 3),
            'progress_trend': progress_trend,
            'performance_rating': self._calculate_rating(avg_confidence, consistency_score)
        }
    
    def analyze_trends(self, user_id: int, days: int = 30) -> Dict:
        """
        Анализирует тренды активности пользователя.
        
        Args:
            user_id: ID пользователя
            days: Количество дней для анализа
            
        Returns:
            Словарь с трендами
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Получаем данные за последние N дней
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Жесты по дням
        cursor.execute('''
        SELECT DATE(timestamp) as date, COUNT(*) as count, AVG(confidence) as avg_conf
        FROM gesture_history
        WHERE user_id = ? AND timestamp >= ?
        GROUP BY DATE(timestamp)
        ORDER BY date
        ''', (user_id, start_date))
        
        daily_gestures = [dict(row) for row in cursor.fetchall()]
        
        # Жесты по часам (для определения peak hours)
        cursor.execute('''
        SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour, 
               COUNT(*) as count
        FROM gesture_history
        WHERE user_id = ? AND timestamp >= ?
        GROUP BY hour
        ORDER BY count DESC
        ''', (user_id, start_date))
        
        hourly_activity = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        # Анализ тренда
        trend_direction = 'stable'
        if len(daily_gestures) >= 7:
            recent = sum([d['count'] for d in daily_gestures[-3:]])
            previous = sum([d['count'] for d in daily_gestures[-7:-3]])
            if recent > previous * 1.2:
                trend_direction = 'increasing'
            elif recent < previous * 0.8:
                trend_direction = 'decreasing'
        
        # Определяем пиковые часы
        peak_hours = []
        if hourly_activity:
            peak_hours = [h['hour'] for h in hourly_activity[:3]]
        
        return {
            'daily_activity': daily_gestures,
            'trend_direction': trend_direction,
            'peak_hours': peak_hours,
            'most_active_day': self._get_most_active_day(daily_gestures),
            'activity_consistency': self._calculate_activity_consistency(daily_gestures)
        }
    
    def get_top_gestures(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        Получает топ жестов пользователя.
        
        Args:
            user_id: ID пользователя
            limit: Количество жестов
            
        Returns:
            Список топ жестов
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT gesture_name, total_count, avg_confidence, best_confidence,
               last_used
        FROM gesture_statistics
        WHERE user_id = ?
        ORDER BY total_count DESC
        LIMIT ?
        ''', (user_id, limit))
        
        top_gestures = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return top_gestures
    
    def get_recent_activity(self, user_id: int, days: int = 7) -> Dict:
        """
        Получает недавнюю активность пользователя.
        
        Args:
            user_id: ID пользователя
            days: Количество дней
            
        Returns:
            Словарь с недавней активностью
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Общее количество жестов
        cursor.execute('''
        SELECT COUNT(*) as count
        FROM gesture_history
        WHERE user_id = ? AND timestamp >= ?
        ''', (user_id, start_date))
        
        total_gestures = cursor.fetchone()['count']
        
        # Количество сессий
        cursor.execute('''
        SELECT COUNT(*) as count
        FROM sessions
        WHERE user_id = ? AND start_time >= ?
        ''', (user_id, start_date))
        
        total_sessions = cursor.fetchone()['count']
        
        # Самый частый жест
        cursor.execute('''
        SELECT gesture_name, COUNT(*) as count
        FROM gesture_history
        WHERE user_id = ? AND timestamp >= ?
        GROUP BY gesture_name
        ORDER BY count DESC
        LIMIT 1
        ''', (user_id, start_date))
        
        most_used = cursor.fetchone()
        most_used_gesture = dict(most_used) if most_used else None
        
        conn.close()
        
        return {
            'period_days': days,
            'total_gestures': total_gestures,
            'total_sessions': total_sessions,
            'avg_gestures_per_day': round(total_gestures / days, 1),
            'most_used_gesture': most_used_gesture
        }
    
    def generate_recommendations(self, user_id: int) -> List[str]:
        """
        Генерирует рекомендации для пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Список рекомендаций
        """
        recommendations = []
        
        stats = self.db.get_user_statistics(user_id)
        performance = self.analyze_performance(user_id)
        trends = self.analyze_trends(user_id)
        
        # Рекомендации по уверенности
        if performance['avg_confidence'] < 0.7:
            recommendations.append(
                "💡 Улучшите освещение и фон для повышения точности распознавания"
            )
            recommendations.append(
                "🎯 Практикуйте жесты медленнее для лучшего распознавания"
            )
        
        # Рекомендации по разнообразию
        if performance['unique_gestures'] < 5:
            recommendations.append(
                "🌟 Попробуйте изучить больше жестов для расширения словаря"
            )
        
        # Рекомендации по активности
        if trends['trend_direction'] == 'decreasing':
            recommendations.append(
                "📅 Ваша активность снижается. Установите регулярные сессии практики"
            )
        
        # Рекомендации по консистентности
        if performance['consistency_score'] < 0.5:
            recommendations.append(
                "🎓 Сфокусируйтесь на отработке техники для более стабильных результатов"
            )
        
        # Рекомендации по длительности сессий
        if performance['avg_session_duration'] < 300:  # менее 5 минут
            recommendations.append(
                "⏰ Увеличьте длительность сессий до 10-15 минут для лучших результатов"
            )
        
        # Позитивные рекомендации
        if performance['avg_confidence'] > 0.85:
            recommendations.append(
                "🏆 Отличная работа! Ваша точность распознавания очень высокая"
            )
        
        if trends['trend_direction'] == 'increasing':
            recommendations.append(
                "📈 Вы на правильном пути! Продолжайте регулярные занятия"
            )
        
        # Если нет рекомендаций, добавляем общую
        if not recommendations:
            recommendations.append(
                "✅ Отличная работа! Продолжайте в том же духе"
            )
        
        return recommendations
    
    def get_gesture_timeline(self, user_id: int, gesture_name: str = None) -> List[Dict]:
        """
        Получает временную линию жестов.
        
        Args:
            user_id: ID пользователя
            gesture_name: Фильтр по названию жеста
            
        Returns:
            Список событий на временной линии
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        if gesture_name:
            cursor.execute('''
            SELECT timestamp, gesture_name, confidence, hand_type
            FROM gesture_history
            WHERE user_id = ? AND gesture_name = ?
            ORDER BY timestamp DESC
            LIMIT 100
            ''', (user_id, gesture_name))
        else:
            cursor.execute('''
            SELECT timestamp, gesture_name, confidence, hand_type
            FROM gesture_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 100
            ''', (user_id,))
        
        timeline = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return timeline
    
    def compare_sessions(self, user_id: int, session_id1: str, session_id2: str) -> Dict:
        """
        Сравнивает две сессии пользователя.
        
        Args:
            user_id: ID пользователя
            session_id1: ID первой сессии
            session_id2: ID второй сессии
            
        Returns:
            Словарь с сравнением
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Получаем информацию о сессиях
        cursor.execute('''
        SELECT * FROM sessions WHERE session_id = ?
        ''', (session_id1,))
        session1 = dict(cursor.fetchone())
        
        cursor.execute('''
        SELECT * FROM sessions WHERE session_id = ?
        ''', (session_id2,))
        session2 = dict(cursor.fetchone())
        
        # Получаем жесты для каждой сессии
        cursor.execute('''
        SELECT gesture_name, COUNT(*) as count, AVG(confidence) as avg_conf
        FROM gesture_history
        WHERE session_id = ?
        GROUP BY gesture_name
        ''', (session_id1,))
        gestures1 = {row['gesture_name']: dict(row) for row in cursor.fetchall()}
        
        cursor.execute('''
        SELECT gesture_name, COUNT(*) as count, AVG(confidence) as avg_conf
        FROM gesture_history
        WHERE session_id = ?
        GROUP BY gesture_name
        ''', (session_id2,))
        gestures2 = {row['gesture_name']: dict(row) for row in cursor.fetchall()}
        
        conn.close()
        
        # Сравнение
        comparison = {
            'session1': session1,
            'session2': session2,
            'differences': {
                'total_gestures': session2['total_gestures'] - session1['total_gestures'],
                'avg_confidence': session2['avg_confidence'] - session1['avg_confidence'],
                'duration': session2['duration_seconds'] - session1['duration_seconds']
            },
            'gestures1': gestures1,
            'gestures2': gestures2,
            'improvement': self._calculate_improvement(session1, session2)
        }
        
        return comparison
    
    # ===== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ =====
    
    def _format_duration(self, seconds: int) -> str:
        """Форматирует длительность в читаемый вид."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        
        if hours > 0:
            return f"{hours}ч {minutes}м"
        elif minutes > 0:
            return f"{minutes}м"
        else:
            return f"{seconds}с"
    
    def _safe_divide(self, a: float, b: float) -> float:
        """Безопасное деление."""
        return round(a / b, 2) if b > 0 else 0.0
    
    def _calculate_progress_trend(self, user_id: int) -> str:
        """Вычисляет тренд прогресса."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT avg_confidence FROM sessions
        WHERE user_id = ? AND avg_confidence IS NOT NULL
        ORDER BY start_time DESC
        LIMIT 10
        ''', (user_id,))
        
        confidences = [row['avg_confidence'] for row in cursor.fetchall()]
        conn.close()
        
        if len(confidences) < 5:
            return 'insufficient_data'
        
        recent = np.mean(confidences[:5])
        previous = np.mean(confidences[5:])
        
        if recent > previous * 1.05:
            return 'improving'
        elif recent < previous * 0.95:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_rating(self, confidence: float, consistency: float) -> str:
        """Вычисляет общую оценку производительности."""
        score = (confidence * 0.6 + consistency * 0.4)
        
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'very_good'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.6:
            return 'fair'
        else:
            return 'needs_improvement'
    
    def _get_most_active_day(self, daily_activity: List[Dict]) -> Optional[str]:
        """Определяет самый активный день."""
        if not daily_activity:
            return None
        
        most_active = max(daily_activity, key=lambda x: x['count'])
        return most_active['date']
    
    def _calculate_activity_consistency(self, daily_activity: List[Dict]) -> float:
        """Вычисляет консистентность активности."""
        if not daily_activity:
            return 0.0
        
        counts = [d['count'] for d in daily_activity]
        if len(counts) < 2:
            return 1.0
        
        std_dev = np.std(counts)
        mean = np.mean(counts)
        
        if mean == 0:
            return 0.0
        
        # Коэффициент вариации (чем ниже, тем консистентнее)
        cv = std_dev / mean
        # Инвертируем и нормализуем в диапазон 0-1
        consistency = max(0, 1 - min(cv, 1))
        
        return round(consistency, 3)
    
    def _calculate_improvement(self, session1: Dict, session2: Dict) -> Dict:
        """Вычисляет улучшение между сессиями."""
        improvement = {
            'gestures': {
                'value': session2['total_gestures'] - session1['total_gestures'],
                'percentage': self._percentage_change(
                    session1['total_gestures'], 
                    session2['total_gestures']
                )
            },
            'confidence': {
                'value': session2['avg_confidence'] - session1['avg_confidence'],
                'percentage': self._percentage_change(
                    session1['avg_confidence'], 
                    session2['avg_confidence']
                )
            }
        }
        
        return improvement
    
    def _percentage_change(self, old: float, new: float) -> float:
        """Вычисляет процентное изменение."""
        if old == 0:
            return 0.0
        return round(((new - old) / old) * 100, 1)
    
    def generate_report(self, user_id: int, period_days: int = 30) -> Dict:
        """
        Генерирует полный отчет о пользователе.
        
        Args:
            user_id: ID пользователя
            period_days: Период для отчета в днях
            
        Returns:
            Словарь с полным отчетом
        """
        user_info = self.db.get_user_info(user_id)
        dashboard = self.get_user_dashboard(user_id)
        achievements = self.db.get_user_achievements(user_id)
        
        # Подсчитываем завершенные достижения
        completed_achievements = len([a for a in achievements if a.get('is_completed')])
        total_points = sum([a['points'] for a in achievements if a.get('is_completed')])
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_days': period_days,
            'user_info': {
                'username': user_info['username'],
                'full_name': user_info.get('full_name'),
                'level': user_info['user_level'],
                'member_since': user_info['created_at']
            },
            'summary': dashboard['overview'],
            'performance': dashboard['performance'],
            'trends': dashboard['trends'],
            'achievements': {
                'completed': completed_achievements,
                'total': len(achievements),
                'points': total_points
            },
            'recommendations': dashboard['recommendations'],
            'top_gestures': dashboard['top_gestures']
        }
        
        return report




