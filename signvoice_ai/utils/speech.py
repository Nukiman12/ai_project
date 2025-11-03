"""
Модуль для голосового вывода текста через pyttsx3.
"""

import pyttsx3
import threading
import queue


class TextToSpeech:
    """
    Класс для преобразования текста в речь.
    Использует pyttsx3 для синтеза речи.
    """
    
    def __init__(self, rate=150, volume=0.8):
        """
        Инициализация синтезатора речи.
        
        Args:
            rate: Скорость речи (слов в минуту)
            volume: Громкость (0.0 - 1.0)
        """
        self.engine = None
        self.rate = rate
        self.volume = volume
        self.is_speaking = False
        self.last_text = ""
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.stop_flag = False
        
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            # Получаем доступные голоса
            voices = self.engine.getProperty('voices')
            if len(voices) > 0:
                # Используем первый доступный голос
                self.engine.setProperty('voice', voices[0].id)
            
            # Запускаем поток для обработки очереди речи
            self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.worker_thread.start()
            
            print("Синтезатор речи инициализирован")
        except Exception as e:
            print(f"Ошибка инициализации синтезатора речи: {e}")
            self.engine = None
    
    def _speech_worker(self):
        """
        Рабочий поток для обработки очереди речевых запросов.
        """
        while not self.stop_flag:
            try:
                text = self.speech_queue.get(timeout=1)
                if text is None:
                    continue
                
                self.is_speaking = True
                if self.engine is not None:
                    self.engine.say(text)
                    self.engine.runAndWait()
                self.is_speaking = False
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка при озвучивании: {e}")
                self.is_speaking = False
    
    def speak(self, text, force=False):
        """
        Озвучивает текст.
        
        Args:
            text: Текст для озвучивания
            force: Если True, прерывает текущее озвучивание и говорит новый текст
        """
        if self.engine is None:
            print("Синтезатор речи не инициализирован")
            return
        
        if text == "" or text is None:
            return
        
        # Если это тот же текст и не force, не озвучиваем повторно
        if text == self.last_text and not force:
            return
        
        self.last_text = text
        
        # Если force, очищаем очередь
        if force:
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Добавляем в очередь
        try:
            self.speech_queue.put_nowait(text)
        except queue.Full:
            print("Очередь речи переполнена, пропускаем")
    
    def stop(self):
        """
        Останавливает синтезатор речи.
        """
        self.stop_flag = True
        if self.engine is not None:
            self.engine.stop()
        
        # Очищаем очередь
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
    
    def set_rate(self, rate):
        """
        Устанавливает скорость речи.
        
        Args:
            rate: Скорость речи (слов в минуту)
        """
        self.rate = rate
        if self.engine is not None:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume):
        """
        Устанавливает громкость.
        
        Args:
            volume: Громкость (0.0 - 1.0)
        """
        self.volume = volume
        if self.engine is not None:
            self.engine.setProperty('volume', volume)

