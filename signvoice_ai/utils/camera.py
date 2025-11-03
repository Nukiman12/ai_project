"""
Модуль для работы с веб-камерой через OpenCV.
"""

import cv2


class Camera:
    """
    Класс для работы с веб-камерой.
    Обеспечивает захват видеопотока и обработку кадров.
    """
    
    def __init__(self, camera_index=0, width=640, height=480):
        """
        Инициализация камеры.
        
        Args:
            camera_index: Индекс камеры (обычно 0 для основной)
            width: Ширина кадра
            height: Высота кадра
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_opened = False
    
    def open(self):
        """
        Открывает камеру для захвата видео.
        
        Returns:
            True если камера успешно открыта, False иначе
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Ошибка: не удалось открыть камеру {self.camera_index}")
            return False
        
        # Устанавливаем размер кадра
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.is_opened = True
        print(f"Камера {self.camera_index} успешно открыта ({self.width}x{self.height})")
        return True
    
    def read(self):
        """
        Читает один кадр с камеры.
        
        Returns:
            Кортеж (success, frame), где:
            - success: True если кадр прочитан успешно
            - frame: Массив кадра (numpy array)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """
        Освобождает ресурсы камеры.
        """
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("Камера закрыта")
    
    def __enter__(self):
        """Контекстный менеджер: открывает камеру"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: закрывает камеру"""
        self.release()

