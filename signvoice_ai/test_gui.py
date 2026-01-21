"""
Тестовый файл для проверки GUI
"""

print("=" * 60)
print("Тест запуска GUI...")
print("=" * 60)

try:
    print("\n[1/5] Проверка customtkinter...")
    import customtkinter as ctk
    print("✓ CustomTkinter OK")
    
    print("\n[2/5] Проверка PIL...")
    from PIL import Image, ImageTk
    print("✓ PIL OK")
    
    print("\n[3/5] Проверка cv2...")
    import cv2
    print("✓ OpenCV OK")
    
    print("\n[4/5] Проверка mediapipe...")
    import mediapipe as mp
    print("✓ Mediapipe OK")
    
    print("\n[5/5] Проверка torch...")
    import torch
    print("✓ PyTorch OK")
    
    print("\n" + "=" * 60)
    print("Все зависимости установлены!")
    print("=" * 60)
    
    print("\nСоздаем тестовое окно...")
    
    # Создаем простое тестовое окно
    root = ctk.CTk()
    root.title("Тест SignVoiceAI")
    root.geometry("400x200")
    
    label = ctk.CTkLabel(
        root,
        text="✅ GUI работает!\n\nЕсли вы видите это окно,\nзначит CustomTkinter установлен правильно.",
        font=ctk.CTkFont(size=14)
    )
    label.pack(expand=True, pady=20)
    
    button = ctk.CTkButton(
        root,
        text="Закрыть",
        command=root.destroy,
        width=200,
        height=40
    )
    button.pack(pady=10)
    
    print("✓ Окно создано! Проверьте экран...")
    print("\nЕсли окно не появилось, проверьте:")
    print("  - Alt+Tab для переключения между окнами")
    print("  - Панель задач внизу экрана")
    print("  - Возможно окно за другими окнами")
    
    root.mainloop()
    
    print("\n✓ Окно закрыто успешно")
    
except ImportError as e:
    print(f"\n❌ ОШИБКА: Не установлен модуль - {e}")
    print("\nУстановите зависимости:")
    print("  pip install -r requirements.txt")
    
except Exception as e:
    print(f"\n❌ ОШИБКА: {e}")
    import traceback
    traceback.print_exc()




