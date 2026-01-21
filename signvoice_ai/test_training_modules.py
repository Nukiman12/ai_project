"""
Диагностический скрипт для проверки доступности модулей обучения.
"""

print("=" * 70)
print("ДИАГНОСТИКА МОДУЛЕЙ ОБУЧЕНИЯ")
print("=" * 70)

# 1. Проверка PyTorch
print("\n[1/5] Проверка PyTorch...")
try:
    import torch
    print(f"  ✅ PyTorch установлен (версия {torch.__version__})")
except ImportError as e:
    print(f"  ❌ PyTorch НЕ установлен: {e}")
    print("  → Установите: pip install torch")

# 2. Проверка scikit-learn
print("\n[2/5] Проверка scikit-learn...")
try:
    import sklearn
    print(f"  ✅ scikit-learn установлен (версия {sklearn.__version__})")
except ImportError as e:
    print(f"  ❌ scikit-learn НЕ установлен: {e}")
    print("  → Установите: pip install scikit-learn")

# 3. Проверка advanced_gesture_model
print("\n[3/5] Проверка advanced_gesture_model...")
try:
    from model.advanced_gesture_model import AdvancedGestureClassifier
    print("  ✅ AdvancedGestureClassifier импортирован")
    
    # Пробуем создать модель
    model = AdvancedGestureClassifier(input_size=63, num_classes=3)
    print(f"  ✅ Модель создана успешно")
except ImportError as e:
    print(f"  ❌ Не удалось импортировать: {e}")
except Exception as e:
    print(f"  ❌ Ошибка создания модели: {e}")

# 4. Проверка training_module
print("\n[4/5] Проверка training_module...")
try:
    from training.training_module import DataCollector, ModelTrainer, GestureDataset
    print("  ✅ DataCollector импортирован")
    print("  ✅ ModelTrainer импортирован")
    print("  ✅ GestureDataset импортирован")
    
    # Пробуем создать DataCollector
    collector = DataCollector(save_dir="test_data")
    print(f"  ✅ DataCollector создан успешно")
except ImportError as e:
    print(f"  ❌ Не удалось импортировать: {e}")
except Exception as e:
    print(f"  ❌ Ошибка создания DataCollector: {e}")

# 5. Проверка torch.utils.data
print("\n[5/5] Проверка torch.utils.data.DataLoader...")
try:
    from torch.utils.data import DataLoader
    print("  ✅ DataLoader доступен")
except ImportError as e:
    print(f"  ❌ DataLoader недоступен: {e}")

print("\n" + "=" * 70)
print("ИТОГ")
print("=" * 70)

# Финальная проверка как в main_enterprise_gui.py
try:
    from training.training_module import DataCollector, ModelTrainer, GestureDataset
    from model.advanced_gesture_model import AdvancedGestureClassifier
    from torch.utils.data import DataLoader
    print("\n✅ ВСЕ МОДУЛИ ДОСТУПНЫ!")
    print("   TRAINING_AVAILABLE = True")
    print("\n   Кнопка обучения должна работать!")
except ImportError as e:
    print(f"\n❌ МОДУЛИ НЕДОСТУПНЫ!")
    print(f"   TRAINING_AVAILABLE = False")
    print(f"   Ошибка: {e}")
    print("\n   Кнопка обучения НЕ РАБОТАЕТ!")
    print("\n   РЕШЕНИЕ:")
    print("   1. Установите PyTorch: pip install torch")
    print("   2. Установите scikit-learn: pip install scikit-learn")
    print("   3. Перезапустите приложение")

print("\n" + "=" * 70)


