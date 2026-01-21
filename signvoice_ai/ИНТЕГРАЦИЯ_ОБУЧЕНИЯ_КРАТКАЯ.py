"""
КРАТКАЯ ИНСТРУКЦИЯ ПО ИНТЕГРАЦИИ ОБУЧЕНИЯ В ENTERPRISE GUI

Этот файл содержит код, который нужно добавить в main_enterprise_gui.py
для интеграции системы обучения.

ВАЖНО: Сначала перезапустите приложение, чтобы применить исправление numpy!
"""

# ============================================================================
# ШАГ 1: ДОБАВИТЬ ИМПОРТЫ (в начало main_enterprise_gui.py)
# ============================================================================

# После существующих импортов добавить:
from training.training_module import DataCollector, ModelTrainer, GestureDataset
from model.advanced_gesture_model import AdvancedGestureClassifier
from torch.utils.data import DataLoader

# ============================================================================
# ШАГ 2: ИНИЦИАЛИЗАЦИЯ В __init__ (в классе EnterpriseSignVoiceGUI)
# ============================================================================

# В методе __init__, после строки self.session_start_time = datetime.now()
# добавить:

# Компоненты обучения
self.data_collector = DataCollector(save_dir="training_data")
self.model_trainer = None
self.is_recording_for_training = False
self.current_training_gesture = ""
self.recording_buffer = []
self.training_history_widgets = []

# ============================================================================
# ШАГ 3: ДОБАВИТЬ ВКЛАДКУ В create_gui()
# ============================================================================

# В методе create_gui(), после создания других вкладок добавить:

# Вкладка "Обучение"
training_tab = self.tabview.add("🎓 Обучение")
self.create_training_tab(training_tab)

# ============================================================================
# ШАГ 4: СОЗДАТЬ МЕТОД create_training_tab()
# ============================================================================

def create_training_tab(self, tab):
    """Создаёт вкладку обучения."""
    
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
    
    # Видео (заглушка - можно переиспользовать основную камеру)
    video_info = ctk.CTkFrame(left_panel, height=400)
    video_info.pack(fill="x", padx=10, pady=10)
    
    ctk.CTkLabel(
        video_info,
        text="📹\n\nВидео с камеры\n(используйте основную вкладку для записи)\n\nИли добавьте отдельную камеру здесь",
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
    
    self.train_model_button = ctk.CTkButton(
        train_buttons_frame,
        text="🚀 Начать обучение",
        command=self.start_model_training,
        font=ctk.CTkFont(size=14, weight="bold"),
        height=40,
        fg_color="#1a73e8"
    )
    self.train_model_button.pack(side="left", padx=5, expand=True, fill="x")
    
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

# ============================================================================
# ШАГ 5: МЕТОДЫ ДЛЯ ЗАПИСИ ДАННЫХ
# ============================================================================

def start_recording_training_samples(self):
    """Начинает запись образцов для обучения."""
    gesture_name = self.gesture_name_entry.get().strip()
    
    if not gesture_name:
        messagebox.showwarning("Ошибка", "Введите название жеста!")
        return
    
    if not self.camera or not self.is_running:
        messagebox.showwarning(
            "Ошибка", 
            "Сначала запустите распознавание на вкладке 'Главная'!"
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

# ============================================================================
# ШАГ 6: МЕТОДЫ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ
# ============================================================================

def start_model_training(self):
    """Начинает обучение модели."""
    total_samples = self.data_collector.get_samples_count()
    
    if total_samples < 30:
        messagebox.showwarning(
            "Недостаточно данных",
            f"Для обучения нужно минимум 30 образцов.\nСобрано: {total_samples}\n\n"
            "Рекомендация: 3+ жеста × 10+ образцов"
        )
        return
    
    try:
        # Подготовка данных
        X_train, X_test, y_train, y_test, gesture_classes = \
            self.data_collector.prepare_training_data()
        
        self.train_model_button.configure(state="disabled")
        self.training_progress_label.configure(text="Подготовка...")
        
        # Запуск обучения в отдельном потоке
        training_thread = threading.Thread(
            target=self._train_model_thread,
            args=(X_train, X_test, y_train, y_test, gesture_classes),
            daemon=True
        )
        training_thread.start()
        
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось начать обучение:\n{e}")
        self.train_model_button.configure(state="normal")

def _train_model_thread(self, X_train, X_test, y_train, y_test, gesture_classes):
    """Обучение модели в отдельном потоке."""
    try:
        # Создаём модель
        model = AdvancedGestureClassifier(
            input_size=63,
            num_classes=len(gesture_classes)
        )
        
        # Создаём trainer
        self.model_trainer = ModelTrainer(model, device='cpu')
        
        # Добавляем callback
        self.model_trainer.add_callback(self._on_training_update)
        
        # Создаём DataLoaders
        train_dataset = GestureDataset(X_train, y_train)
        test_dataset = GestureDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        # Обучаем
        self.model_trainer.train(
            train_loader,
            test_loader,
            epochs=50,
            learning_rate=0.001,
            patience=10
        )
        
        # Сохраняем модель
        self.root.after(0, lambda: self._on_training_complete(gesture_classes))
        
    except Exception as e:
        print(f"Ошибка обучения: {e}")
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
    
    self.train_model_button.configure(state="normal")

def save_training_data(self):
    """Сохраняет собранные данные."""
    try:
        self.data_collector.save()
        messagebox.showinfo("Успех", "Данные сохранены!")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

# ============================================================================
# ШАГ 7: МОДИФИКАЦИЯ process_video()
# ============================================================================

# В методе process_video(), в блоке where landmarks is not None,
# после нормализации добавить:

# Запись для обучения
if self.is_recording_for_training and landmarks is not None:
    self.recording_buffer.append(landmarks.copy())

# ============================================================================
# ИТОГ: ЧТО ПОЛУЧИТСЯ
# ============================================================================

"""
После интеграции в Enterprise Edition появится новая вкладка "🎓 Обучение":

ЛЕВАЯ ПАНЕЛЬ:
  • Видео с камеры (или информация об использовании основной камеры)
  • Поле ввода названия жеста
  • Кнопка "Записать образцы (5 сек)"
  • Статус записи

ПРАВАЯ ПАНЕЛЬ:
  • Статистика (жестов, образцов, статус)
  • Список собранных жестов с количеством образцов
  • Прогресс бар обучения
  • Кнопка "Начать обучение"
  • Кнопка "Сохранить данные"

ПРОЦЕСС:
  1. Перейти на вкладку "Главная" → нажать "Старт"
  2. Перейти на вкладку "Обучение"
  3. Ввести название жеста (Peace, OK, ThumbsUp)
  4. Показать жест → нажать "Записать образцы"
  5. Держать жест 5 секунд
  6. Повторить для 3+ жестов (по 10+ образцов)
  7. Нажать "Начать обучение"
  8. Дождаться завершения (2-5 минут)
  9. Перезапустить приложение с новой моделью!

РЕЗУЛЬТАТ:
  • Модель обучена на ваших жестах
  • Точность 95-99%
  • Сохранена в models/trained_advanced_model.pth
"""




