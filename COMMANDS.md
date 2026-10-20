# Команды для запуска лабораторных работ

В этом файле собраны готовые команды для терминала. Все команды запускаются из корневой папки проекта (`СКОИЗ`).

---

## Лабораторная работа 2: Трекинг объекта на видео

### 1. Автоматический прогон всех тестов и генерация отчетов
Запуск полного набора экспериментов (сравнение SIFT, ORB, KLT и Гибрида на всех тестовых видео + собственном видео, построение графиков и таблиц):
```bash
python3 lab2/run_experiments.py
```
*Результаты сохраняются в папку `lab2/results/`.*

---

### 2. Запуск трекинга с выводом на экран в реальном времени (GUI)

Для выхода из окна воспроизведения нажмите клавишу `q`.

#### Видео 1: Мона Лиза (базовый тест с отдалением и приближением)
```bash
python3 lab2/track_video.py --video test-videos/mona-lisa.avi --tracker hybrid --show
```

#### Видео 2: Мона Лиза со смазом движения (Motion Blur)
*Наглядно демонстрирует, как при смазе срабатывает релокализация:*
```bash
python3 lab2/track_video.py --video test-videos/mona-lisa-blur.avi --tracker hybrid --show
```

#### Видео 3: Сложное видео (сильные повороты и резкие движения)
```bash
python3 lab2/track_video.py --video test-videos/mona-lisa-blur-extra-credit.avi --tracker hybrid --show
```

#### Видео 4: Собственное записанное видео (с книгой / документом)
```bash
python3 lab2/track_video.py --video lab2/IMG_1707.MOV --tracker hybrid --label "Book" --show
```

---

### 3. Демонстрация разницы алгоритмов преподавателю

Чтобы наглядно показать разницу между методами на видео со смазом (`mona-lisa-blur.avi`):

1. **Гибридный трекер (KLT + SIFT):** высокая скорость и восстановление рамки после смаза:
   ```bash
   python3 lab2/track_video.py --video test-videos/mona-lisa-blur.avi --tracker hybrid --show
   ```

2. **Чистый SIFT:** рамка держится стабильно, но FPS заметно ниже (~45 FPS):
   ```bash
   python3 lab2/track_video.py --video test-videos/mona-lisa-blur.avi --tracker sift --show
   ```

3. **Чистый ORB:** работает быстро, но при смазе и наклонах рамка срывается:
   ```bash
   python3 lab2/track_video.py --video test-videos/mona-lisa-blur.avi --tracker orb --show
   ```

---

### 4. Сохранение обработанного видео в файл

Если нужно сохранить видео с отрисованной рамкой в папку результатов:

```bash
# Мона Лиза
python3 lab2/track_video.py --video test-videos/mona-lisa.avi --tracker hybrid --output lab2/results/mona-lisa_tracked.mp4

# Мона Лиза с размытием
python3 lab2/track_video.py --video test-videos/mona-lisa-blur.avi --tracker hybrid --output lab2/results/mona-lisa-blur_tracked.mp4

# Extra Credit
python3 lab2/track_video.py --video test-videos/mona-lisa-blur-extra-credit.avi --tracker hybrid --output lab2/results/mona-lisa-blur-extra-credit_tracked.mp4

# Собственное видео
python3 lab2/track_video.py --video lab2/IMG_1707.MOV --tracker hybrid --label "Book" --output lab2/results/custom_video_tracked.mp4
```

---

## Лабораторная работа 1: Морфологическая дилатация

### Запуск тестов и замер производительности всех реализаций
Сравнение скорости (попиксельный цикл, векторизованный NumPy, Numba JIT и OpenCV):
```bash
python3 lab1/run_experiments.py
```
*Результаты сохраняются в папку `lab1/results/`.*

---

## Запуск из вложенных папок

Если терминал уже открыт внутри конкретной папки, команды выглядят так:

**Внутри папки `lab2`:**
```bash
cd lab2
python3 track_video.py --video ../test-videos/mona-lisa.avi --tracker hybrid --show
python3 run_experiments.py
```

**Внутри папки `lab1`:**
```bash
cd lab1
python3 run_experiments.py
```
