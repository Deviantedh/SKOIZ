import argparse  # Библиотека для разбора аргументов командной строки (флаги --video, --tracker и т.д.)
import sys       # Системные функции Python (выход из программы через sys.exit)
import time      # Функции для замера времени выполнения и расчёта FPS
import cv2       # Главная библиотека компьютерного зрения OpenCV
# Импортируем из соседнего файла tracker.py наши классы трекеров и функцию отрисовки
from tracker import HybridTracker, KeypointTracker, draw_tracking_overlay


def main():
    # Создаём парсер параметров командной строки с кратким описанием работы скрипта
    parser = argparse.ArgumentParser(description="Трекинг плоского объекта на видео по первому кадру.")
    
    # Флаг --video: путь к входному видеофайлу. По умолчанию берём тестовую Мона Лизу
    parser.add_argument("--video", type=str, default="test-videos/mona-lisa.avi", help="Путь к видео")
    
    # Флаг --tracker: выбор одного из трёх алгоритмов трекинга (гибридный, чистый SIFT или чистый ORB)
    parser.add_argument(
        "--tracker", type=str, default="hybrid", choices=["hybrid", "sift", "orb"], help="Тип алгоритма трекинга"
    )
    
    # Флаг --label: текстовая плашка, которая будет написана над зелёной рамкой (по умолчанию "Object")
    parser.add_argument("--label", type=str, default="Object", help="Подпись для найденного объекта")
    
    # Флаг --output: если указан путь (например output.avi), скрипт запишет обработанное видео с рамкой
    parser.add_argument("--output", type=str, default=None, help="Путь для сохранения размеченного видео")
    
    # Флаг --show: если передать его в терминале, появится графическое окно с показом видео в реальном времени
    parser.add_argument("--show", action="store_true", help="Отображать окно воспроизведения (GUI)")

    # Считываем переданные пользователем аргументы из sys.argv
    args = parser.parse_args()

    # Печатаем путь к файлу для наглядности в консоли
    print(f"Открытие видео: {args.video}...")
    
    # Создаём объект захвата видеопотока из указанного файла
    cap = cv2.VideoCapture(args.video)
    
    # Считываем самый первый кадр (кадр 0).
    # ret — булев флаг успеха чтения (True/False), frame0 — сам массив изображения (BGR)
    ret, frame0 = cap.read()
    
    # Если видео не открылось или повреждено — выводим ошибку и завершаем скрипт
    if not ret:
        print(f"Ошибка: не удалось прочитать первый кадр видео '{args.video}'.")
        sys.exit(1)

    # Инициализируем выбранный алгоритм трекинга в зависимости от значения аргумента --tracker
    if args.tracker == "hybrid":
        # Создаём наш продвинутый гибридный трекер (KLT оптический поток + SIFT-релокализатор)
        tracker = HybridTracker(label=args.label)
    else:
        # Создаём классический трекер на сопоставлении точек (либо SIFT, либо ORB)
        tracker = KeypointTracker(method=args.tracker, label=args.label)

    # Передаём первый кадр трекеру в качестве эталона (шаблона объекта).
    # Трекер запоминает этот кадр, находит на нём ключевые точки и вычисляет их дескрипторы.
    tracker.set_template(frame0)
    print(f"Шаблон объекта успешно инициализирован (метод: {args.tracker.upper()}).")

    # Считываем параметры видеопотока:
    # Частота кадров (FPS), если видеопоток не отдал значение — ставим 30.0 по умолчанию
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Ширина кадра в пикселях
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Высота кадра в пикселях
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Общее количество кадров в видеофайле
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Переменная для записи видеофайла на диск (если пользователь указал флаг --output)
    writer = None
    if args.output:
        # Выбираем кодек сжатия MJPG (Motion JPEG — поддерживается везде без сторонних dll)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # Создаём объект для записи результирующего видеофайла с теми же размерами и FPS
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Сохранение видео в: {args.output}")

    # Счётчик кадров, на которых объект был успешно найден и обведён
    tracked_count = 0
    # Номер текущего кадра (начинаем с нуля)
    frame_idx = 0
    # Засекаем текущее время для точного замера скорости работы (FPS)
    t_start = time.time()

    # Отрисовываем рамку на первом кадре (шаблоне) вокруг объекта
    vis0 = draw_tracking_overlay(frame0, tracker.template_corners if hasattr(tracker, 'template_corners') else None, label=args.label)
    # Если включена запись в файл — сохраняем туда размеченный первый кадр
    if writer:
        writer.write(vis0)

    # Главный цикл покадровой обработки видео
    while True:
        # Читаем следующий кадр из видеофайла
        ret, frame = cap.read()
        # Если кадры закончились (конец видео) или произошла ошибка — прерываем цикл
        if not ret:
            break
        # Увеличиваем счётчик обработанных кадров
        frame_idx += 1

        # Обрабатываем кадр алгоритмом трекинга:
        if args.tracker == "hybrid":
            # Гибридный трекер возвращает 5 значений:
            # tracked (успех/неудача), corners (4 угла рамки), n_pts (число точек),
            # inliers (координаты точек-инлаеров), reloc (сработала ли релокализация SIFT)
            tracked, corners, n_pts, inliers, reloc = tracker.process_frame(frame)
        else:
            # Обычный KeypointTracker возвращает 4 значения:
            tracked, corners, inliers_cnt, inliers = tracker.process_frame(frame)

        # Если объект на кадре успешно обнаружен — увеличиваем счётчик успешных кадров
        if tracked:
            tracked_count += 1

        # Наносим на изображение зелёную рамку, точки-инлаеры и красивую плашку с названием объекта
        vis = draw_tracking_overlay(frame, corners, inliers, label=args.label)

        # Если включена запись — записываем размеченный кадр в итоговый видеофайл
        if writer:
            writer.write(vis)

        # Если включен показ в окне — выводим изображение на экран
        if args.show:
            cv2.imshow("Tracking", vis)
            # Ждём 1 миллисекунду нажатия клавиши. Если нажат Escape (код 27) или 'q' — выходим досрочно
            if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                break

        # Каждые 50 кадров печатаем прогресс в терминал, чтобы видеть ход работы
        if frame_idx % 50 == 0:
            print(f"Обработано кадров: {frame_idx}/{total_frames} | Распознано: {tracked_count}")

    # Вычисляем общее затраченное время в секундах
    elapsed = time.time() - t_start
    # Освобождаем захваченный видеофайл
    cap.release()
    # Закрываем и сохраняем результирующий видеофайл (если он писался)
    if writer:
        writer.release()
    # Закрываем все графические окна OpenCV (если они открывались)
    if args.show:
        cv2.destroyAllWindows()

    # Печатаем итоговую статистику трекинга
    print("\n* Результаты трекинга ")
    print(f"Всего кадров: {frame_idx + 1}")
    # Процент успешно отслеженных кадров от общего числа
    print(f"Отслежено кадров: {tracked_count + 1} ({(tracked_count + 1) / (frame_idx + 1) * 100:.1f}%)")
    # Реальная средняя скорость обработки видео (кадров в секунду)
    print(f"Средняя скорость обработки: {(frame_idx + 1) / elapsed:.1f} FPS")


# Стандартная точка входа в скрипт Python
if __name__ == "__main__":
    main()
