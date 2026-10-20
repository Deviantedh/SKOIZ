import argparse
import sys
import time
import cv2
from tracker import HybridTracker, KeypointTracker, draw_tracking_overlay


def main():
    parser = argparse.ArgumentParser(description="Трекинг плоского объекта на видео по первому кадру.")
    parser.add_argument("--video", type=str, default="test-videos/mona-lisa.avi", help="Путь к видеофайлу")
    parser.add_argument(
        "--tracker", type=str, default="hybrid", choices=["hybrid", "sift", "orb"], help="Тип алгоритма трекинга"
    )
    parser.add_argument("--label", type=str, default="Object", help="Подпись для найденного объекта")
    parser.add_argument("--output", type=str, default=None, help="Путь для сохранения размеченного видео")
    parser.add_argument("--show", action="store_true", help="Отображать окно воспроизведения (GUI)")

    args = parser.parse_args()

    print(f"Открытие видео: {args.video}...")
    cap = cv2.VideoCapture(args.video)
    ret, frame0 = cap.read()
    if not ret:
        print(f"Ошибка: не удалось прочитать первый кадр видео '{args.video}'.")
        sys.exit(1)

    # Инициализация трекера
    if args.tracker == "hybrid":
        tracker = HybridTracker(label=args.label)
    else:
        tracker = KeypointTracker(method=args.tracker, label=args.label)

    tracker.set_template(frame0)
    print(f"Шаблон объекта успешно инициализирован (метод: {args.tracker.upper()}).")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Сохранение видео в: {args.output}")

    tracked_count = 0
    frame_idx = 0
    t_start = time.time()

    # Первый кадр
    vis0 = draw_tracking_overlay(frame0, tracker.template_corners if hasattr(tracker, 'template_corners') else None, label=args.label)
    if writer:
        writer.write(vis0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if args.tracker == "hybrid":
            tracked, corners, n_pts, inliers, reloc = tracker.process_frame(frame)
        else:
            tracked, corners, inliers_cnt, inliers = tracker.process_frame(frame)

        if tracked:
            tracked_count += 1

        vis = draw_tracking_overlay(frame, corners, inliers, label=args.label)

        if writer:
            writer.write(vis)

        if args.show:
            cv2.imshow("Tracking", vis)
            if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                break

        if frame_idx % 50 == 0:
            print(f"Обработано кадров: {frame_idx}/{total_frames} | Распознано: {tracked_count}")

    elapsed = time.time() - t_start
    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print("\n* Результаты трекинга ")
    print(f"Всего кадров: {frame_idx + 1}")
    print(f"Отслежено кадров: {tracked_count + 1} ({(tracked_count + 1) / (frame_idx + 1) * 100:.1f}%)")
    print(f"Средняя скорость обработки: {(frame_idx + 1) / elapsed:.1f} FPS")


if __name__ == "__main__":
    main()
