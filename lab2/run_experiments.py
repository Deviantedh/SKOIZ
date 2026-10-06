import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

from tracker import HybridTracker, KeypointTracker, draw_tracking_overlay


def evaluate_tracker(video_path: str, tracker_type: str, label: str = "Mona Lisa"):
    """
    Прогон трекера по видео с замером метрик:
    - процент отслеженных кадров;
    - среднее количество инлаеров/точек;
    - FPS обработки.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        return 0, 0, 0

    if tracker_type == "sift":
        tracker = KeypointTracker(method="sift", label=label)
    elif tracker_type == "orb":
        tracker = KeypointTracker(method="orb", label=label)
    elif tracker_type == "hybrid":
        tracker = HybridTracker(label=label)
    elif tracker_type == "klt_pure":
        # KLT без релокализации
        tracker = HybridTracker(label=label, enable_relocalization=False)

    tracker.set_template(frame0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    tracked_count = 0
    inliers_list = []

    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if tracker_type in ["hybrid", "klt_pure"]:
            tracked, corners, n_pts, inliers, _ = tracker.process_frame(frame)
        else:
            tracked, corners, n_pts, inliers = tracker.process_frame(frame)

        if tracked:
            tracked_count += 1
            inliers_list.append(n_pts)

    elapsed = time.perf_counter() - t0
    cap.release()

    fps = total_frames / elapsed if elapsed > 0 else 0
    ratio = (tracked_count / total_frames * 100) if total_frames > 0 else 0
    avg_inliers = float(np.mean(inliers_list)) if inliers_list else 0.0

    return ratio, avg_inliers, fps


def generate_visual_demo(video_path: str, output_image_path: str, label: str = "Mona Lisa"):
    """
    Генерирует коллаж из ключевых кадров видео с наложенной рамкой трекинга.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        return

    tracker = HybridTracker(label=label)
    tracker.set_template(frame0)

    target_indices = [0, 50, 180, 320, 480, 550]
    saved_frames = {}

    # Frame 0
    vis0 = draw_tracking_overlay(frame0, tracker.template_corners, label=label)
    saved_frames[0] = vis0

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        tracked, corners, n_pts, inliers, _ = tracker.process_frame(frame)
        vis = draw_tracking_overlay(frame, corners, inliers, label=label)
        if idx in target_indices:
            saved_frames[idx] = vis

    cap.release()

    # Сетка 2x3
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    keys = sorted(saved_frames.keys())[:6]

    descriptions = [
        "Кадр 0: Шаблон (крупный план)",
        "Кадр 50: Начало движения камеры",
        "Кадр 180: Поворот и отдаление",
        "Кадр 320: Сильное отдаление объекта",
        "Кадр 480: Возвращение и приближение",
        "Кадр 550: Финальная траектория",
    ]

    for ax, k, desc in zip(axes.ravel(), keys, descriptions):
        img_rgb = cv2.cvtColor(saved_frames[k], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(desc, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Результаты трекинга картины 'Мона Лиза' (Hybrid KLT + SIFT)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=180)
    plt.close()
    print(f"Коллаж трекинга сохранен: {output_image_path}")


def generate_blur_relocalization_demo(output_image_path: str):
    """
    Демонстрация срыва трекинга при размытии движения (motion blur)
    и мгновенной релокализации через ключевые точки.
    """
    cap = cv2.VideoCapture("test-videos/mona-lisa-blur-extra-credit.avi")
    ret, frame0 = cap.read()
    if not ret:
        return

    # Запускаем два трекера параллельно: Pure KLT и Hybrid KLT+SIFT
    klt = HybridTracker(label="Mona Lisa", enable_relocalization=False)
    klt.set_template(frame0)

    hybrid = HybridTracker(label="Mona Lisa", enable_relocalization=True)
    hybrid.set_template(frame0)

    frames_data = {}
    # Момент размытия приходится примерно на кадры 205-235
    capture_frames = [190, 215, 240]

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        tracked_klt, corners_klt, _, inliers_klt, _ = klt.process_frame(frame)
        tracked_hyb, corners_hyb, _, inliers_hyb, reloc_hyb = hybrid.process_frame(frame)

        if idx in capture_frames:
            frames_data[idx] = {
                "orig": frame.copy(),
                "klt": draw_tracking_overlay(frame, corners_klt, inliers_klt, label="Pure KLT"),
                "hyb": draw_tracking_overlay(frame, corners_hyb, inliers_hyb, label="Hybrid Tracker"),
                "tracked_klt": tracked_klt,
                "tracked_hyb": tracked_hyb,
                "reloc": reloc_hyb,
            }

    cap.release()

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    row_titles = [
        "До смаза (Кадр 190): Стабильное ведение",
        "Момент смаза (Кадр 215): Экстремальный Motion Blur",
        "После смаза (Кадр 240): Выход из зоны смаза",
    ]

    for row, f_idx in enumerate(capture_frames):
        d = frames_data[f_idx]
        # Левая колонка: Pure KLT
        ax_left = axes[row, 0]
        ax_left.imshow(cv2.cvtColor(d["klt"], cv2.COLOR_BGR2RGB))
        klt_label = "[Pure KLT: Рамка потеряна]" if not d["tracked_klt"] else "[Pure KLT: Отслеживание]"
        ax_left.set_title(f"{row_titles[row]}\n{klt_label}", fontsize=10, color="red" if not d["tracked_klt"] else "black")
        ax_left.axis("off")

        # Правая колонка: Hybrid
        ax_right = axes[row, 1]
        ax_right.imshow(cv2.cvtColor(d["hyb"], cv2.COLOR_BGR2RGB))
        hyb_label = "[Hybrid: Восстановлен через SIFT!]" if row == 2 else "[Hybrid: Отслеживание]"
        ax_right.set_title(f"{row_titles[row]}\n{hyb_label}", fontsize=10, color="green" if row == 2 else "black")
        ax_right.axis("off")

    plt.suptitle("Сравнение поведения при Motion Blur: Pure KLT vs Hybrid KLT+SIFT", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=180)
    plt.close()
    print(f"Демонстрация релокализации сохранена: {output_image_path}")


def generate_custom_demo(output_image_path: str):
    """
    Демонстрация работы на собственном синтезированном/записанном объекте.
    """
    video_path = "lab2/custom_test_video.avi"
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        return

    tracker = HybridTracker(label="Computer Vision")
    tracker.set_template(frame0)

    target_frames = [0, 30, 68, 120]
    saved = {0: draw_tracking_overlay(frame0, tracker.template_corners, label="Book Cover")}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        tracked, corners, n_pts, inliers, _ = tracker.process_frame(frame)
        if idx in target_frames:
            saved[idx] = draw_tracking_overlay(frame, corners, inliers, label="Book Cover")

    cap.release()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = [
        "Кадр 0 (Шаблон книги)",
        "Кадр 30 (Движение и ракурс)",
        "Кадр 68 (Размытие движения)",
        "Кадр 120 (Финал траектории)",
    ]
    for ax, k, title in zip(axes, target_frames, titles):
        ax.imshow(cv2.cvtColor(saved[k], cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Тестирование на собственном видео (обложка книги)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=180)
    plt.close()
    print(f"Демонстрация собственного видео сохранена: {output_image_path}")


def main():
    os.makedirs("assets/images", exist_ok=True)

    videos = [
        ("Оригинал (mona-lisa.avi)", "test-videos/mona-lisa.avi"),
        ("Размытие 1 (mona-lisa-blur.avi)", "test-videos/mona-lisa-blur.avi"),
        ("Размытие 2 (extra-credit.avi)", "test-videos/mona-lisa-blur-extra-credit.avi"),
    ]

    methods = [
        ("Pure SIFT", "sift"),
        ("Pure ORB", "orb"),
        ("Pure KLT", "klt_pure"),
        ("Hybrid (KLT+SIFT)", "hybrid"),
    ]

    results_table = {}

    print("=== Запуск сравнительного тестирования алгоритмов трекинга ===")
    for v_name, v_path in videos:
        results_table[v_name] = {}
        print(f"\nТестирование на видео: {v_name}")
        for m_name, m_type in methods:
            ratio, inliers, fps = evaluate_tracker(v_path, m_type)
            results_table[v_name][m_name] = {"ratio": ratio, "inliers": inliers, "fps": fps}
            print(f"  [{m_name:18s}] Отслежено: {ratio:5.1f}% | Ср. точек: {inliers:4.1f} | Скорость: {fps:4.1f} FPS")

    # Построение графика сравнения качества трекинга (% отслеженных кадров)
    labels = [v[0].split(" ")[0] for v in videos]
    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(11, 6))
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]

    for i, (m_name, _) in enumerate(methods):
        values = [results_table[v[0]][m_name]["ratio"] for v in videos]
        plt.bar(x + (i - 1.5) * width, values, width, label=m_name, color=colors[i])

    plt.ylabel("% успешно отслеженных кадров", fontsize=12)
    plt.xlabel("Тестовое видео", fontsize=12)
    plt.title("Сравнение устойчивости алгоритмов трекинга к Motion Blur", fontsize=14, fontweight="bold")
    plt.xticks(x, [v[0] for v in videos], fontsize=10)
    plt.ylim(0, 110)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    chart_path = "assets/images/lab2_comparison_chart.png"
    plt.savefig(chart_path, dpi=180)
    plt.close()
    print(f"График сравнения сохранен: {chart_path}")

    # Генерация наглядных коллажей
    print("\nГенерация демонстрационных кадров...")
    generate_visual_demo("test-videos/mona-lisa.avi", "assets/images/lab2_tracking_demo.png")
    generate_blur_relocalization_demo("assets/images/lab2_blur_relocalization.png")
    generate_custom_demo("assets/images/lab2_custom_demo.png")

    print("\nВсе эксперименты успешно завершены!")


if __name__ == "__main__":
    main()
