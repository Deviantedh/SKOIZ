import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

from tracker import HybridTracker, KeypointTracker, draw_tracking_overlay


# оценка метрик трекера по видео
def evaluate_tracker(video_path: str, tracker_type: str, label: str = "Mona Lisa"):
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
        # klt без релокализации
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


# генерация коллажа трекинга
def generate_visual_demo(video_path: str, output_image_path: str, label: str = "Mona Lisa"):
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        return

    tracker = HybridTracker(label=label)
    tracker.set_template(frame0)

    target_indices = [0, 50, 180, 320, 480, 550]
    saved_frames = {}

    # начальный кадр
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    keys = sorted(saved_frames.keys())[:6]

    descriptions = [
        "Кадр 0",
        "Кадр 50",
        "Кадр 180",
        "Кадр 320",
        "Кадр 480",
        "Кадр 550",
    ]

    for ax, k, desc in zip(axes.ravel(), keys, descriptions):
        img_rgb = cv2.cvtColor(saved_frames[k], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(desc, fontsize=11)
        ax.axis("off")

    plt.suptitle("Результаты трекинга (Hybrid KLT + SIFT)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=180)
    plt.close()
    print(f"Коллаж трекинга сохранен: {output_image_path}")


# сравнение klt и гибридного трекера при смазе
def generate_blur_relocalization_demo(output_image_path: str):
    cap = cv2.VideoCapture("test-videos/mona-lisa-blur-extra-credit.avi")
    ret, frame0 = cap.read()
    if not ret:
        return

    # klt без релокализации и гибридный трекер
    klt = HybridTracker(label="Mona Lisa", enable_relocalization=False)
    klt.set_template(frame0)

    hybrid = HybridTracker(label="Mona Lisa", enable_relocalization=True)
    hybrid.set_template(frame0)

    frames_data = {}
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
        "Кадр 190",
        "Кадр 215",
        "Кадр 240",
    ]

    for row, f_idx in enumerate(capture_frames):
        d = frames_data[f_idx]
        # Pure KLT
        ax_left = axes[row, 0]
        ax_left.imshow(cv2.cvtColor(d["klt"], cv2.COLOR_BGR2RGB))
        ax_left.set_title(f"{row_titles[row]} (Pure KLT)", fontsize=11)
        ax_left.axis("off")

        # Hybrid
        ax_right = axes[row, 1]
        ax_right.imshow(cv2.cvtColor(d["hyb"], cv2.COLOR_BGR2RGB))
        ax_right.set_title(f"{row_titles[row]} (Hybrid)", fontsize=11)
        ax_right.axis("off")

    plt.suptitle("Поведение при размытии: Pure KLT vs Hybrid KLT+SIFT", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=180)
    plt.close()
    print(f"Демонстрация релокализации сохранена: {output_image_path}")


# тест на собственном записанном видео
def generate_custom_demo(output_image_path: str):
    video_path = "lab2/IMG_1707.MOV"
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        return

    # координаты памятки на первом кадре
    corners0 = np.float32([[925, 332], [1302, 441], [1195, 660], [805, 545]]).reshape(-1, 1, 2)
    tracker = HybridTracker(label="Document")
    tracker.set_template(frame0, corners=corners0)

    target_frames = [0, 50, 380, 620]
    saved = {0: draw_tracking_overlay(frame0, corners0, label="Document")}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        tracked, corners, n_pts, inliers, _ = tracker.process_frame(frame)
        if idx in target_frames:
            saved[idx] = draw_tracking_overlay(frame, corners, inliers, label="Document")

    cap.release()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = [
        "Кадр 0 (Шаблон документа)",
        "Кадр 50 (Движение и ракурс)",
        "Кадр 380 (Отдаление камеры)",
        "Кадр 620 (Поворот и наклон)",
    ]
    for ax, k, title in zip(axes, target_frames, titles):
        ax.imshow(cv2.cvtColor(saved[k], cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.suptitle("Тестирование на собственном видео (памятка на ткани)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=180)
    plt.close()
    print(f"Демонстрация собственного видео сохранена: {output_image_path}")


# сохранение размеченного видео с рамкой
def render_and_save_video(video_path: str, output_path: str, label: str = "Object", corners0: np.ndarray = None):
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        print(f"Ошибка чтения видео: {video_path}")
        return

    tracker = HybridTracker(label=label)
    tracker.set_template(frame0, corners=corners0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # используем mp4v для компактного размера и совместимости
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    vis0 = draw_tracking_overlay(frame0, tracker.template_corners, label=label)
    writer.write(vis0)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        tracked, corners, n_pts, inliers, _ = tracker.process_frame(frame)
        vis = draw_tracking_overlay(frame, corners, inliers, label=label)
        writer.write(vis)

    cap.release()
    writer.release()
    print(f"Видео сохранено: {output_path} ({count + 1} кадров)")


def main():
    os.makedirs("lab2/results", exist_ok=True)

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

    print("* Запуск тестирования алгоритмов трекинга")
    for v_name, v_path in videos:
        results_table[v_name] = {}
        print(f"\nВидео: {v_name}")
        for m_name, m_type in methods:
            ratio, inliers, fps = evaluate_tracker(v_path, m_type)
            results_table[v_name][m_name] = {"ratio": ratio, "inliers": inliers, "fps": fps}
            print(f"  [{m_name:18s}] Отслежено: {ratio:5.1f}% | Точек: {inliers:4.1f} | Скорость: {fps:4.1f} FPS")

    # график сравнения
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
    plt.title("Сравнение устойчивости алгоритмов трекинга", fontsize=13)
    plt.xticks(x, [v[0] for v in videos], fontsize=10)
    plt.ylim(0, 110)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    chart_path = "lab2/results/lab2_comparison_chart.png"
    plt.savefig(chart_path, dpi=180)
    plt.close()
    print(f"График сравнения сохранен: {chart_path}")

    # сохранение кадров
    print("\nГенерация демонстрационных кадров...")
    generate_visual_demo("test-videos/mona-lisa.avi", "lab2/results/lab2_tracking_demo.png")
    generate_blur_relocalization_demo("lab2/results/lab2_blur_relocalization.png")
    generate_custom_demo("lab2/results/lab2_custom_demo.png")

    # сохранение размеченных видео с рамкой в папку результатов
    print("\nСохранение размеченных видео с рамкой в lab2/results/...")
    render_and_save_video("test-videos/mona-lisa.avi", "lab2/results/mona-lisa_tracked.mp4", label="Mona Lisa")
    render_and_save_video("test-videos/mona-lisa-blur.avi", "lab2/results/mona-lisa-blur_tracked.mp4", label="Mona Lisa")
    render_and_save_video("test-videos/mona-lisa-blur-extra-credit.avi", "lab2/results/mona-lisa-blur-extra-credit_tracked.mp4", label="Mona Lisa")

    corners_custom = np.float32([[925, 332], [1302, 441], [1195, 660], [805, 545]]).reshape(-1, 1, 2)
    render_and_save_video("lab2/IMG_1707.MOV", "lab2/results/custom_video_tracked.mp4", label="Document", corners0=corners_custom)

    print("\nЭксперименты завершены")


if __name__ == "__main__":
    main()
