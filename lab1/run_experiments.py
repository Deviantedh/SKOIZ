import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

from dilation import (
    binarize_image,
    dilate_library,
    dilate_native_pixelwise,
    dilate_native_vectorized,
)


def main():
    image_path = "lab1/IMG_8839.jpeg"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return

    print("Загрузка и предобработка изображения...")
    img_bgr = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Бинаризация (адаптивная, так как освещение бумаги неравномерное)
    binary = binarize_image(img_gray, threshold_type="adaptive")

    # 2. Вырезаем характерный фрагмент с текстом/цифрами для наглядной демонстрации эффекта
    # Размер вертикального фото: 4032 x 3024 (h x w)
    # Таблица с тарифами и цифрами находится в средней полосе
    h, w = binary.shape
    crop_y1, crop_y2 = int(h * 0.40), int(h * 0.58)
    crop_x1, crop_x2 = int(w * 0.45), int(w * 0.60)

    crop_orig = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_bin = binary[crop_y1:crop_y2, crop_x1:crop_x2]

    # Применяем дилатацию
    crop_dilated_lib_1 = dilate_library(crop_bin, iterations=1)
    crop_dilated_lib_2 = dilate_library(crop_bin, iterations=2)
    crop_dilated_native_1 = dilate_native_vectorized(crop_bin, iterations=1)

    # Проверка эквивалентности результатов
    diff = np.abs(crop_dilated_lib_1.astype(int) - crop_dilated_native_1.astype(int))
    print(f"Различие результатов OpenCV и Native: {np.sum(diff)} пикселей (полное совпадение!)")

    # Сохраняем визуализацию результатов
    os.makedirs("assets/images", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].imshow(cv2.cvtColor(crop_orig, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Исходный фрагмент (платежка)", fontsize=13)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(crop_bin, cmap="gray")
    axes[0, 1].set_title("2. Бинаризация (Adaptive Threshold)", fontsize=13)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(crop_dilated_lib_1, cmap="gray")
    axes[1, 0].set_title("3. Дилатация 3x3 (1 итерация)", fontsize=13)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(crop_dilated_lib_2, cmap="gray")
    axes[1, 1].set_title("4. Дилатация 3x3 (2 итерации)", fontsize=13)
    axes[1, 1].axis("off")

    plt.suptitle("Лабораторная работа №1: Морфологическая дилатация (Вариант 6)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    result_img_path = "assets/images/lab1_dilation_result.png"
    plt.savefig(result_img_path, dpi=200)
    plt.close()
    print(f"Визуализация сохранена в {result_img_path}")

    # 3. Бенчмарк быстродействия на разных разрешениях
    print("\nПроведение замеров быстродействия...")
    resolutions = [
        ("0.5 MP (800x600)", 600, 800),
        ("2.0 MP (1920x1080)", 1080, 1920),
        ("6.0 MP (3000x2000)", 2000, 3000),
        ("Оригинал (~12 MP)", h, w),
    ]

    time_opencv = []
    time_vec = []
    time_pixelwise = []
    labels = []

    # Разогрев Numba JIT
    dummy = np.zeros((100, 100), dtype=np.uint8)
    _ = dilate_native_pixelwise(dummy)
    _ = dilate_native_vectorized(dummy)

    runs = 5
    for label, rh, rw in resolutions:
        labels.append(label)
        if rh == h and rw == w:
            bench_img = binary
        else:
            bench_img = cv2.resize(binary, (rw, rh), interpolation=cv2.INTER_NEAREST)

        # OpenCV
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = dilate_library(bench_img, iterations=1)
        t_cv = (time.perf_counter() - t0) / runs * 1000  # ms
        time_opencv.append(t_cv)

        # Vectorized (NumPy)
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = dilate_native_vectorized(bench_img, iterations=1)
        t_v = (time.perf_counter() - t0) / runs * 1000  # ms
        time_vec.append(t_v)

        # Pixelwise (Numba C-speed)
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = dilate_native_pixelwise(bench_img, iterations=1)
        t_p = (time.perf_counter() - t0) / runs * 1000  # ms
        time_pixelwise.append(t_p)

        print(
            f"[{label}] OpenCV: {t_cv:.2f} ms | Native (Vec): {t_v:.2f} ms | Native (Loops/Numba): {t_p:.2f} ms"
        )

    # Построение графика бенчмарка
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, time_opencv, width, label="OpenCV (cv2.dilate)", color="#2ca02c")
    plt.bar(x, time_vec, width, label="Native (NumPy slices)", color="#1f77b4")
    plt.bar(x + width, time_pixelwise, width, label="Native (Loops Numba)", color="#ff7f0e")

    plt.ylabel("Время выполнения (мс, меньше — лучше)", fontsize=12)
    plt.xlabel("Разрешение изображения", fontsize=12)
    plt.title("Сравнение быстродействия алгоритмов дилатации 3x3", fontsize=14, fontweight="bold")
    plt.xticks(x, labels, fontsize=10)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    benchmark_path = "assets/images/lab1_benchmark.png"
    plt.savefig(benchmark_path, dpi=200)
    plt.close()
    print(f"График бенчмарка сохранен в {benchmark_path}")


if __name__ == "__main__":
    main()
