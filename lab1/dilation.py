import time
import cv2
import numba
import numpy as np


# Делаем бинаризацию изображения (инвертировано)
def binarize_image(image_gray: np.ndarray, threshold_type: str = "adaptive") -> np.ndarray:
    if threshold_type == "otsu":
        _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # порог с адаптацией
        binary = cv2.adaptiveThreshold(
            image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
    return binary


# реализация через библиотеку OpenCV.
def dilate_library(binary_image: np.ndarray, kernel: np.ndarray = None, iterations: int = 1) -> np.ndarray:
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(binary_image, kernel, iterations=iterations)


# реализация векторной дилатации со структурным элементом 3*3
# считаем локальный максимум через сдвиги срезов.
def dilate_native_vectorized(binary_image: np.ndarray, kernel: np.ndarray = None, iterations: int = 1) -> np.ndarray:

    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)

    current_img = binary_image.copy()
    h, w = current_img.shape

    shifts = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if kernel[dy + 1, dx + 1]:
                shifts.append((dy, dx))

    for _ in range(iterations):
        padded = np.pad(current_img, 1, mode="constant", constant_values=0)
        output = np.zeros_like(current_img)
        for dy, dx in shifts:
            sub = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            output = np.maximum(output, sub)
        current_img = output

    return current_img

# Низкоуровневый обход пикселей вложенными циклами (попиксельная без векторов)
@numba.njit(fastmath=True)
def _dilate_loops_numba(padded: np.ndarray, kernel: np.ndarray, h: int, w: int) -> np.ndarray:

    output = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            max_val = 0
            for ky in range(3):
                for kx in range(3):
                    if kernel[ky, kx] != 0:
                        val = padded[y + ky, x + kx]
                        if val > max_val:
                            max_val = val
            output[y, x] = max_val
    return output

# реализация попиксельной дилатации (нативная)
def dilate_native_pixelwise(binary_image: np.ndarray, kernel: np.ndarray = None, iterations: int = 1) -> np.ndarray:

    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)

    current_img = binary_image.copy()
    h, w = current_img.shape

    for _ in range(iterations):
        padded = np.pad(current_img, 1, mode="constant", constant_values=0)
        current_img = _dilate_loops_numba(padded, kernel, h, w)

    return current_img
