import cv2
import numpy as np


# генерация тестового видео с текстурированным объектом
def create_synthetic_custom_video(output_path: str = "lab2/custom_test_video.avi", num_frames: int = 150):
    width, height = 640, 360
    fps = 30.0

    # текстура обложки
    obj_w, obj_h = 320, 240
    obj_img = np.zeros((obj_h, obj_w, 3), dtype=np.uint8)

    # градиентный фон
    for y in range(obj_h):
        for x in range(obj_w):
            obj_img[y, x] = [int(120 + 80 * (x / obj_w)), int(60 + 60 * (y / obj_h)), 180]

    # геометрические узоры и текст
    cv2.rectangle(obj_img, (20, 20), (obj_w - 20, obj_h - 20), (255, 255, 255), 3)
    cv2.circle(obj_img, (160, 120), 55, (0, 220, 255), -1)
    cv2.rectangle(obj_img, (130, 90), (190, 150), (255, 50, 50), -1)

    for i in range(10, obj_w - 10, 30):
        cv2.line(obj_img, (i, 20), (i, 50), (0, 255, 0), 2)
        cv2.line(obj_img, (i, obj_h - 50), (i, obj_h - 20), (255, 255, 0), 2)

    cv2.putText(obj_img, "COMPUTER", (45, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(obj_img, "VISION", (75, 210), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    # текстурный фон
    table_bg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        val = int(80 + 30 * np.sin(y / 8.0) + 15 * np.cos(y / 3.0))
        table_bg[y, :] = [val // 2, val, val + 40]

    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    table_bg = np.clip(table_bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # начальный крупный план
    frame0 = cv2.resize(obj_img, (width, height))
    out.write(frame0)

    src_corners = np.float32([[0, 0], [obj_w - 1, 0], [obj_w - 1, obj_h - 1], [0, obj_h - 1]])

    # кадры движения камеры
    for t in range(1, num_frames + 1):
        progress = t / float(num_frames)
        cx = 320 + 70 * np.sin(progress * 2 * np.pi)
        cy = 180 + 40 * np.cos(progress * 2 * np.pi)

        scale = 0.9 - 0.35 * np.sin(progress * np.pi)
        angle = np.radians(25 * np.sin(progress * 2 * np.pi))

        hw = (obj_w * scale) / 2.0
        hh = (obj_h * scale) / 2.0

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx_w, dy_w = hw * cos_a, hw * sin_a
        dx_h, dy_h = -hh * sin_a, hh * cos_a

        tilt = 0.15 * np.cos(progress * np.pi)

        dst_corners = np.float32(
            [
                [cx - dx_w - dx_h * (1 - tilt), cy - dy_w - dy_h * (1 - tilt)],
                [cx + dx_w - dx_h * (1 + tilt), cy + dy_w - dy_h * (1 + tilt)],
                [cx + dx_w + dx_h * (1 + tilt), cy + dy_w + dy_h * (1 + tilt)],
                [cx - dx_w + dx_h * (1 - tilt), cy - dy_w + dy_h * (1 - tilt)],
            ]
        )

        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        warped_obj = cv2.warpPerspective(obj_img, M, (width, height))

        obj_mask = cv2.warpPerspective(np.ones((obj_h, obj_w), dtype=np.uint8) * 255, M, (width, height))
        _, obj_mask = cv2.threshold(obj_mask, 128, 255, cv2.THRESH_BINARY)

        frame = table_bg.copy()
        frame[obj_mask > 0] = warped_obj[obj_mask > 0]

        # эффект смаза движения
        if 60 <= t <= 75:
            ksize = 17
            kernel_mb = np.zeros((ksize, ksize))
            kernel_mb[int((ksize - 1) / 2), :] = np.ones(ksize)
            kernel_mb /= ksize
            frame = cv2.filter2D(frame, -1, kernel_mb)

        out.write(frame)

    out.release()
    print(f"Видео сохранено: {output_path} ({num_frames + 1} кадров)")


if __name__ == "__main__":
    create_synthetic_custom_video()
