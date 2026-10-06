import cv2
import numpy as np


def draw_styled_label(frame: np.ndarray, text: str, pt: tuple, font_scale: float = 0.8):
    """
    Отрисовка текста с черным контуром для идеальной читаемости на любом фоне.
    """
    x, y = int(pt[0]), int(pt[1])
    # Ограничение по границам кадра
    x = max(10, min(x, frame.shape[1] - 150))
    y = max(25, min(y, frame.shape[0] - 10))

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Черная обводка
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    # Белый текст
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)


class KeypointTracker:
    """
    Детекторный трекер на основе сопоставления ключевых точек (SIFT / ORB).
    На каждом кадре ищет объект путем сопоставления с шаблоном (Frame 0)
    и оценки проективного преобразования (гомографии) методом RANSAC.
    """

    def __init__(self, method: str = "sift", label: str = "Object", min_inliers: int = 8):
        self.method = method.lower()
        self.label = label
        self.min_inliers = min_inliers

        if self.method == "sift":
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
            self.ratio_thresh = 0.75
        else:
            self.detector = cv2.ORB_create(nfeatures=1500)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            self.ratio_thresh = 0.8

        self.template_kp = None
        self.template_des = None
        self.template_corners = None
        self.h0, self.w0 = 0, 0

    def set_template(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        self.h0, self.w0 = gray.shape[:2]
        self.template_kp, self.template_des = self.detector.detectAndCompute(gray, None)
        self.template_corners = np.float32(
            [[0, 0], [self.w0 - 1, 0], [self.w0 - 1, self.h0 - 1], [0, self.h0 - 1]]
        ).reshape(-1, 1, 2)

    def process_frame(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        kp, des = self.detector.detectAndCompute(gray, None)

        tracked = False
        projected_corners = None
        inliers_count = 0
        inlier_points = []

        if des is not None and self.template_des is not None and len(kp) >= 4:
            matches = self.matcher.knnMatch(self.template_des, des, k=2)
            good = [
                m
                for match in matches
                if len(match) == 2
                for m, n in [match]
                if m.distance < self.ratio_thresh * n.distance
            ]

            if len(good) >= self.min_inliers:
                src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None and mask is not None:
                    inliers_count = int(np.sum(mask))
                    if inliers_count >= self.min_inliers:
                        proj = cv2.perspectiveTransform(self.template_corners, H)
                        area = cv2.contourArea(np.int32(proj))
                        # Геометрическая валидация: выпуклость и разумная площадь
                        if cv2.isContourConvex(np.int32(proj)) and 500 < area < (self.w0 * self.h0 * 2.0):
                            projected_corners = proj
                            tracked = True
                            mask_flat = mask.ravel()
                            inlier_points = [dst_pts[i][0] for i in range(len(mask_flat)) if mask_flat[i] == 1]

        return tracked, projected_corners, inliers_count, inlier_points


class HybridTracker:
    """
    Гибридный трекер: KLT (оптический поток Лукаса-Канаде) для плавного и
    быстрого покадрового отслеживания + SIFT-релокализатор для автоматического
    восстановления при размытии движения (motion blur) и срыве трекинга.
    """

    def __init__(self, label: str = "Object", enable_relocalization: bool = True):
        self.label = label
        self.enable_relocalization = enable_relocalization
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

        self.template_gray = None
        self.template_kp = None
        self.template_des = None
        self.template_corners = None
        self.w0, self.h0 = 0, 0

        self.prev_gray = None
        self.p0 = None
        self.current_corners = None
        self.is_tracking = False
        self.reloc_counter = 0

    def set_template(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        self.template_gray = gray
        self.h0, self.w0 = gray.shape[:2]
        self.template_kp, self.template_des = self.sift.detectAndCompute(gray, None)
        self.template_corners = np.float32(
            [[0, 0], [self.w0 - 1, 0], [self.w0 - 1, self.h0 - 1], [0, self.h0 - 1]]
        ).reshape(-1, 1, 2)
        self.current_corners = self.template_corners.copy()

        # Инициализируем ключевые точки для KLT
        self.p0 = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=8)
        self.prev_gray = gray
        self.is_tracking = True

    def process_frame(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        tracked = False
        relocalized = False
        inlier_points = []

        # 1. Попытка покадрового трекинга через KLT (Optical Flow)
        if self.is_tracking and self.p0 is not None and len(self.p0) >= 10:
            p1, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.p0,
                None,
                winSize=(25, 25),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            good_new = p1[status == 1]
            good_old = self.p0[status == 1]

            if len(good_new) >= 10:
                H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 3.0)
                if H is not None:
                    cand = cv2.perspectiveTransform(self.current_corners, H)
                    area = cv2.contourArea(np.int32(cand))
                    if cv2.isContourConvex(np.int32(cand)) and 500 < area < (self.w0 * self.h0 * 2.0):
                        self.current_corners = cand
                        tracked = True
                        if mask is not None:
                            inliers = good_new[mask.ravel() == 1]
                            self.p0 = inliers.reshape(-1, 1, 2)
                            inlier_points = inliers
                        else:
                            self.p0 = good_new.reshape(-1, 1, 2)
                            inlier_points = good_new

        # 2. Если трекинг потерян или точек мало — запускаем SIFT-релокализацию (если включена)
        if self.enable_relocalization and (not tracked or (self.p0 is not None and len(self.p0) < 15)):
            kp_curr, des_curr = self.sift.detectAndCompute(gray, None)
            if des_curr is not None and len(kp_curr) >= 4:
                matches = self.bf.knnMatch(self.template_des, des_curr, k=2)
                good = [
                    m
                    for match in matches
                    if len(match) == 2
                    for m, n in [match]
                    if m.distance < 0.75 * n.distance
                ]

                if len(good) >= 8:
                    src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H_reloc, mask_reloc = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if H_reloc is not None and mask_reloc is not None:
                        inliers_cnt = int(np.sum(mask_reloc))
                        if inliers_cnt >= 8:
                            cand = cv2.perspectiveTransform(self.template_corners, H_reloc)
                            area = cv2.contourArea(np.int32(cand))
                            if cv2.isContourConvex(np.int32(cand)) and 500 < area < (self.w0 * self.h0 * 2.0):
                                self.current_corners = cand
                                tracked = True
                                relocalized = True
                                self.reloc_counter += 1

                                # Переинициализация точек KLT внутри найденного контура
                                mask_img = np.zeros_like(gray)
                                cv2.fillPoly(mask_img, [np.int32(cand)], 255)
                                new_pts = cv2.goodFeaturesToTrack(
                                    gray, maxCorners=250, qualityLevel=0.01, minDistance=8, mask=mask_img
                                )
                                if new_pts is not None and len(new_pts) >= 10:
                                    self.p0 = new_pts
                                    inlier_points = [p[0] for p in new_pts]

        self.is_tracking = tracked
        if not tracked:
            self.p0 = None

        self.prev_gray = gray
        return tracked, self.current_corners if tracked else None, len(inlier_points), inlier_points, relocalized


def draw_tracking_overlay(frame: np.ndarray, corners, inlier_points=None, label: str = "Object"):
    """
    Наносит на кадр зеленую ограничивающую рамку, подпись объекта и ключевые точки.
    """
    vis = frame.copy()

    # Рисуем ключевые точки (зеленые маркеры)
    if inlier_points is not None:
        for pt in inlier_points:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1, cv2.LINE_AA)

    # Рисуем рамку объекта
    if corners is not None:
        pts = np.int32(corners).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Подпись берем в левом верхнем углу рамки
        top_left = corners[0][0]
        draw_styled_label(vis, label, (top_left[0], top_left[1] - 8))

    return vis
