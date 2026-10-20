import cv2        # Библиотека компьютерного зрения OpenCV (алгоритмы SIFT, ORB, KLT, гомография)
import numpy as np  # Библиотека для работы с многомерными числовыми массивами и матрицами


# Функция отрисовки красивой и читаемой текстовой метки с черной обводкой и белым текстом
def draw_styled_label(frame: np.ndarray, text: str, pt: tuple, font_scale: float = 0.8):
    # Координаты левого верхнего угла текста (x, y) приводим к целым числам (пикселям)
    x, y = int(pt[0]), int(pt[1])
    
    # Ограничиваем координаты, чтобы текст не вылезал за границы экрана (не улетал в минус и за край)
    x = max(10, min(x, frame.shape[1] - 150))
    y = max(25, min(y, frame.shape[0] - 10))

    # Выбираем стандартный четкий шрифт без засечек
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Шаг 1: Рисуем черную толстую обводку (толщина 4 пикселя, цвет (0, 0, 0) в BGR)
    # Это нужно, чтобы текст отлично читался на абсолютно любом фоне (и на белом, и на темном)
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    
    # Шаг 2: Поверх обводки рисуем белый текст (толщина 2 пикселя, цвет (255, 255, 255))
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)


# =========================================================================================
# КЛАСС 1: ДЕТЕКТОРНЫЙ ТРЕКЕР НА ОСНОВЕ СОПОСТАВЛЕНИЯ КЛЮЧЕВЫХ ТОЧЕК (SIFT ИЛИ ORB)
# На каждом кадре ищет объект с нуля: находит точки, сравнивает с шаблоном и строит гомографию
# =========================================================================================
class KeypointTracker:

    # Конструктор класса
    # method: какой алгоритм использовать — "sift" или "orb"
    # label: название объекта на рамке
    # min_inliers: минимальное количество подтвержденных точек-инлаеров для признания объекта найденным
    def __init__(self, method: str = "sift", label: str = "Object", min_inliers: int = 8):
        # Приводим название метода к нижнему регистру для надежности
        self.method = method.lower()
        # Сохраняем текстовую метку
        self.label = label
        # Минимальный порог точек: для гомографии нужно минимум 4 точки, 8 берем с запасом для надежности
        self.min_inliers = min_inliers

        # Настраиваем детекторы и сопоставители (матчеры) в зависимости от выбранного метода
        if self.method == "sift":
            # Создаём детектор SIFT (Scale-Invariant Feature Transform)
            self.detector = cv2.SIFT_create()
            # Для SIFT дескрипторы состоят из чисел с плавающей точкой, поэтому расстояние считаем по Евклиду (L2)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
            # Порог теста Лоу (отношение расстояния 1-го соседа ко 2-му должно быть меньше 0.75)
            self.ratio_thresh = 0.75
        else:
            # Создаём детектор ORB (Oriented FAST and Rotated BRIEF), ограничиваем до 1500 лучших точек
            self.detector = cv2.ORB_create(nfeatures=1500)
            # Дескрипторы ORB бинарные (0 и 1), поэтому расстояние считаем по Хэммингу (побитовый XOR)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            # Порог теста Лоу для бинарных дескрипторов обычно берут чуть мягче — 0.8
            self.ratio_thresh = 0.8

        # Переменные для хранения информации о шаблоне (первом кадре с объектом)
        self.template_kp = None       # Список ключевых точек шаблона
        self.template_des = None      # Массив дескрипторов («паспортов») точек шаблона
        self.template_corners = None  # 4 исходных угла рамки вокруг шаблона
        self.h0, self.w0 = 0, 0       # Высота и ширина первого кадра

    # Метод установки шаблона (запоминаем первый кадр или заданный контур объекта)
    def set_template(self, frame: np.ndarray, corners: np.ndarray = None):
        # Если кадр цветной (3 канала BGR) — переводим его в градации серого (Gray), алгоритмам цвет не нужен
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        # Сохраняем размеры первого кадра
        self.h0, self.w0 = gray.shape[:2]

        # Если переданы точные координаты углов объекта внутри кадра (например, полигон)
        if corners is not None:
            # Преобразуем координаты к формату массива float32 вида (N, 1, 2)
            self.template_corners = np.float32(corners).reshape(-1, 1, 2)
            # Создаем черную маску того же размера, что и кадр
            mask = np.zeros_like(gray)
            # Закрашиваем белым (255) область внутри переданных углов объекта
            cv2.fillPoly(mask, [np.int32(self.template_corners)], 255)
            # Находим точки и вычисляем дескрипторы строго внутри этой маски (игнорируем фон вокруг!)
            self.template_kp, self.template_des = self.detector.detectAndCompute(gray, mask)
        else:
            # Если объект занимает весь первый кадр — ищем точки по всему кадру без маски
            self.template_kp, self.template_des = self.detector.detectAndCompute(gray, None)
            # 4 угла объекта — это 4 угла всего кадра: (0,0), (w,0), (w,h), (0,h)
            self.template_corners = np.float32(
                [[0, 0], [self.w0 - 1, 0], [self.w0 - 1, self.h0 - 1], [0, self.h0 - 1]]
            ).reshape(-1, 1, 2)

    # Метод обработки очередного кадра видео
    def process_frame(self, frame: np.ndarray):
        # Переводим текущий кадр в градации серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        # Ищем ключевые точки (kp) и вычисляем дескрипторы (des) на текущем кадре
        kp, des = self.detector.detectAndCompute(gray, None)

        # Переменные результата по умолчанию (объект пока не найден)
        tracked = False            # Флаг: успешно ли найден объект
        projected_corners = None   # Координаты 4 спроецированных углов рамки на кадре
        inliers_count = 0          # Число подтвержденных точек-инлаеров
        inlier_points = []         # Список координат этих точек для визуализации

        # Проверяем: найдены ли дескрипторы на кадре, есть ли шаблон и найдено ли хотя бы 4 точки
        if des is not None and self.template_des is not None and len(kp) >= 4:
            # Ищем 2 лучших соответствия (k=2) для каждой точки шаблона среди точек текущего кадра
            # k=2 нужно именно для применения фильтрации по тесту Лоу
            matches = self.matcher.knnMatch(self.template_des, des, k=2)
            
            # Фильтрация по критерию Дэвида Лоу (Lowe's ratio test):
            # m — самый близкий кандидат (расстояние d1), n — второй по близости (расстояние d2)
            # Если d1 < 0.75 * d2 — совпадение уникальное и надежное, оставляем его!
            good = [
                m
                for match in matches
                if len(match) == 2
                for m, n in [match]
                if m.distance < self.ratio_thresh * n.distance
            ]

            # Если после фильтрации осталось достаточно хороших пар (больше min_inliers)
            if len(good) >= self.min_inliers:
                # Извлекаем координаты точек на шаблоне (src_pts)
                src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                # Извлекаем координаты соответствующих точек на текущем кадре (dst_pts)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # Вычисляем матрицу гомографии H (3x3) с помощью алгоритма RANSAC
                # Порог ошибки репроекции = 5.0 пикселей (точки с ошибкой <= 5px признаются инлаерами)
                # mask содержит 1 для хороших точек (инлаеров) и 0 для выбросов (outliers)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                # Если матрица гомографии успешно сошлась и маска существует
                if H is not None and mask is not None:
                    # Считаем, сколько точек реально поддержали эту гомографию (сумма единиц в маске)
                    inliers_count = int(np.sum(mask))
                    
                    # Если число подтвердивших точек превышает наш порог надежности
                    if inliers_count >= self.min_inliers:
                        # Умножаем исходные 4 угла шаблона на матрицу гомографии H (перспективная проекция)
                        # Получаем точные координаты 4 углов объекта на текущем кадре
                        proj = cv2.perspectiveTransform(self.template_corners, H)
                        # Вычисляем площадь полученного четырехугольника
                        area = cv2.contourArea(np.int32(proj))
                        # Площадь исходного шаблона
                        tmpl_area = cv2.contourArea(np.int32(self.template_corners))
                        
                        # Геометрическая валидация (защита от ложных "вывернутых" рамок):
                        # 1. cv2.isContourConvex: контур должен быть выпуклым (не "бантик" и не самопересечение)
                        # 2. Площадь должна быть адекватной (от 15% до 400% от размера шаблона)
                        if cv2.isContourConvex(np.int32(proj)) and 0.15 * tmpl_area < area < 4.0 * tmpl_area:
                            projected_corners = proj  # Сохраняем найденную рамку
                            tracked = True            # Ставим флаг: объект успешно найден!
                            mask_flat = mask.ravel()  # Разворачиваем маску в одномерный массив
                            # Сохраняем координаты только подтверждённых точек-инлаеров (где mask == 1)
                            inlier_points = [dst_pts[i][0] for i in range(len(mask_flat)) if mask_flat[i] == 1]

        # Возвращаем результаты наружу
        return tracked, projected_corners, inliers_count, inlier_points


# =========================================================================================
# КЛАСС 2: ГИБРИДНЫЙ ТРЕКЕР: KLT (ОПТИЧЕСКИЙ ПОТОК) + SIFT-РЕЛОКАЛИЗАТОР
# Сочетает сверхвысокую скорость KLT (>400 FPS) и надежность SIFT при срыве трекинга
# =========================================================================================
class HybridTracker:

    # Конструктор гибридного трекера
    def __init__(self, label: str = "Object", enable_relocalization: bool = True):
        self.label = label
        # Включена ли автоматическая релокализация через SIFT при потере точек KLT
        self.enable_relocalization = enable_relocalization
        # Инициализируем детекторы SIFT и матчер L2 для модуля релокализации
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

        # Переменные эталонного шаблона
        self.template_gray = None     # Черно-белый кадр шаблона
        self.template_kp = None       # SIFT-точки шаблона
        self.template_des = None      # SIFT-дескрипторы шаблона
        self.template_corners = None  # Исходные 4 угла рамки шаблона
        self.w0, self.h0 = 0, 0       # Размеры шаблона

        # Состояние отслеживания в реальном времени
        self.prev_gray = None         # Предыдущий кадр в градациях серого (для оптического потока KLT)
        self.p0 = None                # Массив текущих координат отслеживаемых точек KLT
        self.current_corners = None   # Текущие 4 угла рамки на последнем кадре
        self.is_tracking = False      # Флаг: ведётся ли сейчас непрерывный трекинг
        self.reloc_counter = 0        # Счётчик, сколько раз сработала релокализация

    # Инициализация первого кадра (шаблона)
    def set_template(self, frame: np.ndarray, corners: np.ndarray = None):
        # Переводим первый кадр в ч/б
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        self.template_gray = gray
        self.h0, self.w0 = gray.shape[:2]

        if corners is not None:
            # Запоминаем переданные координаты углов объекта
            self.template_corners = np.float32(corners).reshape(-1, 1, 2)
            # Создаем полигональную маску объекта
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [np.int32(self.template_corners)], 255)
            # Вычисляем SIFT-дескрипторы шаблона внутри контура для будущей релокализации
            self.template_kp, self.template_des = self.sift.detectAndCompute(gray, mask)
            self.current_corners = self.template_corners.copy()
            # Находим характерные углы для оптического потока KLT через cv2.goodFeaturesToTrack (детектор Ши-Томаси)
            # maxCorners=300: до 300 точек, qualityLevel=0.01: порог качества, minDistance=8: расстояние между точками
            self.p0 = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=8, mask=mask)
        else:
            # Если углы не переданы — берем весь кадр
            self.template_kp, self.template_des = self.sift.detectAndCompute(gray, None)
            self.template_corners = np.float32(
                [[0, 0], [self.w0 - 1, 0], [self.w0 - 1, self.h0 - 1], [0, self.h0 - 1]]
            ).reshape(-1, 1, 2)
            self.current_corners = self.template_corners.copy()
            self.p0 = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=8)

        # Сохраняем текущий кадр как предыдущий для следующего шага оптического потока
        self.prev_gray = gray
        self.is_tracking = True

    # Главный метод обработки кадра гибридным алгоритмом
    def process_frame(self, frame: np.ndarray):
        # Переводим кадр в ч/б
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        tracked = False            # Объект найден или нет
        relocalized = False        # Сработала ли релокализация
        inlier_points = []         # Точки для отрисовки

        # =====================================================================
        # ЭТАП 1: БЫСТРЫЙ ПОКАДРОВЫЙ ТРЕКИНГ ЧЕРЕЗ ОПТИЧЕСКИЙ ПОТОК KLT (Lucas-Kanade)
        # =====================================================================
        # Если мы находимся в режиме трекинга и точек KLT достаточно много (>= 10)
        if self.is_tracking and self.p0 is not None and len(self.p0) >= 10:
            # Вычисляем пирамидальный оптический поток Лукаса-Канаде между prev_gray и текущим gray
            # p1 — новые координаты точек на текущем кадре
            # status — флаг для каждой точки (1: точка успешно отслежена, 0: потеряна)
            p1, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.p0,
                None,
                winSize=(25, 25),   # Размер окна поиска окрестности 25x25 пикселей
                maxLevel=3,         # 3 уровня пирамиды изображений (позволяет отслеживать быстрые смещения)
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            
            # Отбираем только те точки, у которых status == 1 (успешно найденные)
            good_new = p1[status == 1]
            good_old = self.p0[status == 1]

            # Если успешно отследилось не менее 10 точек
            if len(good_new) >= 10:
                # Оцениваем матрицу перехода H между положением точек на прошлом кадре и на текущем
                H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 3.0)
                if H is not None:
                    # Применяем преобразование H к текущей рамке
                    cand = cv2.perspectiveTransform(self.current_corners, H)
                    area = cv2.contourArea(np.int32(cand))
                    tmpl_area = cv2.contourArea(np.int32(self.template_corners))
                    
                    # Проверяем геометрию: рамка должна быть выпуклой и адекватного размера
                    if cv2.isContourConvex(np.int32(cand)) and 0.15 * tmpl_area < area < 4.0 * tmpl_area:
                        self.current_corners = cand  # Обновляем координаты рамки
                        tracked = True               # Трекинг успешен!
                        
                        # Обновляем список отслеживаемых точек (оставляем только инлаеры RANSAC)
                        if mask is not None:
                            inliers = good_new[mask.ravel() == 1]
                            self.p0 = inliers.reshape(-1, 1, 2)
                            inlier_points = inliers
                        else:
                            self.p0 = good_new.reshape(-1, 1, 2)
                            inlier_points = good_new

        # =====================================================================
        # ЭТАП 2: АВТОМАТИЧЕСКАЯ РЕЛОКАЛИЗАЦИЯ ЧЕРЕЗ SIFT ПРИ СРЫВЕ ТРЕКИНГА
        # Срабатывает, если KLT потерял объект или точек стало критически мало (< 15)
        # =====================================================================
        if self.enable_relocalization and (not tracked or (self.p0 is not None and len(self.p0) < 15)):
            # Запускаем глобальный поиск: детектируем SIFT-точки на текущем кадре
            kp_curr, des_curr = self.sift.detectAndCompute(gray, None)
            
            # Если точки найдены
            if des_curr is not None and len(kp_curr) >= 4:
                # Сопоставляем дескрипторы шаблона с дескрипторами текущего кадра
                matches = self.bf.knnMatch(self.template_des, des_curr, k=2)
                # Фильтруем по критерию Лоу (порог 0.75)
                good = [
                    m
                    for match in matches
                    if len(match) == 2
                    for m, n in [match]
                    if m.distance < 0.75 * n.distance
                ]

                # Если нашлось хотя бы 8 хороших совпадений с исходным шаблоном
                if len(good) >= 8:
                    src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    
                    # Ищем гомографию между шаблоном и текущим кадром через RANSAC
                    H_reloc, mask_reloc = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if H_reloc is not None and mask_reloc is not None:
                        inliers_cnt = int(np.sum(mask_reloc))
                        # Если не менее 8 точек подтвердили совпадение
                        if inliers_cnt >= 8:
                            # Проецируем углы исходного шаблона
                            cand = cv2.perspectiveTransform(self.template_corners, H_reloc)
                            area = cv2.contourArea(np.int32(cand))
                            tmpl_area = cv2.contourArea(np.int32(self.template_corners))
                            
                            # Валидация выпуклости и площади
                            if cv2.isContourConvex(np.int32(cand)) and 0.15 * tmpl_area < area < 4.0 * tmpl_area:
                                self.current_corners = cand  # Восстанавливаем рамку!
                                tracked = True               # Объект снова найден!
                                relocalized = True          # Отмечаем, что сработала релокализация
                                self.reloc_counter += 1      # Увеличиваем счётчик восстановлений

                                # ВОССТАНОВЛЕНИЕ ТОЧЕК ДЛЯ KLT:
                                # Создаём маску найденного контура объекта
                                mask_img = np.zeros_like(gray)
                                cv2.fillPoly(mask_img, [np.int32(cand)], 255)
                                # Заново находим свежие точки Ши-Томаси строго внутри найденной рамки
                                new_pts = cv2.goodFeaturesToTrack(
                                    gray, maxCorners=250, qualityLevel=0.01, minDistance=8, mask=mask_img
                                )
                                # Передаём новые точки KLT, чтобы на следующем кадре снова работал быстрый оптический поток
                                if new_pts is not None and len(new_pts) >= 10:
                                    self.p0 = new_pts
                                    inlier_points = [p[0] for p in new_pts]

        # Запоминаем статус: отслежен ли объект
        self.is_tracking = tracked
        # Если объект потерян — сбрасываем точки KLT
        if not tracked:
            self.p0 = None

        # Запоминаем текущий кадр как предыдущий для следующего шага
        self.prev_gray = gray
        # Возвращаем флаг успеха, координаты рамки, число точек, сами точки и флаг релокализации
        return tracked, self.current_corners if tracked else None, len(inlier_points), inlier_points, relocalized


# =========================================================================================
# ФУНКЦИЯ ОТРИСОВКИ РЕЗУЛЬТАТОВ ТРЕКИНГА НА КАДРЕ
# Рисует зеленый полигон вокруг объекта, маленькие кружки ключевых точек и текстовую плашку
# =========================================================================================
def draw_tracking_overlay(frame: np.ndarray, corners, inlier_points=None, label: str = "Object"):
    # Делаем копию кадра, чтобы не портить оригинал
    vis = frame.copy()

    # Рисуем зеленые точки-инлаеры (маленькие кружочки радиусом 2 пикселя)
    if inlier_points is not None:
        for pt in inlier_points:
            x, y = int(pt[0]), int(pt[1])
            # cv2.circle: координаты (x,y), радиус 2, цвет зеленый (0, 255, 0), толщина -1 (заливка)
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1, cv2.LINE_AA)

    # Рисуем зеленый четырехугольник рамки объекта
    if corners is not None:
        # Приводим 4 угла к формату массива целых чисел (пикселей)
        pts = np.int32(corners).reshape((-1, 1, 2))
        # cv2.polylines: рисует замкнутый многоугольник (isClosed=True), толщина 2, сглаживание LINE_AA
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Выбираем координаты левого верхнего угла рамки
        top_left = corners[0][0]
        # Рисуем стильную контрастную подпись чуть выше верхнего левого угла рамки
        draw_styled_label(vis, label, (top_left[0], top_left[1] - 8))

    # Возвращаем размеченный кадр
    return vis
