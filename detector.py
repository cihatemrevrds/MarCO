import cv2 as cv
import numpy as np

# Constants
MIN_CONTOUR_AREA = 1000
last_red_center = None
last_blue_center = None

# Kırmızı renk HSV aralıkları (iki ayrı bölge)
RED_LOWER_1 = np.array([0, 100, 50])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([160, 100, 50])
RED_UPPER_2 = np.array([180, 255, 255])

# Mavi renk HSV aralığı
BLUE_LOWER = np.array([100, 80, 50])
BLUE_UPPER = np.array([140, 255, 255])


def detect_colored_squares(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Renk maskeleri
    red_mask = cv.inRange(hsv, RED_LOWER_1, RED_UPPER_1) | cv.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    blue_mask = cv.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_OPEN, kernel)
    blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_CLOSE, kernel)

    # Maske üzerinden kontur alma
    red_contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv.findContours(blue_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Her renk için konturları sırayla işle
    process_contours(frame, red_contours, 'RedSquare', (0, 0, 255))
    process_contours(frame, blue_contours, 'BlueSquare', (255, 0, 0))

    return frame


def process_contours(frame, contours, label, color):
    global last_red_center, last_blue_center
    for contour in contours:
        area = cv.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        # Konturu sadeleştir
        approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)

        if len(approx) == 4 and cv.isContourConvex(approx):
            x, y, w, h = cv.boundingRect(approx)
            # Kare oranını kontrol et
            aspect_ratio = w / float(h)
            if 0.8 <= aspect_ratio <= 1.2:
                cx = x + w // 2
                cy = y + h // 2

                if label == "RedSquare":
                    last_red_center = (cx, cy)
                elif label == "BlueSquare":
                    last_blue_center = (cx, cy)

                draw_detected_square(frame, x, y, w, h, label, color)


def draw_detected_square(frame, x, y, w, h, label, color):
    cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_center_red_square():
    return last_red_center

def get_center_blue_square():
    return last_blue_center

