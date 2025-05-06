import cv2 as cv
import numpy as np

# Constants
MIN_CONTOUR_AREA = 1000
RED_LOWER_1 = np.array([0, 120, 70])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 70])
RED_UPPER_2 = np.array([180, 255, 255])
BLUE_LOWER = np.array([100, 150, 50])
BLUE_UPPER = np.array([140, 255, 255])

def detect_colored_squares(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray, 5)
    edges = cv.Canny(blurred, 90, 150)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv.contourArea(contour) < MIN_CONTOUR_AREA:
            continue

        approx = cv.approxPolyDP(contour, 0.05 * cv.arcLength(contour, True), True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv.boundingRect(contour)
        square_roi = frame[y:y+h, x:x+w]
        hsv_square = cv.cvtColor(square_roi, cv.COLOR_BGR2HSV)

        red_mask = cv.inRange(hsv_square, RED_LOWER_1, RED_UPPER_1) + \
                   cv.inRange(hsv_square, RED_LOWER_2, RED_UPPER_2)
        blue_mask = cv.inRange(hsv_square, BLUE_LOWER, BLUE_UPPER)

        if cv.countNonZero(red_mask) > 0:
            draw_detected_square(frame, x, y, w, h, 'RedSquare', (0, 0, 255))
        elif cv.countNonZero(blue_mask) > 0:
            draw_detected_square(frame, x, y, w, h, 'BlueSquare', (255, 0, 0))

    return frame

def draw_detected_square(frame, x, y, w, h, label, color):
    cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
