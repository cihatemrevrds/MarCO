import cv2 as cv
import numpy as np

def empty(x):
    pass

# Trackbar penceresi
cv.namedWindow("HSV Trackbars")
cv.resizeWindow("HSV Trackbars", 400, 300)

# Her kanal için min ve max değerler
cv.createTrackbar("H Min", "HSV Trackbars", 0, 179, empty)
cv.createTrackbar("H Max", "HSV Trackbars", 179, 179, empty)
cv.createTrackbar("S Min", "HSV Trackbars", 0, 255, empty)
cv.createTrackbar("S Max", "HSV Trackbars", 255, 255, empty)
cv.createTrackbar("V Min", "HSV Trackbars", 0, 255, empty)
cv.createTrackbar("V Max", "HSV Trackbars", 255, 255, empty)

# Kamera aç
cap = cv.VideoCapture(1, cv.CAP_DSHOW)

while True:
    success, frame = cap.read()
    if not success:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Trackbar'lardan değer oku
    h_min = cv.getTrackbarPos("H Min", "HSV Trackbars")
    h_max = cv.getTrackbarPos("H Max", "HSV Trackbars")
    s_min = cv.getTrackbarPos("S Min", "HSV Trackbars")
    s_max = cv.getTrackbarPos("S Max", "HSV Trackbars")
    v_min = cv.getTrackbarPos("V Min", "HSV Trackbars")
    v_max = cv.getTrackbarPos("V Max", "HSV Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(frame, frame, mask=mask)

    # Görüntüleri göster
    cv.imshow("Original", frame)
    cv.imshow("Mask", mask)
    cv.imshow("Filtered", result)

    key = cv.waitKey(1)
    if key == ord('d'):
        break

cap.release()
cv.destroyAllWindows()
