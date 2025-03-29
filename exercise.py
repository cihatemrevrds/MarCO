import cv2 as cv
import numpy as np
import threading
import time

# opens webcam and displays video feed
# press 'd' to close the window

class VideoStream:
    def __init__(self, src=0):
        self.capture = cv.VideoCapture(src, cv.CAP_DSHOW)
        if not self.capture.isOpened():
            raise Exception('Could not open video device')
        self.is_running = True
        self.frame = None
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.is_running:
            isTrue, frame = self.capture.read()
            if isTrue:
                self.frame = frame
            else:
                print('Frame is empty!')
                self.frame = None

    def read(self):
        return self.frame

    def stop(self):
        self.is_running = False
        self.thread.join()
        self.capture.release()

video_stream = VideoStream(0)
time.sleep(2)
while True:
    frame = video_stream.read()
    try:
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #blurredFrame = cv.GaussianBlur(grayFrame, (15, 15), 0)
        blurredFrame = cv.medianBlur(grayFrame, 5)
        cannyFrame = cv.Canny(blurredFrame, 90, 150)
        contours, _ = cv.findContours(cannyFrame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 1000:
                approx = cv.approxPolyDP(contour, 0.05*cv.arcLength(contour, True), True)
                cornerCount = len(approx)
                if cornerCount == 4:
                    x, y, w, h = cv.boundingRect(contour)
                    square = frame[y:y+h, x:x+w]
                    hsvSquare = cv.cvtColor(square, cv.COLOR_BGR2HSV)

                    lower_red1 = np.array([0, 120, 70])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 120, 70])
                    upper_red2 = np.array([180, 255, 255])

                    lower_blue = np.array([100, 150, 50])
                    upper_blue = np.array([140, 255, 255])

                    maskRed1 = cv.inRange(hsvSquare, lower_red1, upper_red1)
                    maskRed2 = cv.inRange(hsvSquare, lower_red2, upper_red2)
                    maskBlue = cv.inRange(hsvSquare, lower_blue, upper_blue)

                    if cv.countNonZero(maskRed1 + maskRed2) > 0 or cv.countNonZero(maskRed2) > 0:
                        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv.putText(frame, 'RedSquare', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    elif cv.countNonZero(maskBlue) > 0:
                        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv.putText(frame, 'BlueSquare', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    #cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    #cv.putText(frame, 'Square', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            #cv.drawContours(frame, contour, -1, (0, 255, 0), 3)
            #cv.imshow('gray', grayFrame)
            #cv.imshow('blurred', blurredFrame)
            #cv.imshow('canny', cannyFrame)
            cv.imshow('Monitor', frame)

        if cv.waitKey(1) & 0xFF == ord('d'):
            break
    except Exception as e:
        print(str(e))
    time.sleep(0.02)
video_stream.stop()
cv.destroyAllWindows()
