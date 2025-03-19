import cv2 as cv
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
        blurredFrame = cv.medianBlur(grayFrame, 5)
        cannyFrame = cv.Canny(blurredFrame, 120, 150)
        contours, _ = cv.findContours(cannyFrame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 1000:
                approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
                cornerCount = len(approx)
                if cornerCount == 4:
                    x, y, w, h = cv.boundingRect(contour)
                    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv.putText(frame, 'Target', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv.drawContours(frame, contour, -1, (0, 255, 0), 3)
                #cv.imshow('gray', grayFrame)
                #cv.imshow('blurred', blurredFrame)
                #cv.imshow('canny', cannyFrame)
                cv.imshow('Contours', frame)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break
    except Exception as e:
        print(str(e))
video_stream.stop()
cv.destroyAllWindows()
