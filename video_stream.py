import cv2 as cv
import threading

class VideoStream:
    def __init__(self, src=0):
        self.capture = cv.VideoCapture(src, cv.CAP_DSHOW)
        if not self.capture.isOpened():
            raise Exception('Could not open video device')
        
        self.is_running = True
        self.frame = None
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def _update_frame(self):
        while self.is_running:
            success, frame = self.capture.read()
            if success:
                self.frame = frame
            else:
                print('[WARNING] Empty frame received.')
                self.frame = None

    def read(self):
        return self.frame

    def stop(self):
        self.is_running = False
        self.thread.join()
        self.capture.release()
