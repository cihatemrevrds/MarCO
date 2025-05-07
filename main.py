import cv2 as cv
import time
from video_stream import VideoStream
from detector import detect_colored_squares

def main():
    video_stream = VideoStream(1)
    time.sleep(2)

    try:
        while True:
            frame = video_stream.read()
            if frame is None:
                continue

            processed_frame = detect_colored_squares(frame)
            cv.imshow('Monitor', processed_frame)

            if cv.waitKey(1) & 0xFF == ord('d'):
                break

            time.sleep(0.02)

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        video_stream.stop()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()
