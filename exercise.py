import cv2 as cv

# read image
""" img = cv.imread('111.jpg')
cv.imshow('Jax', img)
cv.waitKey(0) """


# opens webcam and displays video feed
# press 'd' to close the window

capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    if not isTrue:
        print('Error reading video')
        break
    else:
        try:
            grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            blurredFrame = cv.GaussianBlur(grayFrame, (15, 15), 1.5)
            cannyFrame = cv.Canny(blurredFrame, 50, 150)
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
                    #cv.drawContours(frame, contour, -1, (0, 255, 0), 3)
                    cv.imshow('gray', grayFrame)
                    cv.imshow('blurred', blurredFrame)
                    cv.imshow('canny', cannyFrame)
                    cv.imshow('Contours', frame)
            if cv.waitKey(20) & 0xFF == ord('d'):
                break
        except:
            print('Error converting frame')
capture.release()
cv.destroyAllWindows()
