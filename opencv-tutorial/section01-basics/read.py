import cv2 as cv

# Read image
# img = cv.imread("../Resources/Photos/cat.jpg")
img = cv.imread("../Resources/Photos/cat_large.jpg")
cv.imshow("cat", img)

cv.waitKey(0)

# Read video
capture = cv.VideoCapture("../Resources/Videos/dog.mp4")

while True:
    isTrue, frame = capture.read()
    cv.imshow("capture", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()