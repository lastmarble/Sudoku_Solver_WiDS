import argparse
import imutils
import cv2
pentagon = 0
quadrilateral = 0
triangle = 0
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="Hands_on_1.jpg", help="path to input image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=600)
image = resized
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 3)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Thresh", thresh)
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=6)
cv2.imshow("Dilated", mask)
mask = cv2.erode(mask, None, iterations=6)
cv2.imshow("Eroded", mask)
counts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
counts = imutils.grab_contours(counts)
output = mask.copy()
eps = 0.05
AREA = mask.shape[0]*mask.shape[1]/500
for c in counts:
    if cv2.contourArea(c) > AREA:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        output = image.copy()
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
        num_pts = len(approx)
        text = f"eps={eps}, num_pts={num_pts}"
        cv2.putText(output, text, (x - 80, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Approximated Contour", output)
        if num_pts == 3:
            triangle += 1
        elif num_pts == 4:
            quadrilateral += 1
        elif num_pts == 5:
            pentagon += 1
        cv2.waitKey(0)
print(f" triangles = {triangle}, quadrilaterals = {quadrilateral}, pentagons = {pentagon}")
cv2.waitKey(0)
cv2.destroyAllWindows()
