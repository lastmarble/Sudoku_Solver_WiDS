from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


def find_puzzle(image, debug=False):
    # convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    # adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)
    # find contours and sort them by size in descending order
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    puzzleCnt = None
    # loop over the contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzleCnt = approx
            break
    # error
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))
    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    gray_puzzle = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    if debug:
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
    return puzzle, gray_puzzle


def extract_digit(cell, debug=False):
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    # check to see if we are visualizing the cell thresholding step
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)
    # finding the number if it's there
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # empty cell
    if len(contours) == 0:
        return None
    # non-empty cell mask largest contour
    largest_cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
    (h, w) = thresh.shape
    # mask percent of total area
    percent_fill = cv2.countNonZero(mask) / float(h*w)
    if percent_fill < 0.03:
        return None
    # apply the mask to the cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # visualize the digit
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    # return the digit to the calling function
    return digit
