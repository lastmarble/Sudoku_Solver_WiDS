from keras.utils import img_to_array
from extract_puzzle import extract_digit
from extract_puzzle import find_puzzle
from digit_recog_cnn import load_model
import numpy as np
from numpy import asarray
import imutils
import cv2
import argparse


def predict(model, test_data_x):
    predictions = np.argmax(model.predict(test_data_x), axis=-1)
    return predictions


def predict_image(model, image):
    import cv2
    img = cv2.resize(image, (28, 28))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    number = predict(model, img)
    return number

# load model
model = load_model()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="digital_sudo.png", help="path to input image")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

# find the puzzle in the image
(puzzleImage, gray_puzzle) = find_puzzle(image, debug=args["debug"] > 0)

# initialize our 9x9 Sudoku board
board = np.zeros((9, 9), dtype="int")

# infer the location of each cell by dividing gray image into a 9x9 grid
stepX = gray_puzzle.shape[1] // 9
stepY = gray_puzzle.shape[0] // 9

# list to store the coordinates of each cell
cell_locations = []

# append locations
for y_cell in range(0, 9):
    # cells by X for each Y
    row = []
    for x_cell in range(0, 9):
        # cell end points
        first_x = x_cell * stepX
        last_x = (x_cell + 1) * stepX
        first_y = y_cell * stepY
        last_y = (y_cell + 1) * stepY

        # save x and y
        row.append((first_x, first_y, last_x, last_y))

        # crop to get digit from cell
        cell = gray_puzzle[first_y:last_y, first_x:last_x]
        digit = extract_digit(cell, debug=args["debug"] > 0)

        # non-empty digit
        if digit is not None:
            digit = asarray(digit)
            prediction = predict_image(model, digit)
            board[y_cell, x_cell] = prediction

    # save row
    cell_locations.append(row)

print(board)

