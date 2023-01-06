import glob
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw

fonts = glob.glob("fonts/*.ttf")


def printed_dataset():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for num in range(70000):
        x, y = random_digit()
        if num < 60000:
            train_x.append(x)
            train_y.append(y)
        else:
            test_x.append(x)
            test_y.append(y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return train_x, train_y, test_x, test_y


def random_digit():
    # generate random digit
    digit = random.randint(0, 9)
    data = generate_array_pil(digit)
    return data, digit


def generate_array_pil(digit: int):
    text = str(digit)
    size = 28
    # create new image
    img = Image.new("L", (size, size), (0,))
    draw = ImageDraw.Draw(img)
    # assign font
    font_name, font_size = random.choice(fonts), random.randint(21, 28)
    font = ImageFont.truetype(fr'C:\Users\Shreeya\PycharmProjects\sudoku2\{font_name}', font_size)
    # randomise position
    corner_x = size // 2 + random.randint(-2, 2)
    corner_y = size // 2 - random.randint(-1, 1)
    draw.text((corner_x, corner_y), text, (255,), font=font, anchor="mm")
    draw = np.array(img)
    return draw
