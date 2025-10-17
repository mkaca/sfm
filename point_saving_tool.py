import cv2

import numpy as np
import matplotlib.pyplot as plt
import os

ORIGINAL_DIMENSIONS = (5712, 4284, 3)
RESIZE_DIMENSIONS = (ORIGINAL_DIMENSIONS[1]//2, ORIGINAL_DIMENSIONS[0]//2)
SPACE_KEY = 32
LEFT_KEY = 2
UP_KEY = 0
RIGHT_KEY = 3
DOWN_KEY = 1
Q_KEY = 113
O_KEY = 111
W_KEY = 119
S_KEY = 115
A_KEY = 97
D_KEY = 100


def select_points(image):
    img = cv2.imread(os.path.join("images", image))
    img = cv2.resize(img, RESIZE_DIMENSIONS)
    img_h, img_w = img.shape[:2]

    points = []
    cx = img_w // 2
    cy = img_h // 2
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    while True:
        temp_img = img.copy()
        color = colors[len(points) % len(colors)]
        cv2.circle(temp_img, (cx, cy), 15, color, -1)
        cv2.imshow("image", temp_img)
        key_press = cv2.waitKey(0)
        if key_press == LEFT_KEY:
            cx -= 5
        elif key_press == RIGHT_KEY:
            cx += 5
        elif key_press == UP_KEY:
            cy -= 5
        elif key_press == DOWN_KEY:
            cy += 5
        elif key_press == W_KEY:
            cy -= 50
        elif key_press == S_KEY:
            cy += 50
        elif key_press == A_KEY:
            cx -= 50
        elif key_press == D_KEY:
            cx += 50
        elif key_press == Q_KEY:
            break
        elif key_press == O_KEY:
            print(f"Saving point, {cx}, {cy}")
            points.append((cx, cy))
        else:
            print(f"Invalid key press: {key_press}")
            continue

        cv2.destroyAllWindows()
    print(f"Points saved: {points}")
    cv2.destroyAllWindows()
    return points


def main():
    images = sorted(os.listdir("images"))
    # ignore all images without .jpg in the name
    images = [img for img in images if img.endswith(".JPG")]

    for image in images:
        points = select_points(image)
        file_name = image.replace(".JPG", ".npy")
        np.save(file_name, points)


if __name__ == "__main__":
    main()
