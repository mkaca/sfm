import cv2
import os
import numpy as np
from src.constants import RESIZE_DIMENSIONS, FEATURE_MATCHES_LIMIT
from src.vision_utils import getCorrespondencesEpilines


def visualize_matches(matches, features_kp, images):
    if len(matches) > 0 and len(features_kp) >= 2:
        img1 = cv2.imread(os.path.join("images", images[0]))
        img1 = cv2.resize(img1, RESIZE_DIMENSIONS)
        img2 = cv2.imread(os.path.join("images", images[1]))
        img2 = cv2.resize(img2, RESIZE_DIMENSIONS)


        # Sort matches by distance (lower distance = better match)
        sorted_matches = sorted(matches[0], key=lambda x: x.distance)
        best_matches = sorted_matches[:FEATURE_MATCHES_LIMIT]

        print(f"Showing best {len(best_matches)} matches out of {len(matches[0])} total matches")

        # Draw matches with custom thickness
        match_img = cv2.drawMatches(
            img1, features_kp[0],
            img2, features_kp[1],
            best_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0),  # Green color for matches
            singlePointColor=(255, 0, 0),  # Red color for keypoints
            matchesThickness=5  # Make lines thicker (default is 1)
        )

        # Save the result
        cv2.imwrite("matches_visualization.jpg", match_img)
        print("Saved matches visualization as 'matches_visualization.jpg'")


def visualize_points_on_images(pts1, pts2, image1, image2, save_path_prefix="pts_visualization"):
    # visualize the points
    img1 = cv2.imread(os.path.join("images", image1))
    # print(np.shape(img1))
    img1 = cv2.resize(img1, RESIZE_DIMENSIONS)
    # print(np.shape(img1))
    # exit()
    img2 = cv2.imread(os.path.join("images", image2))
    img2 = cv2.resize(img2, RESIZE_DIMENSIONS)
    size_of_points = len(pts1)
    decrement_step = 127 // size_of_points
    for i, point in enumerate(pts1):
        cv2.circle(img1, (int(point[0]), int(point[1])), 15, (decrement_step*i, 0, 255-decrement_step*i), -1)
    print(f"len(pts1) = {len(pts1)}")
    print(f"len(pts2) = {len(pts2)}")
    for i, point in enumerate(pts2):
        cv2.circle(img2, (int(point[0]), int(point[1])), 15, (decrement_step*i, 0, 255-decrement_step*i), -1)
    cv2.imwrite(f"{save_path_prefix}1.jpg", img1)
    cv2.imwrite(f"{save_path_prefix}2.jpg", img2)
    print("Saved points visualization as 'pts_visualization1.jpg' and 'pts_visualization2.jpg'")


def drawEpilines(imgA, imgB, lines, ptsA_homo, ptsB_homo):
    """Draw epipolar lines on img1 and key points on img1 and img2

    Parameters
    ----------
    imgA : int numpy.ndarray, shape (height, width, channel)
        An array of image.
    imgB : int numpy.ndarray, shape (n_correspondences, 2)
        An array of image.
    lines : float, numpy.ndarray, shape (num_points, 3)
        A n array of the epipolar lines.  Each epipolar line is represented as an array of three float number [a, b, c].
        [a, b, c] are the coefficients of a line ax + by + c = 0
    ptsA_homo : int numpy.ndarray, shape (n_correspondences, 3)
        An array of coordinates of correspondences from the image A, in the form of homogeneous coordinate.
    ptsB_homo : int numpy.ndarray, shape (n_correspondences, 3)
        An array of coordinates of correspondences from the image B, in the form of homogeneous coordinate.
    -------
    Return
    annotate_imgA : int numpy.ndarray, shape (height, width, channel)
        An array of image, with epipolar lines and key points drawn on it
    annotate_imgB : int numpy.ndarray, shape (height, width, channel)
        An array of image, with key points drawn on it
    """

    # Convert image to gray color in BGR representation
    annotate_imgA = cv2.cvtColor(cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    annotate_imgB = cv2.cvtColor(cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    row, col, cha = annotate_imgA.shape
    row -= 1  # index are in range of [0, height of image)
    col -= 1  # index are in range of [0, width of image)

    for r, ptA, ptB in zip(lines, ptsA_homo, ptsB_homo):
        # Generate color randomly
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Choosing valid points that lie on image boundaries
        a, b, c = r
        p0 = tuple(map(round, [0, -c / b]))
        p1 = tuple(map(round, [col, -(c + (a * col)) / b]))
        p2 = tuple(map(round, [-c / a, 0]))
        p3 = tuple(map(round, [-(c + (b * row)) / a, row]))
        p = [(x, y) for (x, y) in [p0, p1, p2, p3] if 0 <= x <= col and 0 <= y <= row]

        print(ptA, ptB)
        if len(p) >= 2:
            annotate_imgA = cv2.line(annotate_imgA, p[0], p[1], color, 2)
        annotate_imgA = cv2.circle(annotate_imgA, ptA[:2].astype(int), 6, color, -1)
        annotate_imgB = cv2.circle(annotate_imgB, ptB[:2].astype(int), 6, color, -1)

    return annotate_imgA, annotate_imgB


def drawEpipolarLinesOnImages(pts1_homo, pts2_homo, F_MANUAL, image1, image2):
    # draw the epipolar lines on the images
    img1 = cv2.imread(os.path.join("images", image1))
    img1 = cv2.resize(img1, RESIZE_DIMENSIONS)
    img2 = cv2.imread(os.path.join("images", image2))
    img2 = cv2.resize(img2, RESIZE_DIMENSIONS)
    linesA, linesB = getCorrespondencesEpilines(pts1_homo, pts2_homo, F_MANUAL)
    a, b = drawEpilines(img1, img2, linesA, pts1_homo, pts2_homo)
    c, d = drawEpilines(img2, img1, linesB, pts2_homo, pts1_homo)
    cv2.imshow("img1", a)
    cv2.imshow("img2", b)
    cv2.imshow("img3", c)
    cv2.imshow("img4", d)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
