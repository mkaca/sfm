import cv2
import os
import numpy as np
from src.constants import RESIZE_DIMENSIONS, FEATURE_MATCHES_LIMIT


def visualize_matches(matches, features_kp, images, best_n_matches=FEATURE_MATCHES_LIMIT):
    if len(features_kp) >= 2:
        img1 = cv2.imread(os.path.join("images", images[0]))
        img1 = cv2.resize(img1, RESIZE_DIMENSIONS)
        img2 = cv2.imread(os.path.join("images", images[1]))
        img2 = cv2.resize(img2, RESIZE_DIMENSIONS)


        # Sort matches by distance (lower distance = better match)
        sorted_matches = sorted(matches, key=lambda x: x.distance)
        best_matches = sorted_matches[:best_n_matches]

        print(f"Showing best {len(best_matches)} matches out of {len(matches)} total matches")

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
    img1 = cv2.resize(img1, RESIZE_DIMENSIONS)
    img2 = cv2.imread(os.path.join("images", image2))
    img2 = cv2.resize(img2, RESIZE_DIMENSIONS)
    size_of_points = len(pts1)
    decrement_step = 127.0 / size_of_points
    for i, point in enumerate(pts1):
        cv2.circle(img1, (int(point[0]), int(point[1])), 15, (int(decrement_step*i), 0, 255-int(decrement_step*i)), -1)
    # print(f"len(pts1) = {len(pts1)}")
    # print(f"len(pts2) = {len(pts2)}")
    for i, point in enumerate(pts2):
        cv2.circle(img2, (int(point[0]), int(point[1])), 15, (int(decrement_step*i), 0, 255-int(decrement_step*i)), -1)
    cv2.imwrite(f"{save_path_prefix}1.jpg", img1)
    cv2.imwrite(f"{save_path_prefix}2.jpg", img2)
    print(f"Saved points visualization as '{save_path_prefix}1.jpg' and '{save_path_prefix}2.jpg'")


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
    for r,pt1,pt2 in zip(lines,pts1.astype(int),pts2.astype(int)):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def drawEpipolarLinesOnImages(pts1_2d, pts2_2d, F, image1, image2):
    # draw the epipolar lines on the images
    img1 = cv2.imread(os.path.join("images", image1))
    img1 = cv2.resize(img1, RESIZE_DIMENSIONS)
    img2 = cv2.imread(os.path.join("images", image2))
    img2 = cv2.resize(img2, RESIZE_DIMENSIONS)
    linesA = cv2.computeCorrespondEpilines(pts2_2d.reshape(-1,1,2), 2, F)
    linesA = linesA.reshape(-1,3)
    linesB = cv2.computeCorrespondEpilines(pts1_2d.reshape(-1,1,2), 1, F)
    linesB = linesB.reshape(-1,3)
    img1_epilines, img1 = drawlines(img1, img2, linesA, pts1_2d, pts2_2d)
    img2_epilines, img2 = drawlines(img2, img1, linesB, pts2_2d, pts1_2d)
    # concat the images horizontally
    img_concat = np.concatenate((img1_epilines, img2_epilines), axis=1)
    cv2.imshow("img_epilines", img_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
