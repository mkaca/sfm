"""
Use the images inside the "images" folder to create a 3D point cloud

KC:
SIFT
BFMatcher
Lowe's ratio test
KNN
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from src.visualize import visualize_matches, visualize_points_on_images, drawEpipolarLinesOnImages
from src.constants import FEATURE_MATCHES_LIMIT, INTRINSIC_MATRICES
from src.vision_utils import (
    getNormalisationMat,
    conv2HomogeneousCoordinates,
    get_2d_points_of_matches,
    filter_points_with_homography,
    validate_epipolar_constraint,
    get_3d_points
)
from src.feature import get_features, get_matches

USE_HARDCODED_FEATURE_POINTS = False
MATCHER_TYPE = "flann"
DRAW_EPIPOLAR_LINES = True
VALIDATE_EPIPOLAR_CONSTRAINT = True


def main():
    images = sorted(os.listdir("images"))
    # ignore all images without .jpg in the name
    images = [img for img in images if img.endswith(".JPG")]

    # delete old visualization files
    if os.path.exists("matches_visualization.jpg"):
        os.remove("matches_visualization.jpg")
    if os.path.exists("pts_visualization1.jpg"):
        os.remove("pts_visualization1.jpg")
    if os.path.exists("pts_visualization2.jpg"):
        os.remove("pts_visualization2.jpg")
    if os.path.exists("pts_visualization_transformed_1.jpg"):
        os.remove("pts_visualization_transformed_1.jpg")
    if os.path.exists("pts_visualization_transformed_2.jpg"):
        os.remove("pts_visualization_transformed_2.jpg")

    if not USE_HARDCODED_FEATURE_POINTS:
        features_kp, features_des = get_features(images)
        matches = get_matches(features_des, matcher_type=MATCHER_TYPE)[0] # Note: get the first list of matches --> since only working with 2 images right now
        # Visualize matches between first two images
        visualize_matches(matches, features_kp, images)
        pts1, pts2 = get_2d_points_of_matches(matches, features_kp, feature_match_limit=FEATURE_MATCHES_LIMIT)
    else:
        pts1 = np.load("IMG_6954-14pt.npy")
        pts2 = np.load("IMG_6955-14pt.npy")

    # apply homography with RANSAC to remove bad matches (outliers) --> FROM IMAGE1 to IMAGE2.
    pts1, pts2 = filter_points_with_homography(pts1, pts2, images[1], visualize=True)
    visualize_points_on_images(pts1, pts2, images[0], images[1])

    if len(pts1) < 8:
        print("Not enough points to run the 8-point algorithm. Exiting. ")
        exit()

    K = np.asarray(INTRINSIC_MATRICES["intrinsics_k"])

    # convert to int32
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # get fundamental matrix via triangulation
    # pts1 and pts2 are the matched 2D points (Nx2 arrays)
    # RANSAC is essential here to remove outliers
    F_CV2, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC) #, ransacReprojThreshold=20.0, confidence=0.95)
    print("F_CV2", F_CV2)
    # mask tells us which points were inliers within the computation of the fundamental matrix
    # remove all points that are not in the mask
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    assert (pts1.shape == pts2.shape)
    print(f"pts1 shape after F computation with RANSAC= {pts1.shape}")
    # F_CV2, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC) # recompute F with only inliers

    pts1_homo, pts2_homo = conv2HomogeneousCoordinates(pts1, pts2)

    normalisationMat1 = getNormalisationMat(pts1_homo)
    normalisationMat2 = getNormalisationMat(pts2_homo)
    pts1_norm = np.float64([normalisationMat1 @ s_ptA for s_ptA in pts1_homo])
    pts2_norm = np.float64([normalisationMat2 @ s_ptB for s_ptB in pts2_homo])

    # get Essential Matrix --> assuming basic to derive Intrinsic Matrix
    # E_CV2, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, K, method=cv2.FM_RANSAC) # NOTE: This doesn't work very well
    # F_MANUAL = np.linalg.inv(K).T @ E_CV2 @ np.linalg.inv(K)
    E_from_F = K.T @ F_CV2 @ K # This is correct

    if DRAW_EPIPOLAR_LINES:
        drawEpipolarLinesOnImages(pts1, pts2, F_CV2, images[0], images[1])

    if VALIDATE_EPIPOLAR_CONSTRAINT:
        validate_epipolar_constraint(pts1_homo, pts2_homo, F_CV2)

    # recover pose by decomposing the essential matrix to find R and t
    retval, R, t, mask = cv2.recoverPose(E_from_F, pts1, pts2, cameraMatrix=np.asarray(INTRINSIC_MATRICES["intrinsics_k"]))
    print("RETVAL", retval)
    print("R", R)
    print("t", t)

    # Construct projection matrix P1 and P2 --> P = K * [R | t]
    P1 = np.asarray(INTRINSIC_MATRICES["intrinsics_k"]) @ np.hstack((np.eye(3), np.zeros((3, 1))))  # assumes that this is the global origin
    P2 = np.asarray(INTRINSIC_MATRICES["intrinsics_k"]) @ np.hstack((R, t))
    print(f"P1 = {P1} \n P2 = {P2}")
    print(f"pts1.T shape = {np.shape(pts1.T)} \n pts2.T shape = {np.shape(pts2.T)}")
    print(f"P1 shape = {np.shape(P1)} \n P2 shape = {np.shape(P2)}")
    pts_3d = get_3d_points(pts1, pts2, P1, P2)

    # use matplot lib to visualize the 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
           s=10,        # Increase marker size (the "splat")
           alpha=0.5,   # Add transparency
           c='b')       # Optional: color by depth or certainty
    plt.show()

    # todo: projection doesn't seem great --> I don't see a clear box / plane being identified in the 3D points
    # todo: add gaussian splatting to visualize the 3D points


if __name__ == "__main__":
    main()
