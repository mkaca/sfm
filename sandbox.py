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
from src.constants import RESIZE_DIMENSIONS, FEATURE_MATCHES_LIMIT, INTRINSIC_MATRICES
from src.vision_utils import getNormalisationMat, conv2HomogeneousCoordinates, get_2d_points_of_matches
from src.feature import get_features, get_matches

USE_HARDCODED_FEATURE_POINTS = False
MATCHER_TYPE = "flann"


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
        matches = get_matches(features_des, matcher_type=MATCHER_TYPE)
        # Visualize matches between first two images
        visualize_matches(matches, features_kp, images)
        pts1, pts2 = get_2d_points_of_matches(matches[0], features_kp, feature_match_limit=FEATURE_MATCHES_LIMIT)
    else:
        pts1 = np.load("IMG_6954-14pt.npy")
        pts2 = np.load("IMG_6955-14pt.npy")

    # print(pts1)
    # print(pts2)

    # apply homography with RANSAC to remove bad matches (outliers) --> FROM IMAGE1 to IMAGE2.
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=2200)
    # print(H)
    # print(mask)
    # remove all points that are not in the mask
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    assert (pts1.shape == pts2.shape)

    # use homography to transform pts1 to pts2 space
    # print(pts2[np.newaxis].shape)
    # Note: this part is just for fun
    pts1_for_transform = pts1[np.newaxis].astype(np.float32)
    pts1_transformed = cv2.perspectiveTransform(pts1_for_transform, H)
    pts1_transformed_corrected = pts1_transformed.squeeze()
    print(pts2.shape)
    print(pts1_transformed_corrected.shape)
    visualize_points_on_images(pts1, pts2, images[0], images[1])
    visualize_points_on_images(pts1_transformed_corrected, pts2, images[1], images[1], save_path_prefix="pts_visualization_transformed_")
    # exit()

    if len(pts1) < 8:
        print("Not enough points to run the 8-point algorithm. Exiting. ")
        exit()

    K = np.asarray(INTRINSIC_MATRICES["intrinsics_k"])

    # get fundamental matrix via triangulation
    # pts1 and pts2 are the matched 2D points (Nx2 arrays)
    # RANSAC is essential here to remove outliers
    F_CV2, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=20.0, confidence=0.95)
    print("F_CV2", F_CV2)
    # remove all points that are not in the mask
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    assert (pts1.shape == pts2.shape)
    print(f"pts1 shape after F computation with RANSAC= {pts1.shape}")

    # get Essential Matrix --> assuming basic to derive Intrinsic Matrix
    # E, mask = cv2.findEssentialMat(pts1, pts2, INTRINSIC_MATRICES["intrinsics_k"], INTRINSIC_MATRICES["distortion_coefficients"], method=cv2.RANSAC)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.FM_RANSAC) # NOTE: I don't think this is correct
    # print(mask)
    # print(E)
    # exit()
    E_from_F = K.T @ F_CV2 @ K # This looks better

    F_MANUAL = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    print("F MANUAL: ", F_MANUAL)

    pts1_homo, pts2_homo = conv2HomogeneousCoordinates(pts1, pts2)

    # drawEpipolarLinesOnImages(pts1_homo, pts2_homo, F_CV2, images[0], images[1])
    # exit()

    normalisationMat1 = getNormalisationMat(pts1_homo)
    normalisationMat2 = getNormalisationMat(pts2_homo)
    pts1_norm = np.float64([normalisationMat1 @ s_ptA for s_ptA in pts1_homo])
    pts2_norm = np.float64([normalisationMat2 @ s_ptB for s_ptB in pts2_homo])
    # Validate epipolar constraint: x2^T * F * x1 = 0
    # VALIDATION - confirm the E matrix actually works ... by calculating the residual
    # todo: perhaps this can have errors due to scaling factor issues
    print("\n=== EPIPOLAR CONSTRAINT VALIDATION ===")
    avg_res = 0
    for i in range(len(pts1)):  # Check first 5 points
        pt1 = pts1_homo[i]
        pt2 = pts2_homo[i]
        residual_i = pt2.T @ F_CV2 @ pt1
        avg_res += residual_i
        # print(f"residual_i {i} = {residual_i}")
    print("================================================")
    print("avg_res", avg_res / len(pts1))
    # Note: trying out sampson Distance error --> close to 0 is good
    # This is better because the F
    avg_sampson_res = 0
    for i in range(len(pts1)):
        pt1 = pts1_homo[i]
        pt2 = pts2_homo[i]
        sampson_distance = cv2.sampsonDistance(pt1, pt2, F_CV2)
        avg_sampson_res += sampson_distance
        # print(f"sampson_distance {i} = {sampson_distance}")
    print("================================================")
    print("avg_sampson_res", avg_sampson_res / len(pts1))
    # R1, R2, t = cv2.decomposeEssentialMat(E)
    # print("R1", R1)
    # print("R2", R2)
    # print("t", t)

    # recover pose by decomposing the essential amtrix to find R and t
    retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=np.asarray(INTRINSIC_MATRICES["intrinsics_k"]))
    print("RETVAL", retval)
    print("R", R)
    print("t", t)

    retval, R, t, mask = cv2.recoverPose(E_from_F, pts1, pts2, cameraMatrix=np.asarray(INTRINSIC_MATRICES["intrinsics_k"]))
    print("RETVAL", retval)
    print("R", R)
    print("t", t)

    # exit()

    # Construct projection matrix P1 and P2 --> P = K * [R | t]
    P1 = np.asarray(INTRINSIC_MATRICES["intrinsics_k"]) @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.asarray(INTRINSIC_MATRICES["intrinsics_k"]) @ np.hstack((R, t))
    print(P1) # assumes that this is the global origin
    print(P2)

    # get 3D points
    print(np.shape(pts1.T))
    print(np.shape(pts2.T))
    print(np.shape(P1))
    print(np.shape(P2))
    pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_3d = pts_4d / pts_4d[3]
    # print(pts_3d)
    # convert to N x 3 matrix for plotting, and ignore the last row since it's all 1s, normalized
    # the above is in the format of [[x1, x2... xn], [y1, y2... yn], [z1, z2... zn], [1, 1... 1]]
    # convert it to [[x1, y1, z1], [x2, y2, z2], ... [xn, yn, zn]]
    pts_3d = pts_3d[:3, :].T.reshape(-1, 3)
    print(f"3D points = {pts_3d.shape}")
    # print(f"3D points = {pts_3d}")

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
