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

ORIGINAL_DIMENSIONS = (5712, 4284, 3)
FEATURE_MATCHES_LIMIT = 30
SIFT_CONTRAST_THRESHOLD = 0.195 # a higher value will result in stronger, but fewer features
SIFT_EDGE_THRESHOLD = 35 # decrease this to filter more edges; higher value == more features
SIFT_SIGMA = 1.6 # Gaussian sigma for initial smoothing (default is 1.6)
SIFT_N_OCTAVE_LAYERS = 3 # Layers per octave (default is 3)
LOWEs_THRESOLD= 0.70 # Lowe's ratio test, the lower the value, the more strict the filter
BLUR_PRE_SIFT = True

# Iphone 15 Pro Max intrinsic Matrix: todo: Is this accurate?
INTRINSIC_MATRICES = {
  "distortion_coefficients":
  [12.881983453085578,-6.12469777944087,-0.0009314664065349503,-0.000716748438270507,4.706217456528554,12.883652328360427,-6.038613102475213,4.679237475833294,0.0,0.0,0.0,0.0,0.0,0.0],
  "intrinsics_k": [ # uses center for cx and cy
    [873, 0, ORIGINAL_DIMENSIONS[0]//4],
    [0, 875, ORIGINAL_DIMENSIONS[1]//4],
    [0, 0, 1]
  ]
#     "intrinsics_k": [
#     [872.6762201310132, 0.0, 956.2263400865058],
#     [0.0, 875.1309660292621, 536.7153914508558],
#     [0.0, 0.0, 1.0]
#   ]
}


RESIZE_DIMENSIONS = (ORIGINAL_DIMENSIONS[0]//2, ORIGINAL_DIMENSIONS[1]//2)
RESIZE_DIMENSIONS_RGB = (RESIZE_DIMENSIONS[0], RESIZE_DIMENSIONS[1], 3)


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


def visualize_points_on_images(pts1, pts2, image1, image2):
    # visualize the points
    img1 = cv2.imread(os.path.join("images", image1))
    img1 = cv2.resize(img1, RESIZE_DIMENSIONS)
    img2 = cv2.imread(os.path.join("images", image2))
    img2 = cv2.resize(img2, RESIZE_DIMENSIONS)
    size_of_points = len(pts1)
    decrement_step = 127 // size_of_points
    for i, point in enumerate(pts1):
        cv2.circle(img1, (int(point[0]), int(point[1])), 15, (decrement_step*i, 0, 255-decrement_step*i), -1)
    for i, point in enumerate(pts2):
        cv2.circle(img2, (int(point[0]), int(point[1])), 15, (decrement_step*i, 0, 255-decrement_step*i), -1)
    cv2.imwrite("pts_visualization1.jpg", img1)
    cv2.imwrite("pts_visualization2.jpg", img2)
    print("Saved points visualization as 'pts_visualization1.jpg' and 'pts_visualization2.jpg'")


def get_matches(features_des):
    # match features between images
    matcher = cv2.BFMatcher()  # Brute Force Matcher
    # Alternative: cv2.FlannBasedMatcher() for faster matching with many features
    matches = []
    # Match descriptors between consecutive image pairs
    for i in range(len(features_des) - 1):
        if features_des[i] is not None and features_des[i+1] is not None:
            # Find matches using KNN (k=2 for ratio test)
            knn_matches = matcher.knnMatch(features_des[i], features_des[i+1], k=2)

            # Apply Lowe's ratio test to filter good matches (made by guy who wrote SIFT)
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < LOWEs_THRESOLD * n.distance:  # Lowe's ratio test, the lower the value, the more strict the filter
                        good_matches.append(m)

            matches.append(good_matches)
            print(f"Found {len(good_matches)} good matches between image {i} and {i+1}")
            # todo: temp
            if i > 0:
                break
    return matches


def get_features(images):
    # find features in each image
    features_kp = []
    features_des = []
    for image in images:
        img = cv2.imread(os.path.join("images", image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, RESIZE_DIMENSIONS)

        # Enhanced preprocessing for better feature detection
        if BLUR_PRE_SIFT:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        sift = cv2.SIFT_create(
            contrastThreshold=SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=SIFT_EDGE_THRESHOLD,
            sigma=SIFT_SIGMA,
            nOctaveLayers=SIFT_N_OCTAVE_LAYERS
        )
        kp, des = sift.detectAndCompute(gray, None)
        print(f"Found {len(kp)} features in {image}")
        features_kp.append(kp)
        features_des.append(des)
    return features_kp, features_des


# todo: only gets matches between first 2 images right now
def get_2d_points(matches, features_kp):
    # get the 2D points of each image of the best 10 matches between the 2 images
    pts1 = []
    pts2 = []
    for i in range(min(FEATURE_MATCHES_LIMIT, len(matches[0]))):
        # print(matches[0][i].queryIdx)
        # print(matches[0][i].trainIdx)
        # print(features_kp[0][matches[0][i].queryIdx].pt)
        # print(features_kp[1][matches[0][i].trainIdx].pt)
        pts1.append(features_kp[0][matches[0][i].queryIdx].pt)
        pts2.append(features_kp[1][matches[0][i].trainIdx].pt)
    return np.asarray(pts1), np.asarray(pts2)


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

    features_kp, features_des = get_features(images)

    matches = get_matches(features_des)

    # Visualize matches between first two images
    visualize_matches(matches, features_kp, images)

    pts1, pts2 = get_2d_points(matches, features_kp)
    print(pts1)
    print(pts2)

    # apply homography with RANSAC to remove bad matches (outliers)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.95)
    print(H)
    print(mask)
    # remove all points that are not in the mask
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    print(pts1.shape)
    print(pts2.shape)

    visualize_points_on_images(pts1, pts2, images[0], images[1])

    # get fundamental matrix via triangulation
    # pts1 and pts2 are the matched 2D points (Nx2 arrays)
    # RANSAC is essential here to remove outliers
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)

    # get Essential Matrix --> assuming basic to derive Intrinsic Matrix
    # E, mask = cv2.findEssentialMat(pts1, pts2, INTRINSIC_MATRICES["intrinsics_k"], INTRINSIC_MATRICES["distortion_coefficients"], method=cv2.RANSAC)
    E, mask = cv2.findEssentialMat(pts1, pts2, np.asarray(INTRINSIC_MATRICES["intrinsics_k"]), method=cv2.RANSAC)
    print(E)

    # recover pose by decomposing the essential amtrix to find R and t
    retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=np.asarray(INTRINSIC_MATRICES["intrinsics_k"]))
    print(R)
    print(t)

    # Construct projection matrix P1 and P2 --> P = K * [R | t]
    P1 = np.asarray(INTRINSIC_MATRICES["intrinsics_k"])
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
    # print(f"3D points = {pts_3d}")

    # use matplot lib to visualize the 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2])
    plt.show()

    # todo: matches are OK, but not a lot of features are being foun
    # todo: projection doesn't seem great --> I don't see a clear box / plane being identified in the 3D points
    # todo: add gaussian splatting to visualize the 3D points



if __name__ == "__main__":
    main()
