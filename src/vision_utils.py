import cv2
import numpy as np
from src.constants import FEATURE_MATCHES_LIMIT
from src.visualize import visualize_points_on_images


def getNormalisationMat(pts):
    """Calculate the nomalisation matrix of the given coordinate points set

    Parameters
    ----------
    pts : int numpy.ndarray, shape (n_correspondences, 2)
        An array of coordinate points.
    -------
    Return
    normalisationMat : float numpy.ndarray, shape (3, 3)
        The normalisation matrix of the given point set.
        This matrix translate and scale the points so that the mean coordinate is at (0,0) and average distance to (0,0) is sqrt(2)
    """

    pts = np.float64(pts)
    mean = np.array(np.sum(pts, axis=0) / len(pts), dtype=np.float64)
    scale = np.sum(np.linalg.norm(pts - mean, axis=1), axis=0) / (len(pts) * np.sqrt(2.0))
    normalisationMat = np.array(
        [
            [1.0 / scale, 0.0, -mean[0] / scale],
            [0.0, 1.0 / scale, -mean[1] / scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return normalisationMat


def conv2HomogeneousCoordinates(ptsA, ptsB):
    """Convert points from cartesian coordinate to homogeneous coordinate

    Basically just add a 1 to the end of the points
    # for example, (1, 2) becomes (1, 2, 1)
    # and ([1, 2], [3, 4]) becomes ([1, 2, 1], [3, 4, 1])

    Parameters
    ----------
    ptsA : int numpy.ndarray, shape (n_correspondences, 2) or int numpy.ndarray, shape (2,)
        A coordinate or an array of coordinates of correspondences from image A.
    ptsB : int numpy.ndarray, shape (n_correspondences, 2) or int numpy.ndarray, shape (2,)
        A coordinate or an array of coordinates of correspondences from image B.
    -------
    Return
    ptsA_homo : float64 numpy.ndarray, shape (n_correspondences, 3) or int numpy.ndarray, shape (3,)
        A coordinate or an array of coordinates of correspondences from image A, in the form of homogeneous coordinate.
    ptsB_homo : float64 numpy.ndarray, shape (n_correspondences, 3) or int numpy.ndarray, shape (3,)
        A coordinate or an array of coordinates of correspondences from image B, in the form of homogeneous coordinate.
    """

    if ptsA.ndim == 1:
        ptsA_homo = np.pad(ptsA, (0, 1), "constant", constant_values=1.0)
        ptsB_homo = np.pad(ptsB, (0, 1), "constant", constant_values=1.0)
    else:
        ptsA_homo = np.pad(ptsA, [(0, 0), (0, 1)], "constant", constant_values=1.0)
        ptsB_homo = np.pad(ptsB, [(0, 0), (0, 1)], "constant", constant_values=1.0)

    return np.float64(ptsA_homo), np.float64(ptsB_homo)


def get_2d_points_of_matches(matches_between_2_images, features_kp, feature_match_limit=FEATURE_MATCHES_LIMIT):
    # Note: Only between 2 images
    # get the 2D points of each image of the best 10 matches between the 2 images
    pts1 = []
    pts2 = []

    sorted_matches = sorted(matches_between_2_images, key=lambda x: x.distance)
    best_matches = sorted_matches[:feature_match_limit]

    for i in range(len(best_matches)):
        pts1.append(features_kp[0][best_matches[i].queryIdx].pt)
        pts2.append(features_kp[1][best_matches[i].trainIdx].pt)
    return np.asarray(pts1), np.asarray(pts2)


def filter_points_with_homography(pts1, pts2, image2, visualize=True):
    """
    apply homography with RANSAC to remove bad matches (outliers) --> FROM IMAGE1 to IMAGE2
    So, H is computed from img1 to img2, then the points from img1 are transformed to img2 plane and visualized on img2

    """
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
    if visualize:
        visualize_points_on_images(pts1_transformed_corrected, pts2, image2, image2, save_path_prefix="pts_visualization_H_transformed_")
    return pts1, pts2


def validate_epipolar_constraint(pts1_homo, pts2_homo, F):
    # Validate epipolar constraint: x2^T * F * x1 = 0
    # VALIDATION - confirm the F matrix actually works ... by calculating the residual
    # Note: this can have errors due to scaling factor issues... hence Sampson Distance is also used
    assert (pts1_homo.shape == pts2_homo.shape)
    print("\n=== EPIPOLAR CONSTRAINT VALIDATION ===")
    avg_res = 0
    for i in range(len(pts1_homo)):  # Check first 5 points
        pt1 = pts1_homo[i]
        pt2 = pts2_homo[i]
        residual_i = pt2.T @ F @ pt1
        avg_res += residual_i
        # print(f"residual_i {i} = {residual_i}")
    print("================================================")
    print("avg_res", avg_res / len(pts1_homo))
    # Note: trying out sampson Distance error --> close to 0 is good
    # This is better because the F
    avg_sampson_res = 0
    for i in range(len(pts1_homo)):
        pt1 = pts1_homo[i]
        pt2 = pts2_homo[i]
        sampson_distance = cv2.sampsonDistance(pt1, pt2, F)
        avg_sampson_res += sampson_distance
        # print(f"sampson_distance {i} = {sampson_distance}")
    print("================================================")
    print("avg_sampson_res", avg_sampson_res / len(pts1_homo))


def get_3d_points(pts1, pts2, P1, P2):
    # get 3D points
    pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_3d = pts_4d / pts_4d[3]
    # print(pts_3d)
    # convert to N x 3 matrix for plotting, and ignore the last row since it's all 1s, normalized
    # the above is in the format of [[x1, x2... xn], [y1, y2... yn], [z1, z2... zn], [1, 1... 1]]
    # convert it to [[x1, y1, z1], [x2, y2, z2], ... [xn, yn, zn]]
    pts_3d = pts_3d[:3, :].T.reshape(-1, 3)
    print(f"3D points = {pts_3d.shape}")
    # print(f"3D points = {pts_3d}")
    return pts_3d
