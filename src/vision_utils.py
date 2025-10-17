import numpy as np
from src.constants import FEATURE_MATCHES_LIMIT


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


def getCorrespondencesEpilines(ptsA, ptsB, FundMat):
    """Compute the epipolar lines on image A and B based on the
       correspondences and the fundamental matrix

    Parameters
    ----------
    ptsA : int numpy.ndarray, shape (n_correspondences, 3) or int numpy.ndarray, shape (3,)
        A coordinate or an array of coordinates of correspondences from image A.
    ptsB : int numpy.ndarray, shape (n_correspondences, 3) or int numpy.ndarray, shape (3,)
        A coordinate or an array of coordinates of correspondences from image B.
    F : float numpy.ndarray, shape (3, 3)
        Fundamental matrix.
    -------
    Return
    lines : float, numpy.ndarray, shape (num_points, 3)
        An array of the epipolar lines.  Each epipolar line is represented as an array of
        three float number [a, b, c].  [a, b, c] are the coefficients of a line ax + by + c = 0.
        Lines are normalised ao that sqrt(a^2 + b^2) = 1
    """

    # Convert data type to float64
    ptsA = np.float64(ptsA)
    ptsB = np.float64(ptsB)

    # If input is only a point
    if ptsA.ndim == 1:
        # Compute the lines
        linesA = np.array(ptsB @ FundMat, dtype=np.float64)
        linesB = np.array(FundMat @ ptsA.T, dtype=np.float64)

        # Normalise
        aA, bA, cA = linesA
        aB, bB, cB = linesB
        linesA = linesA / np.sqrt(aA * aA + bA * bA)
        linesB = linesB / np.sqrt(aB * aB + bB * bB)
    else:
        # Compute the lines
        linesA = np.array([pB @ FundMat for pB in ptsB], dtype=np.float64)
        linesB = np.array([FundMat @ pA.T for pA in ptsA], dtype=np.float64)

        # Normalise
        linesA = np.array(
            [
                np.array([ a / np.sqrt(a * a + b * b), b / np.sqrt(a * a + b * b), c / np.sqrt(a * a + b * b), ], dtype=np.float64)
                for a, b, c in linesA
            ],
            dtype=np.float64,
        )
        linesB = np.array(
            [
                np.array([ a / np.sqrt(a * a + b * b), b / np.sqrt(a * a + b * b), c / np.sqrt(a * a + b * b), ], dtype=np.float64,)
                for a, b, c in linesB
            ],
            dtype=np.float64,
        )

    return linesA, linesB


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
    for i in range(min(feature_match_limit, len(matches_between_2_images))):
        pts1.append(features_kp[0][matches_between_2_images[i].queryIdx].pt)
        pts2.append(features_kp[1][matches_between_2_images[i].trainIdx].pt)
    return np.asarray(pts1), np.asarray(pts2)
