import cv2
from src.constants import RESIZE_DIMENSIONS, FEATURE_MATCHES_LIMIT
import os

SIFT_CONTRAST_THRESHOLD = 0.095 # a higher value will result in stronger, but fewer features
SIFT_EDGE_THRESHOLD = 15 # decrease this to filter more edges; higher value == more features
SIFT_SIGMA = 1.6 # Gaussian sigma for initial smoothing (default is 1.6)
SIFT_N_OCTAVE_LAYERS = 5 # Layers per octave (default is 3)
LOWEs_THRESOLD= 0.70 # Lowe's ratio test, the lower the value, the more strict the filter
BLUR_PRE_SIFT = True
USE_HISTOGRAM_EQ = False # Apply histogram equalization
USE_CLAHE = False # Apply CLAHE for adaptive contrast


def get_matches(features_des, matcher_type="bf"):
    # match features between images
    if matcher_type == "bf":
        matcher = cv2.BFMatcher()  # Brute Force Matcher
    elif matcher_type == "flann":
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise NotImplementedError(f"Matcher type {type} not implemented")

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

        # Apply histogram equalization to improve contrast
        if USE_HISTOGRAM_EQ:
            gray = cv2.equalizeHist(gray)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better local contrast
        if USE_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

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
