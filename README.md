# sfm


SURF isn't giving great features

Added manual feature points

Ran RANSAC to see which ones fail the homography transform


# DONE: need to calibrate iphone 16 pro camera to get Intrinsics and fisheye matrix
  --> use fisheye calibration procedure with the grid --> DONE



# todo: issues:
# ensure essential matrix is correct --> residuals are NOT near 0
# draw epipolar lines ... this will tell me if my F and/or E matrices are correct. --> these look like shit


# todo: try using better matching algorithm

## FLANN parameters and initialize
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

## Matching descriptor using KNN algorithm
matches = flann.knnMatch(desA, desB, k=2)


# todo:
1. take pictures that are closer together - DONE
2. try using FLANN matcher with SIFT - DONE
3. visualize epipolar lines - INCORRECT
4. check residual with X2 @ F @ X1 for each point --> should be near zero!!! - DONE
5. If all of the above are good, try to reconstruct the point - DONE
6. Document current state
7. Add todo for structure from Motion with multiple images and a pipeline!!

# Setup Instructions
1. Create conda environment for Python 3.10
2. pip3 install -r requirements.txt
3. pip install -e .

# Definitions

### Homography Matrix
A 3x3 matrix that maps images coordinates from one plane to another plane. This can be obtained by fetching 4 pairs of non-colinear points and optimizing the 8 matrix parameters

### Fundamental Matrix
TODO:

### Essential Matrix
TODO:

### Homogeneous Coordinates
TODO:


### Sampson Distance
When computing the residual, to verify the Fundamental Matrix, the value of the residual is not a geometric distance (like pixels) but an algebraic error value scaled by a normalization constraint.
This means that a simple computation of pt2.T @ F @ pt1 may not be as accurate actually computing the Sampson Distance for each point.
Note that all of the points must be in homogenous coordinates.

### Epipolar Lines
TODO: parallel when there is only a translation, but no rotation?
When there is no rotation, the epipolar lines will be parallel, thus the epipole (center of where the lines intersect) will be at infinity
When there is rotation, the epipolar lines will converge to a single point.

# TODO:
- show projected transform points with homography
