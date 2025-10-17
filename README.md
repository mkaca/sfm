# sfm
Structure From Motion project that computes various transformations between images and recreates the scene in 3D.

## Caveats
1. Motion between images cannot be too great.
2. Scene must have a sufficient amount of unique feature points.
3. Feature points must NOT lie on the same plane --> this will result in poor E and F matrices

## Setup Instructions
1. Create conda environment for Python 3.10
2. pip3 install -r requirements.txt
3. pip install -e . <br>

The Intrinsic matrix was found by calibrating the Iphone 16 Pro rear camera at 1x magnification, using half of the default resolution.


## Definitions

### Homography Matrix
A 3x3 matrix that maps images coordinates from one plane to another plane. This can be obtained by fetching 4 pairs of non-colinear points and optimizing the 8 matrix parameters

### Fundamental Matrix
A 3x3 matrix (F) that describes teh epipolar geometry between 2 images of the same 3D scene. <br>
Given a point in camera1's coordinate frame and view, the F matrix tells us where the corresponding point lies in camera2's coordinate frame and view. v
The images do NOT have to be calibrated. <br>
This matrix contains the same information as the Essential Matrix, in additoin to information about hte intrinsics of both cameras so that they can be related in pixel coordinates.

### Essential Matrix
A 3x3 matrix representation of the epipolar geometry in normalized camera coordinates. <br>
E = t x R, where R = rotation matrix, and t = translation vector (without scale) <br>
This REQUIRES calibrated images; points must be transformed to normalized camera coordinates. <br>
Primarly used to recover the 3D motion between cameras. <br>
E = K_transposed x F x K, where K is the intrinsic 3x3 matrix of the camera. <br>

### Sampson Distance
When computing the residual, to verify the Fundamental Matrix, the value of the residual is not a geometric distance (like pixels) but an algebraic error value scaled by a normalization constraint.
This means that a simple computation of pt2.T @ F @ pt1 may not be as accurate actually computing the Sampson Distance for each point.
Note that all of the points must be in homogenous coordinates.

### Epipolar Lines
When there is no rotation, the epipolar lines will be parallel, thus the epipole (center of where the lines intersect) will be at infinity
When there is rotation, the epipolar lines will converge to a single point.

### Projection Matrix
A 3x4 matrix which is the product of the Intrinsic Camera Matrix (K) and the Extrinsic Matrix ([R∣t]) <br>
P = K x [R∣t] <br>

# TODO:
- show projected transform points with homography
- Add todo for structure from Motion with multiple images and a pipeline!!
