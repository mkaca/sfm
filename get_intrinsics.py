
import os
import cv2
import numpy as np

ORIGINAL_DIMENSIONS = (5712, 4284, 3)
RESIZE_DIMENSIONS = (ORIGINAL_DIMENSIONS[1]//2, ORIGINAL_DIMENSIONS[0]//2)

CHECKERBOARD_DIMS = (8, 6)

DEBUG = True


def map_to_rgb(row, col):
    # Normalize values to [0, 1]
    normalized_row = row / CHECKERBOARD_DIMS[0]
    normalized_col = col / CHECKERBOARD_DIMS[1]

    # Map to RGB (example mapping)
    r = int(normalized_row * 255)  # Red component from row
    g = int(normalized_col * 255)  # Green component from column
    b = int((1 - (normalized_row + normalized_col) / 2) * 255)  # Blue as inverse average

    # Ensure values are in [0, 255] range
    return (r, g, b)


def get_calibration_params(images, debug=False):
    corners_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD_DIMS[0] * CHECKERBOARD_DIMS[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD_DIMS[0], 0:CHECKERBOARD_DIMS[1]].T.reshape(-1, 2)
    N_imm = 0

    objpoints = [] # IDs of points
    imgpoints = [] # 2d points in image plane.

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_DIMS, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1), corners_criteria)
            if debug:
                # print(f"corners = {corners}")
                for i in range(0, len(corners)):
                    # This shows you if the points match up with the checkerboard and if their IDs are associated correctly
                    color = map_to_rgb(objp[0,i,0], objp[0,i,1])  # (0,255,0)
                    cv2.circle(img, (int(corners[i,0,0]), int(corners[i,0,1])), 25, color, 2)
                cv2.imshow("debug img", img)
                cv2.waitKey(0)
            imgpoints.append(corners)
            print(f"added img index {N_imm} with {len(corners)} corners")
            N_imm += 1
        else:
            print(f"Could not use image with index{N_imm} for calibration")
        ###

    if len(objpoints) == 0:
        raise SystemError("Image calibration path is empty!!")

    print(f"obj points shape = {np.shape(objpoints)}")
    print(f"img points shape = {np.shape(imgpoints)}")

    # calculate K & D
    K = np.zeros((3, 3))  # np.eye(3)
    D = np.zeros((4, 1))
    ignore_one_image = False
    ignore_index = 0
    for _ in range(N_imm):
        try:
            rvecs_size = N_imm
            obj_points_local = np.copy(objpoints)
            img_points_local = np.copy(imgpoints)
            if ignore_one_image:
                print(f"ignoring index {ignore_index}")
                rvecs_size -= 1
                obj_points_local = np.delete(objpoints, ignore_index, axis=0)
                img_points_local = np.delete(imgpoints, ignore_index, axis=0)
                print(f"new obj points shape = {np.shape(obj_points_local)}")

            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(rvecs_size)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(rvecs_size)]
            retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj_points_local,
                img_points_local,
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4))
            break
        except Exception as e:
            if ignore_one_image:
                ignore_index += 1
            if "Ill-conditioned matrix" in str(e):
                print("one of the images is bad!!! Finding out which one....")
                ignore_one_image = True

    # check if the calibration is good
    print(f"K = {K}")
    print(f"D = {D}")
    # for x, y in [[320, 240],
    #     [553, 415],
    #     [563, 425],
    #     [150, 100],
    #     [87, 65],
    #     [67, 45],
    #     [60, 40],
    #     [400, 300],
    #     [580, 440],  # 60, 40  -> HARD LIMIT
    #     [570, 430],
    #     ]:
    #     rect_x, rect_y = test_point(K, D, x, y)
    #     if rect_x < -10000 or rect_x > 10000 or rect_y < -10000 or rect_y > 10000:
    #         print(f"point {rect_x}, {rect_y} is out of bounds.. changing D matrix to default")
    #         D = np.asarray([[-0.07512307],
    #             [-0.03045685],
    #             [ 0.04445467],
    #             [ 0.00406436]])
    #         break

    return K, D


def get_images():
    images = sorted(os.listdir("calibration_images_iphone_16pro"))
    # ignore all images without .jpg in the name
    images = [cv2.resize(cv2.imread(os.path.join("calibration_images_iphone_16pro", img)), RESIZE_DIMENSIONS) for img in images if img.endswith(".JPG")]
    return images


def main():
    images = get_images()
    K, D = get_calibration_params(images, debug=DEBUG)
    print(f"K = {K}")
    print(f"D = {D}")


if __name__ == "__main__":
    main()
