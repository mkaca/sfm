ORIGINAL_DIMENSIONS = (5712, 4284, 3)
RESIZE_DIMENSIONS = (ORIGINAL_DIMENSIONS[1]//2, ORIGINAL_DIMENSIONS[0]//2)
RESIZE_DIMENSIONS_RGB = (RESIZE_DIMENSIONS[1], RESIZE_DIMENSIONS[0], 3)

FEATURE_MATCHES_LIMIT = 300


# Iphone 16 Pro Max intrinsic Matrix --> calibrated using get_intrinsics.py
"""
  K = [[2.05233093e+03 0.00000000e+00 1.06474865e+03]
 [0.00000000e+00 2.04960308e+03 1.43317181e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
"""
INTRINSIC_MATRICES = {
  "distortion_coefficients": [
    [  0.37190428],
    [  1.94149091],
    [ -5.65629543],
    [-20.64678548],
  ],
  "intrinsics_k": [ # uses center for cx and cy
    [2052, 0, 1064],
    [0, 2049, 1433],
    [0, 0, 1]
  ],
}
