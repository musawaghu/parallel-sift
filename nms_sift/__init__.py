"""Pure-Python NMS-SIFT plus tiling helpers for the parallelism study."""
from .sift import computeKeypointsAndDescriptors, detect_and_describe, to_keypoint
from .tiling import merge_keypoints, split_tiles
from .parallel import sift_tiled
