# src/viz/colorize.py
import numpy as np
import cv2
from typing import Dict, Tuple
from src.common.constants import PALETTE, IGNORE_INDEX
# from . import __init__

# Convert instance segmentation IDs to colors
def id_to_color(
    mask_id: np.ndarray,
    palette: Dict[int, Tuple[int, int, int]] = PALETTE,
) -> np.ndarray:
    # height and width
    height, width = mask_id.shape
    color = np.zeros((height, width, 3), dtype=np.uint8) # Initialize color image
    for k, rgb in palette.items():
        color[mask_id == k] = rgb
    color[mask_id == IGNORE_INDEX] = (50, 50, 50) # Set ignore index to gray
    return color

# Create a triplet image for visualization
def make_triplet(
    image_bgr: np.ndarray,
    groundtruth_id: np.ndarray,
    pred_id: np.ndarray,
    w_pad: int = 8
) -> np.ndarray:
    groundtruth_color = id_to_color(groundtruth_id)
    pred_color        = id_to_color(pred_id)
    pad               = np.ones((image_bgr.shape[0], w_pad, 3), dtype=np.uint8) * 255 # white
    triplet           = np.concatenate([image_bgr, pad, groundtruth_color, pad, pred_color], axis=1)
    return triplet

def overlay(image_bgr: np.ndarray, mask_id: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = id_to_color(mask_id)
    return cv2.addWeighted(image_bgr, 1.0, color, alpha, 0.0)