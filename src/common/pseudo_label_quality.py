import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter

def mean_confidence(prob_matrix):
    """
    计算每张图片的平均最大置信度
    prob_matrix: [batch, num_classes, H, W]
    返回: [batch] 平均置信度
    """
    max_probs = np.max(prob_matrix, axis=1)
    return np.mean(max_probs, axis=(1,2))

def connectivity_check(pseudo_labels, min_area=100):
    """
    检查每张图片的连通域面积是否达标
    pseudo_labels: [batch, H, W]
    返回: [batch] 是否通过连通性检查
    """
    batch = pseudo_labels.shape[0]
    results = []
    for i in range(batch):
        img = pseudo_labels[i]
        passed = False
        for cls in np.unique(img):
            if cls == 0: continue  # 忽略背景
            mask = (img == cls)
            labeled = label(mask)
            props = regionprops(labeled)
            if any([p.area >= min_area for p in props]):
                passed = True
                break
        results.append(passed)
    return np.array(results)

def boundary_smoothness(pseudo_labels, sigma=1.5, threshold=0.1):
    """
    检查边界平滑度（高斯模糊后变化量小于阈值则认为平滑）
    pseudo_labels: [batch, H, W]
    返回: [batch] 是否通过边界平滑度检查
    """
    batch = pseudo_labels.shape[0]
    results = []
    for i in range(batch):
        img = pseudo_labels[i].astype(float)
        blurred = gaussian_filter(img, sigma=sigma)
        diff = np.abs(img - blurred)
        mean_diff = np.mean(diff)
        results.append(mean_diff < threshold)
    return np.array(results)

def quality_filter(prob_matrix, pseudo_labels, conf_thresh=0.8, min_area=100, smooth_thresh=0.1):
    """
    综合质控，返回通过质控的掩码
    """
    conf_mask = mean_confidence(prob_matrix) > conf_thresh
    conn_mask = connectivity_check(pseudo_labels, min_area)
    smooth_mask = boundary_smoothness(pseudo_labels, threshold=smooth_thresh)
    final_mask = conf_mask & conn_mask & smooth_mask
    return final_mask

