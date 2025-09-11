import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_opening, binary_closing

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

def denoise_pseudo_label(pseudo_labels, min_area=100, morph_op=None, morph_structure=None):
    """
    对伪标签去噪：抹除小连通域，并可选做形态学开/闭运算和空洞填充
    pseudo_labels: [batch, H, W]
    min_area: 小于该面积的连通域会被抹除
    morph_op: None/'open'/'close'，可选形态学操作
    morph_structure: 结构元素，如np.ones((3,3))
    返回: 去噪后的伪标签 [batch, H, W]
    """
    batch, H, W = pseudo_labels.shape
    out = np.zeros_like(pseudo_labels)
    for i in range(batch):
        img = pseudo_labels[i]
        new_img = np.zeros_like(img)
        for cls in np.unique(img):
            if cls == 0: continue
            mask = (img == cls)
            labeled = label(mask)
            for region in regionprops(labeled):
                if region.area >= min_area:
                    new_img[region.coords[:,0], region.coords[:,1]] = cls
        # 空洞填充
        for cls in np.unique(new_img):
            if cls == 0: continue
            mask = (new_img == cls)
            filled = binary_fill_holes(mask)
            new_img[filled & (new_img == 0)] = cls
        # 形态学操作
        if morph_op is not None and morph_structure is not None:
            for cls in np.unique(new_img):
                if cls == 0: continue
                mask = (new_img == cls)
                if morph_op == 'open':
                    mask = binary_opening(mask, structure=morph_structure)
                elif morph_op == 'close':
                    mask = binary_closing(mask, structure=morph_structure)
                new_img[mask] = cls
        out[i] = new_img
    return out

def mask_quality_filter_with_pixel_mask(prob_matrix, pseudo_labels, pixel_masks, conf_thresh=0.8, min_area=100, smooth_thresh=0.1, pixel_mask_thresh=0.7):
    """
    综合质控，增加像素级门控通过率判断。
    返回: final_mask, shape=[N,]，表示每帧是否通过全部质控
    """
    conf_mask = quality_filter(prob_matrix, pseudo_labels, conf_thresh, min_area, smooth_thresh)
    if pixel_masks.ndim == 2:
        pixel_masks = pixel_masks[np.newaxis, ...]
    pixel_pass_ratio = np.mean(pixel_masks, axis=(1,2))  # 每帧True比例
    pixel_mask_mask = pixel_pass_ratio > pixel_mask_thresh
    final_mask = conf_mask & pixel_mask_mask
    return final_mask


def pixel_gate_mask(prob_matrix, pseudo_labels, conf_thresh=0.8):
    """
    生成像素级门控掩码，基于置信度阈值过滤不可靠的像素预测

    参数:
        prob_matrix: 概率矩阵，形状为 [N, C, H, W] 或 [N, H, W]
        pseudo_labels: 伪标签，形状为 [N, H, W]
        conf_thresh: 置信度阈值，默认0.8

    返回:
        pixel_mask: 布尔掩码，形状为 [N, H, W]，True表示该像素可靠
    """
    # 处理二分类和多分类情况
    if prob_matrix.ndim == 4:  # [N, C, H, W] 多分类
        # 获取每个像素的最大概率值
        max_probs = np.max(prob_matrix, axis=1)
        # 获取预测类别
        pred_classes = np.argmax(prob_matrix, axis=1)
    else:  # [N, H, W] 二分类
        max_probs = prob_matrix
        pred_classes = (prob_matrix > 0.5).astype(np.uint8)

    # 创建像素掩码：置信度高于阈值且预测类别与伪标签一致
    pixel_mask = (max_probs > conf_thresh) & (pred_classes == pseudo_labels)

    return pixel_mask