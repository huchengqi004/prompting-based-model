import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import sys
import shutil
import time
import pandas as pd
from tkinter import filedialog, Tk, simpledialog
from itertools import combinations

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# ==================== 用户可调参数 ====================
Maskprompt = True          # True: 使用粗糙前景掩码进行预测 (mask_input)
PointBoxprompt = True      # True: 使用点/框提示进行预测 (来自txt文件)
prefix_len = 2
# 当两者都为True时，两种方式均执行；否则只执行对应的方式。

#     # 路径配置（请根据实际情况修改）

# txt_folder = r'A:\B_diskexpand\Code\GrMflakes\data\OM-Grflakes\Test\images\prompt_results'
# txt_folder = r'A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\coarse-fore100back150\image\prompt_results-grid50'
# rough_mask_folder = r"A:\B_diskexpand\Code\GrMflakes\data\OM-Grflakes\Test\images\GrSiO2_npy\GrSiO2otsu_npy"
# rough_mask_folder = r"A:\B_diskexpand\Code\GrMflakes\data\OM-Grflakes\Test\images\GrSiO2_npy\GrSiO2BGR_npy"
# rough_mask_folder = r"A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\coarse-fore100back150\OMGrCu-npy"
# rough_mask_folder = None
# image_folder = r"A:\B_diskexpand\Code\GrMflakes\data\OM-Grflakes\Test\images"
# image_folder = r"A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\coarse-fore100back150\image"
# ref_folder = r"A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\coarse-fore100back150\GT-22"


txt_folder = r'A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\AFM\image\prompt_results-grid 25'

# rough_mask_folder = r"A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\AFM\Fore_npy\10min-15dufore-npy"
rough_mask_folder = r"A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\AFM\Fore_npy\10min-15duHOGsinkBPF_th48-npy"
image_folder = r"A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\AFM\image"
ref_folder = r"A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\AFM\GT-AFM"




# ==================== 指标计算函数 ====================
def binary_pixel_accuracy(binary_image_predicted, binary_image_true):
    """返回15个指标，小数形式"""
    TP = np.sum((binary_image_true == 255) & (binary_image_predicted == 255))
    FP = np.sum((binary_image_true == 0)    & (binary_image_predicted == 255))
    TN = np.sum((binary_image_true == 0)    & (binary_image_predicted == 0))
    FN = np.sum((binary_image_true == 255)  & (binary_image_predicted == 0))

    total_pixels = TP + FP + TN + FN
    if total_pixels == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    precision_pos = TP / (TP + FP) if (TP+FP) > 0 else 0.0
    precision_neg = TN / (TN + FN) if (TN+FN) > 0 else 0.0
    MPA = (precision_pos + precision_neg) / 2
    IOU_pos = TP / (TP + FP + FN) if (TP+FP+FN) > 0 else 0.0
    IOU_neg = TN / (TN + FN + FP) if (TN+FN+FP) > 0 else 0.0
    MIOU = (IOU_pos + IOU_neg) / 2
    PA = (TP + TN) / total_pixels
    Dice_fg = 2*TP / (2*TP + FP + FN) if (2*TP+FP+FN) > 0 else 1.0
    Dice_bg = 2*TN / (2*TN + FN + FP) if (2*TN+FN+FP) > 0 else 1.0
    mDice = (Dice_fg + Dice_bg) / 2
    Recall_fg = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    Recall_bg = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    mRecall = (Recall_fg + Recall_bg) / 2
    F1_fg = 2 * precision_pos * Recall_fg / (precision_pos + Recall_fg) if (precision_pos + Recall_fg) > 0 else 0.0
    F1_bg = 2 * precision_neg * Recall_bg / (precision_neg + Recall_bg) if (precision_neg + Recall_bg) > 0 else 0.0
    mF1 = (F1_fg + F1_bg) / 2

    return (PA, precision_pos, precision_neg, MPA,
            IOU_pos, MIOU, Dice_fg, Dice_bg, mDice,
            Recall_fg, Recall_bg, mRecall,
            F1_fg, F1_bg, mF1)

def prepare_binary_image(img_path, threshold=128):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary

def resize_to_match(img, ref_shape):
    if img.shape[:2] == ref_shape:
        return img
    return cv2.resize(img, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_NEAREST)

# ==================== SAM 辅助函数 ====================
def load_rough_mask(rough_mask_path, target_size=256):
    ext = os.path.splitext(rough_mask_path)[1].lower()
    if ext == '.npy':
        rough_mask = np.load(rough_mask_path)
        if rough_mask.ndim == 3:
            rough_mask = rough_mask[:, :, 0] if rough_mask.shape[-1] <= 3 else rough_mask[0]
        elif rough_mask.ndim == 4:
            rough_mask = rough_mask[0, :, :, 0] if rough_mask.shape[-1] <= 3 else rough_mask[0, 0]
        if rough_mask.max() > 1:
            rough_mask = rough_mask.astype(np.float32) / 255.0
        else:
            rough_mask = rough_mask.astype(np.float32)
    else:
        rough_mask = cv2.imread(rough_mask_path, cv2.IMREAD_GRAYSCALE)
        if rough_mask is None:
            return None
        rough_mask = rough_mask.astype(np.float32) / 255.0
    if rough_mask.shape[:2] != (target_size, target_size):
        rough_mask = cv2.resize(rough_mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return rough_mask[None, :, :]

def save_sam_predictions_otsu(
        masks, scores, logits, image, boxes, points, labels,
        save_root, prefix="", mode="", iteration=None,
        save_heatmap=True, save_logits_gray=True,
        rank_indices=None, save_pairwise_multiplication=False,
        logits_otsu=True, logits_for_vis=None,extra_suffix=""):
    """
    保存 SAM 预测结果。

    参数：
        masks:   SAM 返回的掩码。
                - 当 logits_otsu=True 时，masks 是原图尺寸的 logits；
                - 当 logits_otsu=False 时，masks 是二值掩码（bool 或 uint8）。
        scores:  各掩码的分数
        logits:  SAM 返回的低分辨率 logits（256x256），仅用于迭代，本函数不使用
        image:   原始图像 (H, W, 3)
        boxes, points, labels: 提示信息
        save_root, prefix, mode, iteration: 输出路径相关
        save_heatmap: 是否保存热力图（要求 logits 可用）
        save_logits_gray: 是否保存灰度图（要求 logits 可用）
        rank_indices: 只保存指定排名的掩码（按分数降序排名）
        save_pairwise_multiplication: 是否生成两两相乘融合（仅当 N=3 且 logits_otsu=True 时有效）
        logits_otsu:  True  -> 使用 Otsu 阈值化 logits 生成掩码
                      False -> 直接使用 masks 作为二值掩码
        logits_for_vis: 可选，原图尺寸的 logits，仅在 logits_otsu=False 时用于生成可视化
    """
    # 创建保存目录
    seg_dir_thr = os.path.join(save_root, "seg_masks_thr")
    prompt_dir = os.path.join(save_root, "prompt_masks")
    os.makedirs(seg_dir_thr, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)

    # 如果需要可视化（热力图或灰度图），创建相应目录
    if save_heatmap or save_logits_gray:
        seg_dir_gray = os.path.join(save_root, "seg_masks_gray")
        seg_dir_heatmap = os.path.join(save_root, "seg_masks_heatmap")
        os.makedirs(seg_dir_gray, exist_ok=True)
        os.makedirs(seg_dir_heatmap, exist_ok=True)

    H_orig, W_orig = image.shape[0], image.shape[1]

    # 构造迭代字符串
    if mode and iteration is not None:
        iter_str = f"{mode}{iteration}st"
    elif mode:
        iter_str = f"{mode}"
    else:
        iter_str = f"{iteration}st" if iteration is not None else ""

    # 按分数降序排序，如果指定 rank_indices 则只选择对应排名
    sorted_indices = np.argsort(scores)[::-1]
    total_masks = len(scores)

    if rank_indices is not None:
        selected_ranks = set(rank_indices)
        selected_indices = []
        for rank, idx in enumerate(sorted_indices, start=1):
            if rank in selected_ranks:
                selected_indices.append(idx)
        if not selected_indices:
            selected_indices = list(range(total_masks))
    else:
        selected_indices = list(range(total_masks))

    saved_masks = []
    m = 0
    for idx in selected_indices:
        score = scores[idx]

        # 决定用于可视化的 logits 数组
        if logits_otsu:
            # 如果 logits_otsu=True，masks 本身就是 logits
            logit_map = masks[idx]
        else:
            # 否则尝试从 logits_for_vis 获取
            if logits_for_vis is None:
                logit_map = None
            else:
                logit_map = logits_for_vis[idx]

        # 调整 logit_map 尺寸到原图，并计算概率图（如果可用）
        if logit_map is not None:
            if logit_map.ndim == 3:
                logit_map = logit_map[0]  # 安全处理
            if logit_map.shape[0] != H_orig or logit_map.shape[1] != W_orig:
                logit_map = cv2.resize(logit_map, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            prob_map = 1.0 / (1.0 + np.exp(-logit_map))
        else:
            prob_map = None

        # ---- 保存热力图 ----
        if save_heatmap and prob_map is not None:
            # OpenCV 版本，尺寸精确
            prob_uint8 = (prob_map * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_VIRIDIS)
            heatmap_name = f"{prefix}_{iter_str}_{m}_score{score:.3f}_heatmap{extra_suffix}.png"
            cv2.imwrite(os.path.join(seg_dir_heatmap, heatmap_name), heatmap_color)

            # 带 colorbar 的版本（matplotlib）
            fig, ax = plt.subplots(figsize=(W_orig/100 + 0.3, H_orig/100), dpi=100)
            im = ax.imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
            ax.axis('off')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Probability', rotation=270, labelpad=15)
            heatmap_cbar_name = f"{prefix}_{iter_str}_{m}_score{score:.3f}_heatmap_withcolorbar{extra_suffix}.png"
            plt.savefig(os.path.join(seg_dir_heatmap, heatmap_cbar_name), bbox_inches='tight', pad_inches=0)
            plt.close()

        # ---- 保存灰度图 ----
        if save_logits_gray and logit_map is not None:
            mean = np.mean(logit_map)
            std = np.std(logit_map)
            vmin = mean - 3 * std
            vmax = mean + 3 * std
            if vmax - vmin > 1e-6:
                logit_norm = np.clip((logit_map - vmin) / (vmax - vmin), 0, 1)
            else:
                logit_norm = np.clip(logit_map, 0, 1)
            logit_uint8 = (logit_norm * 255).astype(np.uint8)
            gray_name = f"{prefix}_{iter_str}_{m}_score{score:.3f}_logits_gray{extra_suffix}.png"
            cv2.imwrite(os.path.join(seg_dir_gray, gray_name), logit_uint8)

        # ---- 生成掩码 ----
        if logits_otsu:
            # Otsu 阈值化概率图
            img_uint8 = (prob_map * 255).astype(np.uint8)
            ret, otsu_binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold_prob = ret / 255.0
            mask_binary = otsu_binary
            base_name = f"{prefix}_{iter_str}_{m}_score{score:.3f}_logits_otsu_thr{threshold_prob:.3f}{extra_suffix}"
        else:
            # 直接使用 SAM 返回的二值掩码
            mask_raw = masks[idx]
            if mask_raw.dtype != np.uint8:
                mask_raw = (mask_raw > 0).astype(np.uint8) * 255
            else:
                mask_raw = mask_raw * 255  # 若已是 0/1
            if mask_raw.shape[0] != H_orig or mask_raw.shape[1] != W_orig:
                mask_raw = cv2.resize(mask_raw, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
            mask_binary = mask_raw
            threshold_prob = None
            base_name = f"{prefix}_{iter_str}_{m}_score{score:.3f}_mask{extra_suffix}"

        # 保存为三通道白色掩码
        mask_save = np.zeros((H_orig, W_orig, 3), dtype=np.uint8)
        mask_save[mask_binary > 0] = [255, 255, 255]

        mask_name = base_name + ".png"
        cv2.imwrite(os.path.join(seg_dir_thr, mask_name), mask_save)

        # ---- 带提示的叠加图 ----
        mask_with_prompts = mask_save.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask_with_prompts, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for idx_p, point in enumerate(points):
            x, y = map(int, point)
            label = labels[idx_p] if idx_p < len(labels) else 1
            color = (0, 0, 255) if label == 1 else (0, 255, 0)
            cv2.circle(mask_with_prompts, (x, y), 5, color, -1)

        prompt_name = base_name + "_withprompts.jpg"
        cv2.imwrite(os.path.join(prompt_dir, prompt_name), mask_with_prompts)

        saved_masks.append({
            'image': mask_save,
            'filename': mask_name,
            'score': score,
            'threshold': threshold_prob,
            'idx': idx,
            'base_name': base_name
        })
        m += 1

        if logits_otsu:
            print(f"✅ 已保存掩码 {idx}: {mask_name} (Otsu阈值={threshold_prob:.3f})")
        else:
            print(f"✅ 已保存掩码 {idx}: {mask_name} (直接二值掩码)")

    # 两两融合（仅当 logits_otsu=True 且 save_pairwise_multiplication=True 且恰好有3个掩码时）
    if logits_otsu and save_pairwise_multiplication and masks.shape[0] == 3 and rank_indices is None:
        print("🔄 生成两两相乘融合结果 ...")
        # 注意：此时 masks 是 logits
        prob_list = [1.0 / (1.0 + np.exp(-masks[k])) for k in range(3)]
        pairs = [(0, 1), (0, 2), (1, 2)]
        for (i, j) in pairs:
            fused = prob_list[i] * prob_list[j]
            fused_norm = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
            fused_uint8 = (fused_norm * 255).astype(np.uint8)
            ret, otsu_bin = cv2.threshold(fused_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = ret / 255.0
            fused_save = np.zeros((H_orig, W_orig, 3), dtype=np.uint8)
            fused_save[otsu_bin > 0] = [255, 255, 255]
            score_i, score_j = scores[i], scores[j]
            pair_name = f"{prefix}_{iter_str}_pair_{i}_{j}_score{score_i:.3f}_{score_j:.3f}_logits_otsu_thr{thr:.3f}.png"
            cv2.imwrite(os.path.join(seg_dir_thr, pair_name), fused_save)
            # 保存提示图
            mask_with_prompts = fused_save.copy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(mask_with_prompts, (x1, y1), (x2, y2), (0, 255, 255), 2)
            for idx_p, point in enumerate(points):
                x, y = map(int, point)
                label = labels[idx_p] if idx_p < len(labels) else 1
                color = (0, 0, 255) if label == 1 else (0, 255, 0)
                cv2.circle(mask_with_prompts, (x, y), 5, color, -1)
            prompt_pair = pair_name.replace(".png", "_withprompts.jpg")
            cv2.imwrite(os.path.join(prompt_dir, prompt_pair), mask_with_prompts)
            print(f"✅ 融合 {i}×{j}: {pair_name} (Otsu阈值={thr:.3f})")

    return saved_masks


def predict_mask(iteration, pathroot, imagename, image, points, labels, boxes, rough_mask_path, prefix, path_seg,
                 use_logits_otsu=False, skip_nomaskin=False):
    """
    使用点/框/粗糙前景进行预测，并保存多个掩码。
    参数 skip_nomaskin: 若为 True，则跳过无粗糙前景的基线预测分支。
    """
    # 构建有效的点/框张量（如果非空）
    input_points = torch.tensor(points) if points else None
    input_labels = torch.tensor(labels) if labels else None
    input_boxes = torch.tensor(boxes) if boxes else None

    # 加载粗糙前景掩码
    mask_input = None
    mask_extra_suffix = ""
    if rough_mask_path is not None and os.path.exists(rough_mask_path):
        mask_input = load_rough_mask(rough_mask_path)
        if mask_input is None:
            print(f"警告: 无法读取粗糙前景 {rough_mask_path}，将不使用 mask_input。")
        else:
            # 提取 npy 文件夹名（不含扩展名）作为后缀

            mask_folder = os.path.basename(os.path.dirname(rough_mask_path))
            mask_extra_suffix = f"_{mask_folder}"
            #
            # mask_stem = os.path.splitext(os.path.basename(rough_mask_path))[0]
            # mask_extra_suffix = f"_{mask_stem}"  # ← 新增
    # 检查是否有任何有效提示（点/框/掩码）
    has_pointbox = (input_points is not None and len(input_points) > 0) or (input_boxes is not None and len(input_boxes) > 0)
    if not has_pointbox and mask_input is None:
        print("错误：没有任何有效提示（点/框或掩码），无法预测。")
        return

    predictor.set_image(image)

    # ---------- 辅助函数：构建预测参数 ----------
    def build_predict_kwargs(mask_input=None, return_logits=False):
        kwargs = {'multimask_output': True, 'return_logits': return_logits}
        if input_points is not None and len(input_points) > 0:
            kwargs['point_coords'] = input_points
            kwargs['point_labels'] = input_labels
        if input_boxes is not None and len(input_boxes) > 0:
            kwargs['box'] = input_boxes
        if mask_input is not None:
            kwargs['mask_input'] = mask_input
        return kwargs

    # ---------- 第一轮：使用粗糙前景（如果提供） ----------
    if mask_input is not None:
        # 获取用于可视化的原图 logits（如果需要）
        logits_for_vis = None
        if not use_logits_otsu:
            kwargs_vis = build_predict_kwargs(mask_input, return_logits=True)
            masks_vis, _, _ = predictor.predict(**kwargs_vis)
            logits_for_vis = masks_vis  # 原图 logits

        # 获取二值掩码（或 logits，取决于 use_logits_otsu）
        kwargs_pred = build_predict_kwargs(mask_input, return_logits=use_logits_otsu)
        masks2, scores2, logits2 = predictor.predict(**kwargs_pred)

        save_sam_predictions_otsu(
            masks=masks2, scores=scores2, logits=logits2, image=image,
            boxes=boxes, points=points, labels=labels,
            save_root=path_seg, prefix=prefix, mode="maskin", iteration=1,
            save_heatmap=True, save_logits_gray=True,
            rank_indices=[1,2,3], save_pairwise_multiplication=False,
            logits_otsu=use_logits_otsu, logits_for_vis=logits_for_vis,
            extra_suffix=mask_extra_suffix,
        )
        avg_score = np.mean(scores2)
        closest_idx = np.argmin(np.abs(scores2 - avg_score))
        mask_input4 = logits2[closest_idx, :, :]  # 低分辨率 logits

        for it in range(2, iteration):
            logits_for_vis = None
            if not use_logits_otsu:
                kwargs_vis = build_predict_kwargs(mask_input4[None, :, :], return_logits=True)
                masks_vis, _, _ = predictor.predict(**kwargs_vis)
                logits_for_vis = masks_vis

            kwargs_pred = build_predict_kwargs(mask_input4[None, :, :], return_logits=use_logits_otsu)
            masks, scores, logits = predictor.predict(**kwargs_pred)
            save_sam_predictions_otsu(
                masks=masks, scores=scores, logits=logits, image=image,
                boxes=boxes, points=points, labels=labels,
                save_root=path_seg, prefix=prefix, mode="maskin", iteration=it,
                save_heatmap=True, save_logits_gray=True,
                rank_indices=[1,2,3], save_pairwise_multiplication=False,
                logits_otsu=use_logits_otsu, logits_for_vis=logits_for_vis,
                extra_suffix=mask_extra_suffix,
            )
            avg_score = np.mean(scores)
            closest_idx = np.argmin(np.abs(scores - avg_score))
            mask_input4 = logits[closest_idx, :, :]

    # ---------- 无粗糙前景的基线（根据 skip_nomaskin 决定是否执行） ----------
    if not skip_nomaskin and has_pointbox:
        # 第一次预测不保存（仅获取低分辨率 logits 用于迭代）
        kwargs_first = build_predict_kwargs(None, return_logits=use_logits_otsu)
        masks1, scores1, logits1 = predictor.predict(**kwargs_first)
        mask_input3 = logits1[np.argmax(scores1), :, :]

        for it in range(2, iteration):
            logits_for_vis = None
            if not use_logits_otsu:
                kwargs_vis = build_predict_kwargs(mask_input3[None, :, :], return_logits=True)
                masks_vis, _, _ = predictor.predict(**kwargs_vis)
                logits_for_vis = masks_vis

            kwargs_pred = build_predict_kwargs(mask_input3[None, :, :], return_logits=use_logits_otsu)
            masks, scores, logits = predictor.predict(**kwargs_pred)
            save_sam_predictions_otsu(
                masks=masks, scores=scores, logits=logits, image=image,
                boxes=boxes, points=points, labels=labels,
                save_root=path_seg, prefix=prefix, mode="nomaskin", iteration=it,
                save_heatmap=True, save_logits_gray=True,
                rank_indices=[1,2,3], save_pairwise_multiplication=False,
                logits_otsu=use_logits_otsu, logits_for_vis=logits_for_vis
            )
            mask_input3 = logits[np.argmax(scores), :, :]

# ==================== 筛选最佳掩模的函数 ====================
def filter_best_masks(seg_root, ref_folder, prefix_len, metric_choice, decimals=1, save_excel=True, saveall=False):
    """
    根据参考图像和指定指标，从 seg_root 下的各个子文件夹中筛选每个前缀的最佳掩模，
    删除其他所有文件，只保留最佳掩模及其对应的灰度图、热力图和提示图。
    如果 saveall=True，则保留所有掩码，不进行筛选和删除。
    同时将所有前缀的最佳指标汇总保存为 best_metrics.xlsx 到 seg_masks_thr 目录（仅当 saveall=False 时）。
    """
    if saveall:
        print("saveall=True，保留所有掩码，不进行筛选。")
        return

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    b_files = [f for f in os.listdir(ref_folder) if f.lower().endswith(valid_exts)]
    b_prefix_map = {}
    for f in b_files:
        stem = os.path.splitext(f)[0]
        prefix = stem[:prefix_len]
        full_path = os.path.join(ref_folder, f)
        if prefix not in b_prefix_map:
            b_prefix_map[prefix] = full_path
        else:
            print(f"警告：参考文件夹中前缀 '{prefix}' 存在多个文件，将使用第一个：{b_prefix_map[prefix]}")

    subdirs = ['seg_masks_thr', 'seg_masks_gray', 'seg_masks_heatmap', 'prompt_masks']
    thr_dir = os.path.join(seg_root, 'seg_masks_thr')
    if not os.path.exists(thr_dir):
        print("seg_masks_thr 文件夹不存在，跳过筛选。")
        return

    all_thr_files = [f for f in os.listdir(thr_dir) if f.lower().endswith(valid_exts)]
    prefix_groups = {}
    for f in all_thr_files:
        stem = os.path.splitext(f)[0]
        prefix = stem[:prefix_len]
        prefix_groups.setdefault(prefix, []).append(f)

    metric_names = [
        'PixelAccuracy', 'precision_positive', 'precision_negative', 'MPA',
        'IOU_positive', 'MIOU', 'Dice_fg', 'Dice_bg', 'mDice',
        'Recall_fg', 'Recall_bg', 'mRecall',
        'F1_fg', 'F1_bg', 'mF1'
    ]
    metric_dict_key = metric_choice + ' (%)'

    best_records = []

    for prefix, files in prefix_groups.items():
        if prefix not in b_prefix_map:
            print(f"前缀 '{prefix}' 在参考文件夹中无对应图像，跳过。")
            continue
        ref_path = b_prefix_map[prefix]
        ref_bin = prepare_binary_image(ref_path)
        if ref_bin is None:
            print(f"无法读取参考图像 {ref_path}，跳过前缀 '{prefix}'。")
            continue
        ref_shape = ref_bin.shape[:2]

        best_score = -1
        best_base = None
        best_metrics = None

        for f in files:
            thr_path = os.path.join(thr_dir, f)
            pred_bin = prepare_binary_image(thr_path)
            if pred_bin is None:
                continue
            if pred_bin.shape[:2] != ref_shape:
                pred_bin = resize_to_match(pred_bin, ref_shape)
            metrics = binary_pixel_accuracy(pred_bin, ref_bin)
            metrics_percent = [m * 100 for m in metrics]
            metric_vals = {
                'PixelAccuracy (%)': metrics_percent[0],
                'precision_positive (%)': metrics_percent[1],
                'precision_negative (%)': metrics_percent[2],
                'MPA (%)': metrics_percent[3],
                'IOU_positive (%)': metrics_percent[4],
                'MIOU (%)': metrics_percent[5],
                'Dice_fg (%)': metrics_percent[6],
                'Dice_bg (%)': metrics_percent[7],
                'mDice (%)': metrics_percent[8],
                'Recall_fg (%)': metrics_percent[9],
                'Recall_bg (%)': metrics_percent[10],
                'mRecall (%)': metrics_percent[11],
                'F1_fg (%)': metrics_percent[12],
                'F1_bg (%)': metrics_percent[13],
                'mF1 (%)': metrics_percent[14]
            }
            score = metric_vals.get(metric_dict_key, 0)
            if score > best_score:
                best_score = score
                best_base = os.path.splitext(f)[0]
                best_metrics = metric_vals

        if best_base is None:
            print(f"前缀 '{prefix}' 没有有效的掩模，跳过。")
            continue

        print(f"前缀 '{prefix}' 最佳掩模: {best_base}, {metric_dict_key}={best_score:.{decimals}f}%")

        # 保留最佳，删除其他
        for f in files:
            if f != best_base + '.png':
                os.remove(os.path.join(thr_dir, f))
                print(f"删除 seg_masks_thr/{f}")

        for sub in ['seg_masks_gray', 'seg_masks_heatmap', 'prompt_masks']:
            sub_path = os.path.join(seg_root, sub)
            if not os.path.exists(sub_path):
                continue
            candidates = [f for f in os.listdir(sub_path) if f.startswith(prefix)]
            if not candidates:
                continue
            best_match = None
            max_common_len = -1
            for fname in candidates:
                base_f = os.path.splitext(fname)[0]
                common = 0
                for a, b in zip(best_base, base_f):
                    if a == b:
                        common += 1
                    else:
                        break
                if common > max_common_len:
                    max_common_len = common
                    best_match = fname
            if best_match is not None:
                for fname in candidates:
                    if fname != best_match:
                        os.remove(os.path.join(sub_path, fname))
                        print(f"删除 {sub}/{fname}")
            else:
                print(f"警告：在 {sub} 中没有找到与 {best_base} 匹配的文件。")

        logic_dir = os.path.join(seg_root, 'logic')
        if os.path.exists(logic_dir):
            shutil.rmtree(logic_dir)
            print(f"删除 logic 文件夹")

        record = {
            'Prefix': prefix,
            'BestImage': best_base + '.png',
            'BestScore (%)': best_score,
            **best_metrics
        }
        best_records.append(record)

    if save_excel and best_records:
        df_best = pd.DataFrame(best_records)
        df_best = df_best.sort_values('Prefix').reset_index(drop=True)

        numeric_cols = [col for col in df_best.columns if col.endswith('(%)') or col == 'BestScore (%)']
        df_best[numeric_cols] = df_best[numeric_cols].round(decimals)

        excel_path = os.path.join(thr_dir, "best_metrics.xlsx")
        try:
            import xlsxwriter
            engine = 'xlsxwriter'
        except ImportError:
            engine = 'openpyxl'
            print("警告：未安装 xlsxwriter，将使用 openpyxl，数字格式可能无法固定。")

        with pd.ExcelWriter(excel_path, engine=engine) as writer:
            df_best.to_excel(writer, sheet_name='BestPerPrefix', index=False)
            if engine == 'xlsxwriter':
                workbook = writer.book
                fmt = workbook.add_format({'num_format': f'0.{ "0" * decimals }'})
                worksheet = writer.sheets['BestPerPrefix']
                worksheet.set_column(0, len(df_best.columns) - 1, None, fmt)

        print(f"📊 最佳指标汇总已保存至：{excel_path}")
    elif not best_records:
        print("没有最佳记录，未生成 Excel 文件。")


# ==================== 主程序 ====================
if __name__ == "__main__":
    start_time = time.time()
    root = Tk()
    root.withdraw()

    # ---------- 检查用户参数和路径 ----------
    # 如果 PointBoxprompt=True 但 txt_folder 未提供或不存在，则报错
    if PointBoxprompt:
        if txt_folder is None:
            print("错误：PointBoxprompt 为 True 但 txt_folder 未指定。")
            sys.exit()
        if not os.path.exists(txt_folder):
            print(f"错误：txt_folder '{txt_folder}' 不存在。")
            sys.exit()
    else:
        print("PointBoxprompt=False，跳过加载 txt 文件。")

    if Maskprompt:
        if rough_mask_folder is None:
            print("错误：Maskprompt 为 True 但 rough_mask_folder 未指定。")
            sys.exit()
        if not os.path.exists(rough_mask_folder):
            print(f"错误：rough_mask_folder '{rough_mask_folder}' 不存在。")
            sys.exit()
    else:
        print("Maskprompt=False，跳过加载粗糙掩码。")

    # ---------- 准备图像列表 ----------
    if not os.path.exists(image_folder):
        print(f"错误：图像文件夹 '{image_folder}' 不存在。")
        sys.exit()
    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    image_files.sort()
    if not image_files:
        print("图像文件夹中没有支持的图像文件。")
        sys.exit()

    # ---------- 根据 PointBoxprompt 读取 txt 文件 ----------
    txt_files = []
    if PointBoxprompt:
        txt_files = [f for f in os.listdir(txt_folder) if f.lower().endswith('.txt')]
        txt_files.sort()
        if not txt_files:
            print("警告：txt 文件夹中没有 .txt 文件，但 PointBoxprompt=True，将无法使用点/框提示。")

    # ---------- 根据 Maskprompt 读取粗糙掩码 ----------
    rough_dict = {}
    if Maskprompt:
        rough_files = [f for f in os.listdir(rough_mask_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.npy'))]
        for rf in rough_files:
            pref = os.path.splitext(rf)[0][:prefix_len]
            rough_dict.setdefault(pref, []).append(rf)
        print(f"已加载 {len(rough_files)} 个粗糙掩码文件。")

    # ---------- 其他固定参数 ----------
    # prefix_len = 2
    prefix_len = 2
    selected_metric = 'MIOU'
    decimals = 1
    iterationvalue = 4          # 迭代轮数 = iterationvalue - 1
    use_logits_otsu = False     # False: 直接使用SAM二值掩码，同时可视化logits; True: 使用Otsu阈值
    saveallmasks = True        # True: 保留所有掩码; False: 筛选最佳掩码

    # ---------- 选择 SAM 模型 ----------
    sam_checkpoint = filedialog.askopenfilename(
        title="选择 SAM 模型权重文件",
        # initialdir=r"A:\B_diskexpand\Code\GrMflakes\data\work_dir\Checkpoint\OM-Grflakes",
        initialdir=r"A:\B_diskexpand\Code\GrMflakes\data\work_dir\Checkpoint\AFM",
        filetypes=[("PyTorch model", "*.pth")]
    )
    if not sam_checkpoint:
        print("未选择模型，程序退出。")
        sys.exit()

    # ---------- 初始化 SAM ----------
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    directory_name = 'GrM-AP'
    base_output_dir = os.path.join(os.path.dirname(image_folder), directory_name)
    os.makedirs(base_output_dir, exist_ok=True)

    path_seg = None

    # ---------- 主循环 ----------
    for file_name in image_files:
        print(f"\n处理图像: {file_name}")
        base_name = os.path.splitext(file_name)[0]
        prefix = base_name[:prefix_len]

        # ---------- 处理点/框提示 ----------
        boxes, points, labels = [], [], []
        if PointBoxprompt:
            matched_txts = [f for f in txt_files if f.startswith(prefix) and f.endswith('.txt')]
            if not matched_txts:
                print(f"  未找到与 '{file_name}' 匹配的 txt 文件，跳过该图像。")
                continue
            selected_txt = matched_txts[0]
            txt_path = os.path.join(txt_folder, selected_txt)
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        if "box prompts" in line:
                            boxes_str = line.split("box prompts:")[1].strip()
                            boxes = eval(boxes_str)
                        elif "point prompts" in line:
                            points_str = line.split("point prompts:")[1].strip()
                            points = eval(points_str)
                        elif "labels" in line:
                            labels_str = line.split("labels:")[1].strip()
                            labels = eval(labels_str)
            except Exception as e:
                print(f"  读取 txt 文件失败: {e}，跳过。")
                continue
        else:
            # PointBoxprompt=False，points/boxes/labels 保持为空
            pass

        # ---------- 处理粗糙掩码 ----------
        rough_path = None
        if Maskprompt:
            rough_matches = rough_dict.get(prefix, [])
            if rough_matches:
                rough_path = os.path.join(rough_mask_folder, rough_matches[0])
                print(f"  使用粗糙前景: {rough_matches[0]}")
            else:
                print(f"  未找到前缀 '{prefix}' 的粗糙前景，Maskprompt 降级。")

        # ---------- 判断是否有有效提示 ----------
        has_pointbox = PointBoxprompt and (len(points) > 0 or len(boxes) > 0)
        has_mask = Maskprompt and rough_path is not None
        if not has_pointbox and not has_mask:
            print("  没有有效提示（点/框或粗糙掩码），跳过。")
            continue

        # 决定是否跳过无掩码分支 (skip_nomaskin)
        if Maskprompt and not PointBoxprompt:
            skip_nomaskin = True
        else:
            skip_nomaskin = False

        # 如果 Maskprompt=True 但没有粗糙掩码，且 PointBoxprompt=False，则无法预测
        if Maskprompt and rough_path is None and not PointBoxprompt:
            print("  粗糙掩码缺失且未启用点/框，无法预测，跳过。")
            continue

        # 读取图像
        image_path = os.path.join(image_folder, file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"  无法读取图像，跳过。")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 准备输出目录
        model_name = os.path.splitext(os.path.basename(sam_checkpoint))[0]
        output_subdir = os.path.join(base_output_dir, model_name, "promptaddimage")
        output_maskdir = os.path.join(base_output_dir, model_name, "masks")
        print("输出结果保存在：", base_output_dir)
        os.makedirs(output_subdir, exist_ok=True)
        os.makedirs(output_maskdir, exist_ok=True)
        path_seg = output_maskdir

        # 调用预测函数
        predict_mask(
            iteration=iterationvalue,
            pathroot=image_folder,
            imagename=file_name,
            image=image_rgb,
            points=points,
            labels=labels,
            boxes=boxes,
            rough_mask_path=rough_path,
            prefix=prefix,
            path_seg=path_seg,
            use_logits_otsu=use_logits_otsu,
            skip_nomaskin=skip_nomaskin
        )

        # 保存提示记录（如果有 txt）
        if PointBoxprompt and matched_txts:
            record_path = os.path.join(output_subdir, f"{os.path.splitext(selected_txt)[0]}_prompts.txt")
            with open(record_path, 'w') as f:
                f.write(f"point prompts: {points}\nlabels: {labels}\nbox prompts: {boxes}\n")

        print(f"  单张耗时: {time.time() - start_time:.2f} 秒")

    # ---------- 筛选最佳掩模 ----------
    if path_seg is not None:
        print("\n开始筛选每个前缀的最佳掩模...")
        filter_best_masks(path_seg, ref_folder, prefix_len, selected_metric,
                          decimals=decimals, save_excel=True, saveall=saveallmasks)
    else:
        print("未生成任何掩模，无法筛选。")

    print("\n所有操作完成！")




