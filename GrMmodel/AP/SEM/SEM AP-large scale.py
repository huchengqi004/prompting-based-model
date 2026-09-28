#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立图像处理流水线：从原始图像直接生成 AP 提示（红色点、黄色框、绿色点）
所有中间结果保存在输出文件夹下的 pipeline_temp 子目录中，不会自动删除。
新增：逻辑操作后对 forecoarse 进行小连通区域过滤，并强制确保 coarse_filtered 非空。
"""

import os
import sys
import cv2
import numpy as np
import glob
import shutil
import time
import tkinter as tk
from tkinter import filedialog


# ==================== 纹理提取核心算法 ====================
def auto_canny_threshold_texture(gray_img, high_percentile=0.9, low_percentile=0.5):
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mag = np.uint8(np.clip(grad_mag, 0, 255))
    non_zero = grad_mag[grad_mag > 0]
    if len(non_zero) == 0:
        return 0, 0, grad_mag
    sorted_mag = np.sort(non_zero)
    n = len(sorted_mag)
    high_idx = min(int(high_percentile * n), n - 1)
    low_idx = min(int(low_percentile * n), n - 1)
    return int(sorted_mag[low_idx]), int(sorted_mag[high_idx]), grad_mag


def gradient_info(img_gray):
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    angle = np.arctan2(grad_y, grad_x)
    angle[mag < 1e-6] = 0
    return mag, angle


def symmetry_classification_length_filtered(edges, angle, min_length,
                                            half_width, threshold_deg):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    long_edges = np.zeros_like(edges, dtype=np.uint8)
    for cnt in contours:
        if cv2.arcLength(cnt, False) >= min_length:
            cv2.drawContours(long_edges, [cnt], -1, 255, 1)
    ys, xs = np.nonzero(long_edges)
    label = np.zeros_like(edges, dtype=bool)
    angle_deg = np.degrees(angle)
    h, w = edges.shape
    for y, x in zip(ys, xs):
        theta = angle[y, x]
        if np.abs(theta) < 1e-6:
            continue
        dx, dy = np.cos(theta), np.sin(theta)
        pos_best_diff, neg_best_diff = 0, 0
        pos_best_dist, neg_best_dist = None, None
        for d in range(-half_width, half_width + 1):
            if d == 0:
                continue
            xsamp = int(round(x + d * dx))
            ysamp = int(round(y + d * dy))
            if 0 <= xsamp < w and 0 <= ysamp < h:
                samp_ang = angle_deg[ysamp, xsamp]
                diff = abs(samp_ang - angle_deg[y, x]) % 360
                diff = min(diff, 360 - diff)
                if d > 0 and diff > pos_best_diff:
                    pos_best_diff = diff
                    pos_best_dist = d
                elif d < 0 and diff > neg_best_diff:
                    neg_best_diff = diff
                    neg_best_dist = d
        if (pos_best_dist is not None and neg_best_dist is not None and
                pos_best_diff > threshold_deg and neg_best_diff > threshold_deg and
                abs(pos_best_dist + neg_best_dist) <= max(1, half_width // 2)):
            label[y, x] = True
    return label


def filter_texture_noise_density(tex_mask, window_size, density_threshold):
    mask_float = tex_mask.astype(np.float32)
    kernel = np.ones((window_size, window_size), np.float32)
    density = cv2.filter2D(mask_float, -1, kernel) / (window_size ** 2)
    filtered = (density > density_threshold) & tex_mask.astype(bool)
    return filtered


def extract_texture_connected(gray, params):
    low_p, high_p, _ = auto_canny_threshold_texture(gray, params['percent_high'], params['percent_low'])
    edges = cv2.Canny(gray, low_p, high_p)
    mag, angle = gradient_info(gray)

    tex1 = symmetry_classification_length_filtered(
        edges, angle,
        min_length=params['min_contour_length'],
        half_width=params['symmetry_half_width'],
        threshold_deg=params['symmetry_threshold_deg']
    )

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    new_tex = np.zeros_like(edges, dtype=np.uint8)
    for cnt in contours:
        if cv2.arcLength(cnt, False) < params['min_contour_length']:
            continue
        pts = cnt[:, 0, :]
        total = len(pts)
        if total == 0:
            continue
        is_tex = np.array([tex1[y, x] for x, y in pts], dtype=bool)
        tex_indices = np.where(is_tex)[0]
        if len(tex_indices) == 0:
            continue
        ratio = len(tex_indices) / total
        if ratio >= params['contour_tex_ratio']:
            for idx in tex_indices:
                x, y = pts[idx]
                new_tex[y, x] = 255

    if params['tex_density1_enable']:
        tex_final = filter_texture_noise_density(new_tex, params['tex_density_win1'], params['tex_density_thr1'])
    else:
        tex_final = (new_tex > 0)

    if params['tex_density2_enable']:
        tex_img_temp = (tex_final.astype(np.uint8)) * 255
        tex_final = filter_texture_noise_density(tex_img_temp, params['tex_density_win2'], params['tex_density_thr2'])

    tex_img = (tex_final.astype(np.uint8)) * 255

    connected_raw = tex_img.copy()
    angle_deg = np.degrees(angle)
    h, w = edges.shape
    half_width = params['symmetry_half_width']
    threshold_deg = params['symmetry_threshold_deg']

    tex_points = np.column_stack(np.where(tex_img > 0))
    for y, x in tex_points:
        theta = angle[y, x]
        mag_val = mag[y, x]
        if mag_val < 1e-6:
            continue
        dx, dy = np.cos(theta), np.sin(theta)
        pos_best_diff, neg_best_diff = 0, 0
        pos_best_dist, neg_best_dist = None, None
        for d in range(-half_width, half_width + 1):
            if d == 0:
                continue
            xsamp = int(round(x + d * dx))
            ysamp = int(round(y + d * dy))
            if 0 <= xsamp < w and 0 <= ysamp < h:
                samp_ang = angle_deg[ysamp, xsamp]
                diff = abs(samp_ang - angle_deg[y, x]) % 360
                diff = min(diff, 360 - diff)
                if d > 0 and diff > pos_best_diff:
                    pos_best_diff = diff
                    pos_best_dist = d
                elif d < 0 and diff > neg_best_diff:
                    neg_best_diff = diff
                    neg_best_dist = d
        if (pos_best_dist is not None and neg_best_dist is not None and
                pos_best_diff > threshold_deg and neg_best_diff > threshold_deg and
                abs(pos_best_dist + neg_best_dist) <= max(1, half_width // 2)):
            x_pos = int(round(x + pos_best_dist * dx))
            y_pos = int(round(y + pos_best_dist * dy))
            x_neg = int(round(x + neg_best_dist * dx))
            y_neg = int(round(y + neg_best_dist * dy))
            cv2.line(connected_raw, (x_neg, y_neg), (x, y), 255, 1, lineType=cv2.LINE_AA)
            cv2.line(connected_raw, (x, y), (x_pos, y_pos), 255, 1, lineType=cv2.LINE_AA)

    morph_shape = params.get('morph_shape', 'rect')
    if morph_shape is None or morph_shape == 'None':
        connected_dilated = connected_raw
    else:
        shape_map = {'rect': cv2.MORPH_RECT, 'ellipse': cv2.MORPH_ELLIPSE, 'cross': cv2.MORPH_CROSS}
        shape = shape_map.get(morph_shape, cv2.MORPH_RECT)
        size = params.get('morph_size', 2)
        iterations = params.get('morph_iter', 1)
        kernel_dilate = cv2.getStructuringElement(shape, (size, size))
        connected_dilated = cv2.dilate(connected_raw, kernel_dilate, iterations=iterations)

    return tex_img, connected_dilated


# ==================== 轮廓提取核心算法 ====================
def auto_canny_threshold_contour(gray_img, high_percentile=0.9, low_percentile=0.5):
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mag = np.uint8(np.clip(grad_mag, 0, 255))
    non_zero = grad_mag[grad_mag > 0]
    if len(non_zero) == 0:
        return 0, 0, grad_mag
    sorted_mag = np.sort(non_zero)
    n = len(sorted_mag)
    high_idx = min(int(high_percentile * n), n - 1)
    low_idx = min(int(low_percentile * n), n - 1)
    return int(sorted_mag[low_idx]), int(sorted_mag[high_idx]), grad_mag


def filter_long_contours(edges, x_multiplier=0.0):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return edges
    lengths = np.array([cv2.arcLength(cnt, False) for cnt in contours])
    q1 = np.percentile(lengths, 25)
    q3 = np.percentile(lengths, 75)
    iqr = q3 - q1
    threshold = q1 + x_multiplier * iqr
    filtered = np.zeros_like(edges)
    for cnt in contours:
        if cv2.arcLength(cnt, False) > threshold:
            cv2.drawContours(filtered, [cnt], -1, 255, 1)
    return filtered


def apply_morphology_to_edge(edge_img, morph_shape, morph_size, morph_iter):
    if morph_shape is None or morph_shape == 'None':
        return edge_img
    shape_map = {'rect': cv2.MORPH_RECT, 'ellipse': cv2.MORPH_ELLIPSE, 'cross': cv2.MORPH_CROSS}
    shape = shape_map.get(morph_shape, cv2.MORPH_RECT)
    kernel = cv2.getStructuringElement(shape, (morph_size, morph_size))
    return cv2.dilate(edge_img, kernel, iterations=morph_iter)


def process_contour_folder(input_dir, output_after_dir, output_morph_dir,
                           percent_high, percent_low, x_multiplier,
                           morph_shape, morph_size, morph_iter,
                           output_before_dir=None, log_callback=None):
    if output_before_dir:
        os.makedirs(output_before_dir, exist_ok=True)
    os.makedirs(output_after_dir, exist_ok=True)
    os.makedirs(output_morph_dir, exist_ok=True)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        return

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        low_p, high_p, _ = auto_canny_threshold_contour(gray, percent_high, percent_low)
        edges_p = cv2.Canny(gray, low_p, high_p)

        if output_before_dir:
            before_name = f"{base}_P_l{low_p}_h{high_p}.png"
            cv2.imwrite(os.path.join(output_before_dir, before_name), edges_p)

        filtered = filter_long_contours(edges_p, x_multiplier=x_multiplier)
        after_name = f"{base}_P_l{low_p}_h{high_p}_longKeep_x{x_multiplier}.png"
        cv2.imwrite(os.path.join(output_after_dir, after_name), filtered)

        morph_edges = apply_morphology_to_edge(filtered, morph_shape, morph_size, morph_iter)
        morph_name = f"{base}_P_l{low_p}_h{high_p}_morph_{morph_shape}_{morph_size}_{morph_iter}.png"
        cv2.imwrite(os.path.join(output_morph_dir, morph_name), morph_edges)


# ==================== 衬度提取核心算法 ====================
def process_contrast_image_single(gray, ratio, invert, morph_shape, morph_size, morph_iter):
    otsu_th, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    new_th = np.clip(ratio * otsu_th, 0, 255)
    _, binary = cv2.threshold(gray, new_th, 255, cv2.THRESH_BINARY)
    if invert:
        binary = cv2.bitwise_not(binary)
    if morph_shape is None or morph_shape == 'None':
        morph_result = binary
    else:
        shape_map = {'rect': cv2.MORPH_RECT, 'ellipse': cv2.MORPH_ELLIPSE, 'cross': cv2.MORPH_CROSS}
        shape = shape_map.get(morph_shape, cv2.MORPH_RECT)
        kernel = cv2.getStructuringElement(shape, (morph_size, morph_size))
        morph_result = cv2.dilate(binary, kernel, iterations=morph_iter)
    return binary, morph_result


def process_contrast_images(input_dir, fore_out_dir, back_out_dir,
                            ratio_fore, ratio_back, invert_fore=True, invert_back=True, log_callback=None):
    os.makedirs(fore_out_dir, exist_ok=True)
    os.makedirs(back_out_dir, exist_ok=True)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        return

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        otsu_th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        new_th = np.clip(ratio_fore * otsu_th, 0, 255)
        _, binary = cv2.threshold(img, new_th, 255, cv2.THRESH_BINARY)
        if invert_fore:
            binary = cv2.bitwise_not(binary)
        cv2.imwrite(os.path.join(fore_out_dir, base + ".png"), binary)

        new_th = np.clip(ratio_back * otsu_th, 0, 255)
        _, binary = cv2.threshold(img, new_th, 255, cv2.THRESH_BINARY)
        if invert_back:
            binary = cv2.bitwise_not(binary)
        cv2.imwrite(os.path.join(back_out_dir, base + ".png"), binary)


# ==================== 纹理提取批量处理 ====================
def process_texture_batch(input_dir, output_fore_conn_dir, output_back_conn_dir,
                          output_fore_tex_dir, output_back_tex_dir,
                          fg_params, bg_params, log_callback=None):
    os.makedirs(output_fore_conn_dir, exist_ok=True)
    os.makedirs(output_back_conn_dir, exist_ok=True)
    os.makedirs(output_fore_tex_dir, exist_ok=True)
    os.makedirs(output_back_tex_dir, exist_ok=True)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        return

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fg_tex, fg_conn = extract_texture_connected(gray, fg_params)
        bg_tex, bg_conn = extract_texture_connected(gray, bg_params)

        cv2.imwrite(os.path.join(output_fore_conn_dir, base + ".png"), fg_conn)
        cv2.imwrite(os.path.join(output_back_conn_dir, base + ".png"), bg_conn)
        cv2.imwrite(os.path.join(output_fore_tex_dir, base + ".png"), fg_tex)
        cv2.imwrite(os.path.join(output_back_tex_dir, base + ".png"), bg_tex)


# ==================== 逻辑操作 ====================
def logic_operation_folder(folderA, folderB, operation, output_dir, prefix_len=0, log_callback=None):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    filesA = [f for f in os.listdir(folderA) if f.lower().endswith(exts)]
    filesB = [f for f in os.listdir(folderB) if f.lower().endswith(exts)]
    if not filesA or not filesB:
        return 0, 0

    os.makedirs(output_dir, exist_ok=True)
    count = 0
    skipped = 0

    def compute_result(bin1, bin2, op):
        if op == "并集":
            return np.bitwise_or(bin1, bin2)
        elif op == "交集":
            return np.bitwise_and(bin1, bin2)
        elif op == "差集(A-B)":
            return np.bitwise_and(bin1, 255 - bin2)
        elif op == "差集(B-A)":
            return np.bitwise_and(bin2, 255 - bin1)
        else:
            raise ValueError("未知操作")

    def preprocess(img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return binary

    if prefix_len > 0:
        def get_prefix(fname):
            base, _ = os.path.splitext(fname)
            return base[:prefix_len]

        dictA = {}
        for f in filesA:
            pref = get_prefix(f)
            dictA.setdefault(pref, []).append(f)
        dictB = {}
        for f in filesB:
            pref = get_prefix(f)
            dictB.setdefault(pref, []).append(f)
        common_pref = set(dictA.keys()) & set(dictB.keys())
        if not common_pref:
            return 0, 0
        for pref in common_pref:
            listA = sorted(dictA[pref])
            listB = sorted(dictB[pref])
            for fnameA, fnameB in zip(listA, listB):
                imgA = cv2.imread(os.path.join(folderA, fnameA))
                imgB = cv2.imread(os.path.join(folderB, fnameB))
                if imgA is None or imgB is None:
                    skipped += 1
                    continue
                binA = preprocess(imgA)
                binB = preprocess(imgB)
                if binA.shape != binB.shape:
                    skipped += 1
                    continue
                result = compute_result(binA, binB, operation)
                out_path = os.path.join(output_dir, fnameA)
                cv2.imwrite(out_path, result)
                count += 1
    else:
        filesA_dict = {f.lower(): f for f in filesA}
        filesB_dict = {f.lower(): f for f in filesB}
        common = set(filesA_dict.keys()) & set(filesB_dict.keys())
        if not common:
            return 0, 0
        for key in common:
            fnameA = filesA_dict[key]
            fnameB = filesB_dict[key]
            imgA = cv2.imread(os.path.join(folderA, fnameA))
            imgB = cv2.imread(os.path.join(folderB, fnameB))
            if imgA is None or imgB is None:
                skipped += 1
                continue
            binA = preprocess(imgA)
            binB = preprocess(imgB)
            if binA.shape != binB.shape:
                skipped += 1
                continue
            result = compute_result(binA, binB, operation)
            out_path = os.path.join(output_dir, fnameA)
            cv2.imwrite(out_path, result)
            count += 1

    return count, skipped


# ==================== 低通滤波 ====================
def process_lowpass_filter(input_dir, output_dir, r, r_noise, log_callback=None):
    os.makedirs(output_dir, exist_ok=True)
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        return False

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        mask = np.zeros((rows, cols, 2), np.uint8)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_noise * r_noise
        mask[mask_area] = 1

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back_filter = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        out_name = f"{base}-r{r}_n{r_noise}_LPF.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), img_back_filter)

    return True

#
# # ==================== 过滤小连通区域（改进版，保留原图若结果全黑） ====================
# def filter_small_connected(input_dir, output_dir, x_val=1.5):
#     """
#     过滤二值图像中面积过小的连通区域（基于四分位数箱线图）。
#     若过滤后无前景，则保留原图。
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
#     image_paths = []
#     for ext in exts:
#         image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
#         image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
#
#     print(f"  filter_small_connected: 输入目录 {input_dir} 中找到 {len(image_paths)} 个图像文件")
#     if not image_paths:
#         print("  警告：输入文件夹无图像，跳过过滤。")
#         return
#
#     for img_path in image_paths:
#         fname = os.path.basename(img_path)
#         print(f"    处理: {fname}")
#         gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if gray is None:
#             print(f"      无法读取 {fname}，跳过")
#             continue
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
#         areas = [stats[label, cv2.CC_STAT_AREA] for label in range(1, num_labels)]
#         if not areas:
#             cv2.imwrite(os.path.join(output_dir, fname), binary)
#             print(f"      全黑，直接保存")
#             continue
#         areas = np.array(areas)
#         q1 = np.percentile(areas, 25)
#         q3 = np.percentile(areas, 75)
#         threshold = q1 + x_val * (q3 - q1)
#         if threshold < 0:
#             threshold = 0
#         print(f"      Q1={q1:.1f}, Q3={q3:.1f}, 阈值={threshold:.1f}")
#
#         result = np.zeros_like(binary)
#         for label in range(1, num_labels):
#             if stats[label, cv2.CC_STAT_AREA] >= threshold:
#                 result[labels == label] = 255
#
#         # 若结果全黑，则保留原图（避免输出为空）
#         if np.sum(result) == 0:
#             print(f"      过滤后为空，保留原图")
#             result = binary
#
#         out_path = os.path.join(output_dir, fname)
#         cv2.imwrite(out_path, result)
#         print(f"      保存至 {out_path}")
def filter_small_connected(input_dir, output_dir, x_val=1.5):
    """
    完全复刻用户独立过滤脚本的逻辑：
    1. 读取灰度图，Otsu 二值化
    2. 连通域分析，计算所有前景区域面积
    3. 计算箱线图阈值 Q1 + x_val*(Q3-Q1)
    4. 保留面积 >= 阈值的区域，其余置黑
    5. 若阈值<=0，则保留所有区域（不过滤）
    """
    os.makedirs(output_dir, exist_ok=True)
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # 收集所有图像文件
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, "*" + ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, "*" + ext.upper())))

    if not image_files:
        print(f"  filter_small_connected: 输入目录 {input_dir} 中没有找到图像文件")
        return

    print(f"  filter_small_connected: 找到 {len(image_files)} 个图像文件，开始处理...")

    for idx, filepath in enumerate(image_files, 1):
        filename = os.path.basename(filepath)
        print(f"    [{idx}/{len(image_files)}] 处理: {filename}")

        # 读取灰度图
        gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"      警告：无法读取 {filename}，跳过。")
            continue

        # Otsu 二值化（前景为白色255）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 连通区域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # 收集所有前景区域的面积（排除背景标签0）
        areas = []
        for label in range(1, num_labels):
            areas.append(stats[label, cv2.CC_STAT_AREA])

        if len(areas) == 0:
            # 无前景区域，直接保存全黑图像
            cv2.imwrite(os.path.join(output_dir, filename), binary)
            print(f"      无前景区域，直接保存全黑。")
            continue

        areas = np.array(areas)
        q1 = np.percentile(areas, 25)
        q3 = np.percentile(areas, 75)
        threshold = q1 + x_val * (q3 - q1)

        # 如果阈值 <= 0，则保留所有区域（不过滤）
        if threshold <= 0:
            print(f"      计算阈值 = {threshold:.1f} <= 0，保留所有区域。")
            threshold = 0

        # 创建结果图像，保留面积 >= threshold 的区域
        result = np.zeros_like(binary)
        kept = 0
        removed = 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= threshold:
                result[labels == label] = 255
                kept += 1
            else:
                removed += 1

        # 保存结果（可能全黑，这是正常的）
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, result)
        print(f"      完成：Q1={q1:.1f}, Q3={q3:.1f}, 阈值={threshold:.1f}, 保留{kept}个, 移除{removed}个")

    print(f"  filter_small_connected: 所有图像处理完毕，结果保存在 {output_dir}")

# ==================== AP Generate 核心算法 ====================
def filter_nested_boxes(boxes, area_ratio_threshold=0.8):
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    filtered_boxes = []
    for current_box in sorted_boxes:
        x1_min, y1_min, x1_max, y1_max = current_box
        current_area = (x1_max - x1_min) * (y1_max - y1_min)
        is_contained = False
        for kept_box in filtered_boxes:
            x2_min, y2_min, x2_max, y2_max = kept_box
            overlap_x_min = max(x1_min, x2_min)
            overlap_y_min = max(y1_min, y2_min)
            overlap_x_max = min(x1_max, x2_max)
            overlap_y_max = min(y1_max, y2_max)
            if overlap_x_max > overlap_x_min and overlap_y_max > overlap_y_min:
                overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
                if overlap_area / current_area > area_ratio_threshold:
                    is_contained = True
                    break
        if not is_contained:
            filtered_boxes.append(current_box)
    return filtered_boxes


def extract_foreground_prompts(binary_image_path, x_ratio=0.2, fore_ratio=0.05):
    binary_img = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    if binary_img is None:
        raise ValueError(f"无法读取前景图像: {binary_image_path}")
    _, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    h, w = binary_img.shape[:2]
    total_area = h * w

    dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 0]
    if not areas:
        return [], [], []

    q1 = np.percentile(areas, 25)
    q3 = np.percentile(areas, 75)
    iqr = q3 - q1
    dynamic_threshold = q1 + x_ratio * iqr
    if dynamic_threshold < 1:
        dynamic_threshold = 1

    points = []
    boxes = []
    labels = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < dynamic_threshold:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + bw, w), min(y + bh, h)
        if x2 <= x1 or y2 <= y1:
            continue

        mask_roi = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        shifted_contour = contour - np.array([x1, y1])
        cv2.drawContours(mask_roi, [shifted_contour.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
        roi = binary_img[y1:y2, x1:x2]
        mask_roi = cv2.bitwise_and(mask_roi, roi)

        ys, xs = np.where(mask_roi == 255)
        if len(xs) == 0:
            continue
        gxs = xs + x1
        gys = ys + y1
        dist_values = dist[gys, gxs]
        if len(dist_values) == 0:
            continue

        max_val = np.max(dist_values)
        if max_val < 1:
            continue
        max_mask = dist_values == max_val
        max_indices = np.where(max_mask)
        px = gxs[max_indices[0][0]]
        py = gys[max_indices[0][0]]
        points.append((px, py))
        labels.append(1)

        if area > fore_ratio * total_area:
            boxes.append([x1, y1, x2, y2])

    boxes = filter_nested_boxes(boxes)
    return points, labels, boxes


def extract_background_prompts(binary_image_path, min_area=200, margin=5,
                               ratio=0.05, top_x=3):
    binary_img = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    if binary_img is None:
        raise ValueError(f"无法读取背景图像: {binary_image_path}")
    _, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    h, w = binary_img.shape[:2]
    total_area = h * w

    masked_img = binary_img.copy()
    masked_img[:margin, :] = 0
    masked_img[-margin:, :] = 0
    masked_img[:, :margin] = 0
    masked_img[:, -margin:] = 0

    dist = cv2.distanceTransform(masked_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    contours, _ = cv2.findContours(masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_points = []
    area_threshold_for_grid = ratio * total_area

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + bw, w), min(y + bh, h)
        if x2 <= x1 or y2 <= y1:
            continue

        mask_roi = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        shifted_contour = contour - np.array([x1, y1])
        cv2.drawContours(mask_roi, [shifted_contour.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
        roi = masked_img[y1:y2, x1:x2]
        mask_roi = cv2.bitwise_and(mask_roi, roi)

        ys, xs = np.where(mask_roi == 255)
        if len(xs) == 0:
            continue
        gxs = xs + x1
        gys = ys + y1
        dist_values = dist[gys, gxs]
        if len(dist_values) == 0:
            continue

        max_val = np.max(dist_values)
        if max_val < 1:
            continue
        max_mask = dist_values == max_val
        max_indices = np.where(max_mask)
        px = gxs[max_indices[0][0]]
        py = gys[max_indices[0][0]]
        if (px, py) not in all_points:
            all_points.append((px, py))

        if area > area_threshold_for_grid:
            threshold_80 = np.percentile(dist_values, 80)
            mask_80_local = np.zeros_like(mask_roi, dtype=np.uint8)
            for idx_pixel in range(len(gxs)):
                if dist_values[idx_pixel] >= threshold_80:
                    mask_80_local[ys[idx_pixel], xs[idx_pixel]] = 255

            if np.any(mask_80_local):
                contours_80, _ = cv2.findContours(mask_80_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                components = []
                for cnt in contours_80:
                    area_comp = cv2.contourArea(cnt)
                    if area_comp < 1:
                        continue
                    mask_comp = np.zeros_like(mask_80_local, dtype=np.uint8)
                    cv2.drawContours(mask_comp, [cnt], -1, 255, thickness=cv2.FILLED)
                    comp_ys, comp_xs = np.where(mask_comp == 255)
                    if len(comp_xs) == 0:
                        continue
                    comp_gxs = comp_xs + x1
                    comp_gys = comp_ys + y1
                    comp_dist = dist[comp_gys, comp_gxs]
                    if len(comp_dist) == 0:
                        continue
                    max_comp = np.max(comp_dist)
                    comp_max_mask = comp_dist == max_comp
                    comp_indices = np.where(comp_max_mask)
                    pt_x = comp_gxs[comp_indices[0][0]]
                    pt_y = comp_gys[comp_indices[0][0]]
                    components.append((area_comp, pt_x, pt_y))

                components.sort(key=lambda item: item[0], reverse=True)
                added = 0
                for i in range(min(top_x, len(components))):
                    _, px, py = components[i]
                    if (px, py) not in all_points:
                        all_points.append((px, py))
                        added += 1
                if added == 0 and len(components) == 0:
                    max_in_80 = np.max(dist_values[dist_values >= threshold_80])
                    max_idx = np.where(dist_values == max_in_80)[0][0]
                    px = gxs[max_idx]
                    py = gys[max_idx]
                    if (px, py) not in all_points:
                        all_points.append((px, py))

    labels = [-1] * len(all_points)
    return all_points, labels


def process_ap_images(original_folder, foreground_folder, background_folder, output_folder,
                      fg_x_ratio=0.2, fg_ratio=0.05,
                      bg_min_area=200, bg_margin=5,
                      bg_ratio=0.05, bg_top_x=3, prefix_len=10):
    os.makedirs(output_folder, exist_ok=True)

    fg_files = [f for f in os.listdir(foreground_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    print(f"  AP生成: 前景文件数={len(fg_files)}")

    bg_files = [f for f in os.listdir(background_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    print(f"  AP生成: 背景文件数={len(bg_files)}")

    bg_dict = {}
    for bf in bg_files:
        key = os.path.splitext(bf)[0][:prefix_len]
        bg_dict.setdefault(key, []).append(bf)

    orig_files = [f for f in os.listdir(original_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    orig_dict = {os.path.splitext(f)[0][:prefix_len]: f for f in orig_files}

    processed_basenames = []

    for fg_fname in fg_files:
        fg_key = os.path.splitext(fg_fname)[0][:prefix_len]
        base = os.path.splitext(fg_fname)[0]

        bg_fname = bg_dict.get(fg_key, [None])[0]
        if bg_fname is None:
            print(f"    警告: 未找到与 '{fg_fname}' 匹配的背景图像，跳过。")
            continue

        orig_fname = orig_dict.get(fg_key, None)
        if orig_fname is None:
            orig_path = None
        else:
            orig_path = os.path.join(original_folder, orig_fname)

        fg_path = os.path.join(foreground_folder, fg_fname)
        bg_path = os.path.join(background_folder, bg_fname)

        try:
            fg_pts, fg_lbls, fg_boxes = extract_foreground_prompts(
                fg_path, x_ratio=fg_x_ratio, fore_ratio=fg_ratio
            )
        except Exception as e:
            print(f"    前景提取失败 {fg_fname}: {e}")
            continue

        try:
            bg_pts, bg_lbls = extract_background_prompts(
                bg_path, min_area=bg_min_area, margin=bg_margin,
                ratio=bg_ratio, top_x=bg_top_x
            )
        except Exception as e:
            print(f"    背景提取失败 {bg_fname}: {e}")
            bg_pts, bg_lbls = [], []

        all_points = fg_pts + bg_pts
        all_labels = fg_lbls + bg_lbls

        if orig_path is not None and os.path.exists(orig_path):
            orig_img = cv2.imread(orig_path)
            if orig_img is None:
                orig_img = cv2.imread(fg_path)
                if orig_img is None:
                    orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            orig_img = cv2.imread(fg_path)
            if orig_img is None:
                orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
        if len(orig_img.shape) == 2:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

        img_overlay = orig_img.copy()
        for box in fg_boxes:
            cv2.rectangle(img_overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        for pt in fg_pts:
            cv2.circle(img_overlay, tuple(pt), 5, (0, 0, 255), -1)
        for pt in bg_pts:
            cv2.circle(img_overlay, tuple(pt), 4, (0, 255, 0), -1)
        out_overlay = os.path.join(output_folder, f"{base}_overlay.png")
        cv2.imwrite(out_overlay, img_overlay)

        fg_bin = cv2.imread(fg_path, cv2.IMREAD_GRAYSCALE)
        bg_bin = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
        if fg_bin is not None and bg_bin is not None:
            _, fg_bin = cv2.threshold(fg_bin, 127, 255, cv2.THRESH_BINARY)
            _, bg_bin = cv2.threshold(bg_bin, 127, 255, cv2.THRESH_BINARY)
            mask_canvas = np.zeros((fg_bin.shape[0], fg_bin.shape[1], 3), dtype=np.uint8)
            mask_canvas[fg_bin == 255] = (255, 255, 255)
            bg_mask = (bg_bin == 255) & (fg_bin == 0)
            mask_canvas[bg_mask] = (128, 128, 128)
            for box in fg_boxes:
                cv2.rectangle(mask_canvas, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
            for pt in fg_pts:
                cv2.circle(mask_canvas, tuple(pt), 5, (0, 0, 255), -1)
            for pt in bg_pts:
                cv2.circle(mask_canvas, tuple(pt), 4, (0, 255, 0), -1)
            out_mask = os.path.join(output_folder, f"{base}_mask_overlay.png")
            cv2.imwrite(out_mask, mask_canvas)

        txt_path = os.path.join(output_folder, f"{base}_prompts.txt")
        with open(txt_path, 'w') as f:
            f.write("point prompts: [")
            for i, pt in enumerate(all_points):
                if i > 0:
                    f.write(", ")
                f.write(f"({pt[0]}, {pt[1]})")
            f.write("]\n")
            f.write("labels: " + str(all_labels) + "\n")
            f.write("box prompts: " + str(fg_boxes) + "\n")

        processed_basenames.append(base)

    return processed_basenames


# ==================== 主流水线函数 ====================
def run_pipeline(input_folder, output_folder, temp_dir=None):
    """
    执行完整流水线：低通滤波 → 纹理提取 → 轮廓提取 → 衬度提取 → 逻辑操作 → 小连通区域过滤 → AP生成
    所有中间文件保存在 temp_dir（默认输出文件夹下的 pipeline_temp），且不会被自动删除。
    过滤后强制确保 coarse_filtered 非空。
    """
    start_time = time.time()
    print(f"开始处理文件夹: {input_folder}")

    if temp_dir is None:
        temp_dir = os.path.join(output_folder, "pipeline_temp")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"临时目录（所有中间结果）: {temp_dir}")

    # 定义子目录
    lpf_dir = os.path.join(temp_dir, "lpf")
    tex_fore_conn = os.path.join(temp_dir, "tex_fore_conn")
    tex_back_conn = os.path.join(temp_dir, "tex_back_conn")
    tex_fore_tex = os.path.join(temp_dir, "tex_fore_tex")
    tex_back_tex = os.path.join(temp_dir, "tex_back_tex")
    contour_morph = os.path.join(temp_dir, "contour_morph")
    contour_after = os.path.join(temp_dir, "contour_after")
    contrast_fore = os.path.join(temp_dir, "contrast_fore")
    contrast_back = os.path.join(temp_dir, "contrast_back")
    contrast_contour = os.path.join(temp_dir, "contrast_contour")
    forecoarse = os.path.join(temp_dir, "forecoarse")
    backcoarse = os.path.join(temp_dir, "backcoarse")

    # 1. 低通滤波
    print("执行低通滤波...")
    success = process_lowpass_filter(input_folder, lpf_dir, r=1, r_noise=100)
    if not success:
        print("低通滤波失败，终止。")
        return False
    print(f"  低通滤波生成 {len(glob.glob(os.path.join(lpf_dir, '*')))} 个文件")



    # # 2. 纹理提取
    # print("执行纹理提取...")
    # fg_params = {
    #     'percent_high': 0.9, 'percent_low': 0.6, 'contour_tex_ratio': 0,
    #     'min_contour_length': 0, 'tex_density1_enable': True,
    #     'tex_density_win1': 40, 'tex_density_thr1': 22,
    #     'tex_density2_enable': True, 'tex_density_win2': 16,
    #     'tex_density_thr2': 10, 'symmetry_half_width': 2,
    #     'symmetry_threshold_deg': 135,
    #     'morph_shape': 'rect', 'morph_size': 2, 'morph_iter': 1
    # }
    #

    # 2. 纹理提取
    print("执行纹理提取...")
    fg_params = {
        'percent_high': 0.9, 'percent_low': 0.8, 'contour_tex_ratio': 0,
        'min_contour_length': 0, 'tex_density1_enable': True,
        'tex_density_win1': 40, 'tex_density_thr1': 22,
        'tex_density2_enable': True, 'tex_density_win2': 16,
        'tex_density_thr2': 10, 'symmetry_half_width': 2,
        'symmetry_threshold_deg': 135,
        'morph_shape': 'rect', 'morph_size': 2, 'morph_iter': 1
    }


    bg_params = {
        'percent_high': 0.9, 'percent_low': 0.7, 'contour_tex_ratio': 0,
        'min_contour_length': 0, 'tex_density1_enable': True,
        'tex_density_win1': 40, 'tex_density_thr1': 20,
        'tex_density2_enable': True, 'tex_density_win2': 16,
        'tex_density_thr2': 5, 'symmetry_half_width': 2,
        'symmetry_threshold_deg': 135,
        'morph_shape': 'rect', 'morph_size': 2, 'morph_iter': 1
    }
    process_texture_batch(input_folder, tex_fore_conn, tex_back_conn,
                          tex_fore_tex, tex_back_tex, fg_params, bg_params)
    print(f"  纹理提取完成，前景连接图 {len(glob.glob(os.path.join(tex_fore_conn, '*')))} 个文件")

    # 3. 轮廓提取
    print("执行轮廓提取...")
    process_contour_folder(lpf_dir, contour_after, contour_morph,
                           0.9, 0.8, 0.2, 'rect', 4, 1, output_before_dir=None)
    print(f"  轮廓提取完成，形态学结果 {len(glob.glob(os.path.join(contour_morph, '*')))} 个文件")


    # 4. 衬度提取
    print("执行衬度提取...")
    process_contrast_images(lpf_dir, contrast_fore, contrast_back,
                            0.9, 1.4, invert_fore=True, invert_back=False)
    print(f"  衬度提取完成，前景 {len(glob.glob(os.path.join(contrast_fore, '*')))} 个文件")

    # # 4. 衬度提取
    # print("执行衬度提取...")
    # process_contrast_images(lpf_dir, contrast_fore, contrast_back,
    #                         0.9, 1.0, invert_fore=True, invert_back=False)
    # print(f"  衬度提取完成，前景 {len(glob.glob(os.path.join(contrast_fore, '*')))} 个文件")

    # 5. 逻辑操作
    print("执行逻辑操作...")
    prefix_len = 2
    # 前景差集
    count1, _ = logic_operation_folder(contrast_fore, contour_morph, "差集(A-B)",
                                       contrast_contour, prefix_len)
    print(f"  前景差集生成 {count1} 个文件")
    # 前景并集
    count2, _ = logic_operation_folder(contrast_contour, tex_fore_conn, "并集",
                                       forecoarse, prefix_len)
    print(f"  前景并集生成 {count2} 个文件")
    # 背景差集
    count3, _ = logic_operation_folder(contrast_back, tex_back_conn, "差集(A-B)",
                                       backcoarse, prefix_len)
    print(f"  背景差集生成 {count3} 个文件")

    # ===== 5.5 过滤前景 coarse 的小连通区域 =====
    print("过滤前景 coarse 的小连通区域...")
    coarse_filtered = os.path.join(temp_dir, "coarse_filtered")
    filter_small_connected(forecoarse, coarse_filtered, x_val=3.0)

    # 检查过滤结果，若为空则强制复制 forecoarse
    filtered_files = glob.glob(os.path.join(coarse_filtered, '*'))
    print(f"  过滤后前景图像数: {len(filtered_files)}")
    if len(filtered_files) == 0:
        print("  ⚠️ 警告：过滤后无图像，将 forecoarse 复制到 coarse_filtered")
        fore_files = glob.glob(os.path.join(forecoarse, '*'))
        for f in fore_files:
            shutil.copy2(f, coarse_filtered)
        filtered_files = glob.glob(os.path.join(coarse_filtered, '*'))
        print(f"  复制后前景图像数: {len(filtered_files)}")

    # 使用过滤后的前景（一定非空）
    foreground_ap = coarse_filtered

    # ===== 6. AP Generate =====
    print("执行 AP Generate...")
    ap_output = output_folder
    os.makedirs(ap_output, exist_ok=True)
    basenames = process_ap_images(
        original_folder=input_folder,
        foreground_folder=foreground_ap,
        background_folder=backcoarse,
        output_folder=ap_output,
        fg_x_ratio=0.2,
        fg_ratio=0.05,
        bg_min_area=200,
        bg_margin=5,
        bg_ratio=0.05,
        bg_top_x=3,
        prefix_len=prefix_len
    )
    print(f"AP Generate 完成，生成了 {len(basenames)} 组提示。")

    print(f"所有中间结果已保存至: {temp_dir}")
    elapsed = time.time() - start_time
    print(f"总耗时: {elapsed:.2f} 秒")
    return True


# ==================== 文件对话框和入口 ====================
def select_folder(title="请选择文件夹"):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        if not os.path.isdir(input_dir):
            print(f"错误：输入文件夹 '{input_dir}' 不存在。")
            sys.exit(1)
        os.makedirs(output_dir, exist_ok=True)
        run_pipeline(input_dir, output_dir)
    else:
        print("请在弹出的对话框中选择输入图像文件夹...")
        input_dir = select_folder("请选择原始图像文件夹")
        if not input_dir:
            print("未选择输入文件夹，程序退出。")
            sys.exit(1)
        print(f"输入文件夹: {input_dir}")
        output_dir = os.path.join(os.path.dirname(input_dir), "SEMAP")
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出文件夹: {output_dir}")
        run_pipeline(input_dir, output_dir)


