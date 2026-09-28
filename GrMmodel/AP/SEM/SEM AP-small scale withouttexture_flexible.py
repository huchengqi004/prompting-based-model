
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立图像处理流水线：从原始图像直接生成 AP 提示（红色点、黄色框、绿色点）
所有中间结果保存在输出文件夹下的 pipeline_temp 子目录中，不会自动删除。
新增：逻辑操作后对 forecoarse 进行小连通区域过滤，并强制确保 coarse_filtered 非空。
修改：背景大区域提示点提取逻辑，并保存分位数区域。
修改：前景大区域不再直接取最大距离点，而是采用分位数区域多组件取点，且不再输出黄色框。
修改：分位数区域合并保存，前景为白色，背景为灰色，输出到 distance_transform_percentile 文件夹。
修改：backcoarse 生成逻辑为 contrast_back - tex_back_conn - contour_morph。
修改：x_multiplier 参数可通过 process_ap_images / run_pipeline 进行配置。
新增：支持基于最小距离的提示点筛选（前景和背景分别设置 grid_size_fore / grid_size_back）。
新增：跨类筛选（dis_foreback），前景和背景距离过近时按灰度规则删除。
新增：分位数区域提取分为低分位数（小区域）和高分位数（大区域），并分别设置组件数。
新增：distance_transform_percentile 图像同时包含前景和背景的小区域和大区域分位数区域，并过滤小区域组件。
新增：输出最终距离变换图叠加提示点的结果。
新增：grid_final 网格筛选，每个网格内保留灰度最优的前景/背景提示点。
新增：确保每个连通区域至少有一个提示点（在最终筛选前进行补点）。
新增：灰度邻域计算改为圆形邻域，半径由 neighbour_l 指定。
新增：距离筛选（grid_size_fore/grid_size_back）改为比较提示点所在连通区域的平均灰度（前景保留低者，背景保留高者）。
新增：grid_final 面积保留比例改为 distransf_area_select_ratio 参数控制。
新增：网格筛选等级2、等级3条件可通过 first_condition_id, second_condition_id, final_condition_id 配置。
新增：轮廓提取拆分为前景和背景两套参数，分别用于前景和背景粗糙筛选。
新增：网格筛选按图像宽高均匀划分 int(w/grid_size) × int(h/grid_size) 个网格。
新增：网格尺寸系数 grid_size_fore_indice 和背景比例 grid_size_backforeratio 可配置。
新增：distance_transform_overlay 图使用筛选后的连通区域，仅保留最终提示点所在的区域。
新增：grid_size 根据 distance_transform_percentile 中前景连通区域面积最大的前 distranfirstnum 个区域的平均面积计算。
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
    if x_multiplier > 0:
        threshold = q1 + x_multiplier * iqr
    else:
        threshold = 1
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
        cv2.imwrite(os.path.join(fore_out_dir, base + "_" + str(new_th) + ".png"), binary)

        new_th = np.clip(ratio_back * otsu_th, 0, 255)
        _, binary = cv2.threshold(img, new_th, 255, cv2.THRESH_BINARY)
        if invert_back:
            binary = cv2.bitwise_not(binary)
        cv2.imwrite(os.path.join(back_out_dir, base + "_" + str(new_th) + ".png"), binary)


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


# ==================== 过滤小连通区域 ====================
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


def extract_foreground_prompts(binary_image_path, x_ratio=0.2, fore_ratio=0.05,
                               dis_transf_perce_low=60, dis_transf_per_high=80,
                               x_lratio=1, x_hratio=3, percentile_mask_full=None):
    """
    提取前景提示点（红色点）和框（大区域原输出框，本次修改后不再输出框）。
    修改逻辑：
    - 面积 <= fore_ratio * total_area 的小区域：
      取区域内距离变换值的 dis_transf_perce_low 分位数区域，
      再根据 x_ratio 对分位数区域中的连通组件进行面积过滤，
      然后取前 x_lratio 个组件（每个组件最大距离点）。
    - 面积 > fore_ratio * total_area 的大区域：
      取距离变换值的 dis_transf_per_high 分位数区域，再取前 x_hratio 个组件（每个组件最大距离点）。
    - 如果提供了 percentile_mask_full，则将分位数区域填充为白色(255)，仅填充过滤后的区域。
    """
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
    labels = []
    boxes = []  # 不再生成框

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

        if area <= fore_ratio * total_area:
            # 小区域：使用低分位数区域提取多个组件
            threshold_percentile = np.percentile(dist_values, dis_transf_perce_low)
            mask_percentile_local = np.zeros_like(mask_roi, dtype=np.uint8)
            for idx_pixel in range(len(gxs)):
                if dist_values[idx_pixel] >= threshold_percentile:
                    mask_percentile_local[ys[idx_pixel], xs[idx_pixel]] = 255

            # 对分位数区域中的连通组件进行面积过滤
            if np.any(mask_percentile_local):
                contours_percentile, _ = cv2.findContours(
                    mask_percentile_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                comp_areas = [cv2.contourArea(cnt) for cnt in contours_percentile]
                if comp_areas:
                    q1_area = np.percentile(comp_areas, 25)
                    q3_area = np.percentile(comp_areas, 75)
                    iqr_area = q3_area - q1_area
                    area_threshold = q1_area + x_ratio * iqr_area
                    kept_components = []
                    for cnt, a in zip(contours_percentile, comp_areas):
                        if a >= area_threshold:
                            kept_components.append(cnt)
                    mask_filtered = np.zeros_like(mask_percentile_local)
                    cv2.drawContours(mask_filtered, kept_components, -1, 255, thickness=cv2.FILLED)
                    mask_percentile_local = mask_filtered
                else:
                    mask_percentile_local = np.zeros_like(mask_percentile_local)

            # 填充分位数区域到全局掩膜（白色），仅填充过滤后的区域
            if percentile_mask_full is not None and np.any(mask_percentile_local):
                percentile_mask_full[y1:y2, x1:x2] = np.maximum(
                    percentile_mask_full[y1:y2, x1:x2],
                    mask_percentile_local
                )

            if np.any(mask_percentile_local):
                contours_percentile, _ = cv2.findContours(
                    mask_percentile_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                components = []
                for cnt in contours_percentile:
                    area_comp = cv2.contourArea(cnt)
                    if area_comp < 1:
                        continue
                    mask_comp = np.zeros_like(mask_percentile_local, dtype=np.uint8)
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
                for i in range(min(x_lratio, len(components))):
                    _, px, py = components[i]
                    points.append((px, py))
                    labels.append(1)
                if len(components) == 0:
                    # 退化情况：取整个区域最大距离点
                    max_val = np.max(dist_values)
                    max_indices = np.where(dist_values == max_val)
                    px = gxs[max_indices[0][0]]
                    py = gys[max_indices[0][0]]
                    points.append((px, py))
                    labels.append(1)
            else:
                # mask 为空，退化取最大距离点
                max_val = np.max(dist_values)
                max_indices = np.where(dist_values == max_val)
                px = gxs[max_indices[0][0]]
                py = gys[max_indices[0][0]]
                points.append((px, py))
                labels.append(1)
        else:
            # 大区域：使用高分位数区域提取多个组件
            threshold_percentile = np.percentile(dist_values, dis_transf_per_high)
            mask_percentile_local = np.zeros_like(mask_roi, dtype=np.uint8)
            for idx_pixel in range(len(gxs)):
                if dist_values[idx_pixel] >= threshold_percentile:
                    mask_percentile_local[ys[idx_pixel], xs[idx_pixel]] = 255

            # 填充分位数区域到全局掩膜（白色）
            if percentile_mask_full is not None and np.any(mask_percentile_local):
                percentile_mask_full[y1:y2, x1:x2] = np.maximum(
                    percentile_mask_full[y1:y2, x1:x2],
                    mask_percentile_local
                )

            if np.any(mask_percentile_local):
                contours_percentile, _ = cv2.findContours(
                    mask_percentile_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                components = []
                for cnt in contours_percentile:
                    area_comp = cv2.contourArea(cnt)
                    if area_comp < 1:
                        continue
                    mask_comp = np.zeros_like(mask_percentile_local, dtype=np.uint8)
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
                for i in range(min(x_hratio, len(components))):
                    _, px, py = components[i]
                    points.append((px, py))
                    labels.append(1)

                if len(components) == 0:
                    # 退化情况
                    max_val = np.max(dist_values)
                    max_indices = np.where(dist_values == max_val)
                    px = gxs[max_indices[0][0]]
                    py = gys[max_indices[0][0]]
                    points.append((px, py))
                    labels.append(1)

    return points, labels, boxes


def extract_background_prompts(binary_image_path, min_area=200, margin=5,
                               ratio=0.05, dis_transf_perce_low=60, dis_transf_per_high=80,
                               x_lratio=1, x_hratio=3, percentile_mask_full=None):
    """
    提取背景提示点。

    修改逻辑：
    - 面积 >= min_area 且 <= ratio * total_area 的小区域：
      取区域内距离变换值的 dis_transf_perce_low 分位数区域，
      再根据 min_area 对分位数区域中的连通组件进行面积过滤，
      然后取前 x_lratio 个组件（每个组件最大距离点）。
    - 面积 > ratio * total_area 的大区域：
      取距离变换值的 dis_transf_per_high 分位数区域，再取前 x_hratio 个组件（每个组件最大距离点）。
    - 如果提供了 percentile_mask_full，则将分位数区域填充为灰色(128)，仅填充过滤后的区域。
    """
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

        if area <= area_threshold_for_grid:
            # 小区域：使用低分位数区域
            threshold_percentile = np.percentile(dist_values, dis_transf_perce_low)
            mask_percentile_local = np.zeros_like(mask_roi, dtype=np.uint8)
            for idx_pixel in range(len(gxs)):
                if dist_values[idx_pixel] >= threshold_percentile:
                    mask_percentile_local[ys[idx_pixel], xs[idx_pixel]] = 255

            # 对分位数区域中的连通组件进行面积过滤（使用 min_area 作为固定阈值）
            if np.any(mask_percentile_local):
                contours_percentile, _ = cv2.findContours(
                    mask_percentile_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                comp_areas = [cv2.contourArea(cnt) for cnt in contours_percentile]
                if comp_areas:
                    kept_components = []
                    for cnt, a in zip(contours_percentile, comp_areas):
                        if a >= min_area:  # 使用传入的 min_area
                            kept_components.append(cnt)
                    mask_filtered = np.zeros_like(mask_percentile_local)
                    cv2.drawContours(mask_filtered, kept_components, -1, 255, thickness=cv2.FILLED)
                    mask_percentile_local = mask_filtered
                else:
                    mask_percentile_local = np.zeros_like(mask_percentile_local)

            # 填充分位数区域到全局掩膜（灰色），仅填充过滤后的区域，且不覆盖前景
            if percentile_mask_full is not None and np.any(mask_percentile_local):
                target_region = percentile_mask_full[y1:y2, x1:x2]
                update_mask = (target_region == 0) & (mask_percentile_local == 255)
                target_region[update_mask] = 128

            if np.any(mask_percentile_local):
                contours_percentile, _ = cv2.findContours(
                    mask_percentile_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                components = []
                for cnt in contours_percentile:
                    area_comp = cv2.contourArea(cnt)
                    if area_comp < 1:
                        continue
                    mask_comp = np.zeros_like(mask_percentile_local, dtype=np.uint8)
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
                for i in range(min(x_lratio, len(components))):
                    _, px, py = components[i]
                    if (px, py) not in all_points:
                        all_points.append((px, py))
                if len(components) == 0:
                    max_val = np.max(dist_values)
                    max_indices = np.where(dist_values == max_val)
                    px = gxs[max_indices[0][0]]
                    py = gys[max_indices[0][0]]
                    if (px, py) not in all_points:
                        all_points.append((px, py))
            else:
                max_val = np.max(dist_values)
                max_indices = np.where(dist_values == max_val)
                px = gxs[max_indices[0][0]]
                py = gys[max_indices[0][0]]
                if (px, py) not in all_points:
                    all_points.append((px, py))
        else:
            # 大区域：使用高分位数区域
            threshold_percentile = np.percentile(dist_values, dis_transf_per_high)
            mask_percentile_local = np.zeros_like(mask_roi, dtype=np.uint8)
            for idx_pixel in range(len(gxs)):
                if dist_values[idx_pixel] >= threshold_percentile:
                    mask_percentile_local[ys[idx_pixel], xs[idx_pixel]] = 255

            # 填充分位数区域到全局掩膜（灰色），但仅填充非前景区域（值为0的像素）
            if percentile_mask_full is not None and np.any(mask_percentile_local):
                target_region = percentile_mask_full[y1:y2, x1:x2]
                update_mask = (target_region == 0) & (mask_percentile_local == 255)
                target_region[update_mask] = 128

            if np.any(mask_percentile_local):
                contours_percentile, _ = cv2.findContours(
                    mask_percentile_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                components = []
                for cnt in contours_percentile:
                    area_comp = cv2.contourArea(cnt)
                    if area_comp < 1:
                        continue
                    mask_comp = np.zeros_like(mask_percentile_local, dtype=np.uint8)
                    cv2.drawContours(mask_comp, [cnt], -1, 255, thickness=cv2.FILLED)
                    comp_ys, comp_xs = np.where(mask_comp == 255)
                    if len(comp_xs) == 0:
                        continue
                    comp_gxs = comp_xs + x1
                    comp_gys = comp_ys + y1
                    comp_dist = dist[comp_gys, comp_gxs]  # 修正原笔误
                    if len(comp_dist) == 0:
                        continue
                    max_comp = np.max(comp_dist)
                    comp_max_mask = comp_dist == max_comp
                    comp_indices = np.where(comp_max_mask)
                    pt_x = comp_gxs[comp_indices[0][0]]
                    pt_y = comp_gys[comp_indices[0][0]]
                    components.append((area_comp, pt_x, pt_y))

                components.sort(key=lambda item: item[0], reverse=True)
                for i in range(min(x_hratio, len(components))):
                    _, px, py = components[i]
                    if (px, py) not in all_points:
                        all_points.append((px, py))

                if len(components) == 0:
                    max_in_percentile = np.max(dist_values[dist_values >= threshold_percentile])
                    max_idx = np.where(dist_values == max_in_percentile)[0][0]
                    px = gxs[max_idx]
                    py = gys[max_idx]
                    if (px, py) not in all_points:
                        all_points.append((px, py))

    labels = [-1] * len(all_points)
    return all_points, labels


def average_gray_circular_neighbor(gray_img, x, y, radius):
    """
    计算以 (x,y) 为中心、半径为 radius 的圆形邻域内的平均灰度值。
    忽略越界像素，若邻域内无有效像素则返回 0。
    """
    if radius is None or radius <= 0:
        if 0 <= x < gray_img.shape[1] and 0 <= y < gray_img.shape[0]:
            return float(gray_img[y, x])
        else:
            return 0.0
    h, w = gray_img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    mask = dist <= radius
    vals = gray_img[mask]
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


def filter_points_by_grid_area(points, percentile_mask, grid_final, is_foreground=True, keep_ratio=0.5):
    """
    根据 grid_final 启用过滤：保留面积前 keep_ratio（默认0.5，即50%）的连通区域内的点。
    比较范围是 percentile_mask 中对应类别的所有连通区域（全局），
    包含所有前景/背景提示点所在的 distance_transform_percentile 连通区域。
    grid_final 仅作为启用开关（>0 启用），不再用于网格划分。

    参数：
        points: 候选点列表 [(x,y), ...]
        percentile_mask: 全局掩膜，前景=255，背景=128
        grid_final: 启用过滤的开关（>0 启用）
        is_foreground: True 表示前景，False 表示背景
        keep_ratio: 保留面积前 keep_ratio 的区域（0<keep_ratio<=1），默认0.5
    """
    if grid_final <= 0 or not points:
        return points

    h, w = percentile_mask.shape
    if is_foreground:
        binary = (percentile_mask == 255).astype(np.uint8) * 255
    else:
        binary = (percentile_mask == 128).astype(np.uint8) * 255

    if np.count_nonzero(binary) == 0:
        return points

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 收集所有有效区域面积（全局）
    region_areas = {lab: stats[lab, cv2.CC_STAT_AREA] for lab in range(1, num_labels)}
    if not region_areas:
        return points

    # 按面积降序排列，保留前 keep_ratio 的区域（至少保留一个）
    sorted_labs = sorted(region_areas.keys(), key=lambda l: region_areas[l], reverse=True)
    keep_count = max(1, int(np.ceil(len(sorted_labs) * keep_ratio)))
    keep_labels = set(sorted_labs[:keep_count])

    # 仅保留位于保留区域内的点
    filtered_points = []
    for pt in points:
        x, y = pt
        if 0 <= x < w and 0 <= y < h:
            lab = labels[y, x]
            if lab in keep_labels:
                filtered_points.append(pt)
    return filtered_points


def compute_grid_size_from_mask(mask, top_n=10, grid_size_fore_indice=4):
    """
    根据二值掩膜中的连通区域计算网格尺寸。
    取面积最大的前 top_n 个连通区域，计算平均面积。
    若 top_n <= 0 或 top_n >= 区域总数，则使用所有区域。
    返回平均面积的平方根乘以 grid_size_fore_indice 并取整（至少为1，若无区域则返回0）。
    """
    if np.count_nonzero(mask) == 0:
        return 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    if not areas:
        return 0

    sorted_areas = sorted(areas, reverse=True)
    if top_n <= 0 or top_n >= len(sorted_areas):
        kept_areas = sorted_areas
    else:
        kept_areas = sorted_areas[:top_n]

    avg_area = float(np.mean(kept_areas))
    grid_size = grid_size_fore_indice * int(np.sqrt(avg_area))
    if grid_size < 1:
        grid_size = 1
    return grid_size


def filter_points_by_grid_conditions(points, gray_img, grid_size, keep_low=True,
                                     neighbour_l=0, region_label_img=None,
                                     region_area=None, region_avg_gray=None,
                                     first_condition_id=2, second_condition_id=3,
                                     final_condition_id=3):
    """
    按网格筛选提示点，每个网格内选择最优提示点，优先级如下：
        等级1：同时满足条件①面积最大、②区域平均灰度最优、③邻域灰度最优
        等级2：同时满足 first_condition_id 和 second_condition_id 指定的两个条件
               （条件编号：1=面积最大，2=区域平均灰度最优，3=邻域灰度最优）
        等级3：满足 final_condition_id 指定的单个条件
    同一等级内按面积降序、区域灰度优、邻域灰度优排序。

    网格划分：根据 grid_size 计算网格列数 n_cols = int(w / grid_size) 和行数 n_rows = int(h / grid_size)，
    均匀覆盖整个图像，网格索引通过比例计算。

    参数：
        points: 候选点列表 [(x,y), ...]
        gray_img: 原始灰度图
        grid_size: 网格边长参考值（像素），实际网格数量由图像宽高决定
        keep_low: True 表示前景（偏好低灰度），False 表示背景（偏好高灰度）
        neighbour_l: 圆形邻域半径，0 表示仅中心像素
        region_label_img: 区域标签图（distance_transform_percentile 中对应类的连通区域）
        region_area: 区域面积字典 {label: area}
        region_avg_gray: 区域平均灰度字典 {label: avg_gray}
        first_condition_id, second_condition_id: 等级2需要同时满足的条件编号（1,2,3）
        final_condition_id: 等级3需要满足的条件编号（1,2,3）
    """
    if grid_size <= 0 or not points:
        return points

    h, w = gray_img.shape[:2]
    # 计算网格列数和行数
    n_cols = max(1, int(w / grid_size))
    n_rows = max(1, int(h / grid_size))

    # 验证条件编号
    valid_ids = {1, 2, 3}
    if first_condition_id not in valid_ids or second_condition_id not in valid_ids \
            or final_condition_id not in valid_ids:
        raise ValueError("条件编号必须为 1, 2 或 3")

    grid_dict = {}
    for pt in points:
        gx = int(pt[0] * n_cols / w)
        gy = int(pt[1] * n_rows / h)
        gx = min(gx, n_cols - 1)
        gy = min(gy, n_rows - 1)
        grid_dict.setdefault((gx, gy), []).append(pt)

    result = []
    tol = 1e-6

    for pts in grid_dict.values():
        if len(pts) == 1:
            result.append(pts[0])
            continue

        # 收集每个点的属性
        items = []
        for p in pts:
            lab = region_label_img[p[1], p[0]] if region_label_img is not None else 0
            area = region_area.get(lab, 0) if region_area is not None else 0
            avg = region_avg_gray.get(lab, None) if region_avg_gray is not None else None
            if avg is None:
                avg = average_gray_circular_neighbor(gray_img, p[0], p[1], neighbour_l)
            neigh = average_gray_circular_neighbor(gray_img, p[0], p[1], neighbour_l)
            items.append({'pt': p, 'area': area, 'avg': avg, 'neigh': neigh})

        # 网格内极值
        max_area = max(item['area'] for item in items)
        if keep_low:
            opt_avg = min(item['avg'] for item in items)
            opt_neigh = min(item['neigh'] for item in items)
        else:
            opt_avg = max(item['avg'] for item in items)
            opt_neigh = max(item['neigh'] for item in items)

        # 计算条件布尔值
        for item in items:
            item['c1'] = abs(item['area'] - max_area) <= tol
            item['c2'] = abs(item['avg'] - opt_avg) <= tol
            item['c3'] = abs(item['neigh'] - opt_neigh) <= tol

        # 判断优先级等级
        def get_level(item):
            # 等级1：全部满足
            if item['c1'] and item['c2'] and item['c3']:
                return 0
            # 等级2：满足 first_condition_id 和 second_condition_id
            cond1 = item[f'c{first_condition_id}']
            cond2 = item[f'c{second_condition_id}']
            if cond1 and cond2:
                return 1
            # 等级3：满足 final_condition_id
            cond_final = item[f'c{final_condition_id}']
            if cond_final:
                return 2
            # 兜底（理论不会执行）
            return 3

        min_level = min(get_level(item) for item in items)
        candidates = [item for item in items if get_level(item) == min_level]

        # 同一等级内次级排序
        def sort_key(item):
            if keep_low:
                return (-item['area'], item['avg'], item['neigh'])
            else:
                return (-item['area'], -item['avg'], -item['neigh'])

        candidates.sort(key=sort_key)
        result.append(candidates[0]['pt'])

    return result


def filter_cross_fg_bg(fg_pts, bg_pts, gray_img, dis_foreback, ratio=0.8, neighbour_l=0):
    """
    对前景点和背景点进行跨类筛选：
    找出所有距离 < dis_foreback 的前景-背景点对，按距离升序处理。
    对每一对：
        - 若前景点圆形邻域平均灰度 < 背景点圆形邻域平均灰度 * ratio，则保留前景点，删除背景点；
        - 否则同时删除该前景点和背景点。
    """
    if dis_foreback <= 0 or not fg_pts or not bg_pts:
        return fg_pts, bg_pts

    fg_pts = list(fg_pts)
    bg_pts = list(bg_pts)
    n_fg = len(fg_pts)
    n_bg = len(bg_pts)

    pairs = []
    for i in range(n_fg):
        for j in range(n_bg):
            p1 = fg_pts[i]
            p2 = bg_pts[j]
            dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if dist < dis_foreback:
                pairs.append((dist, i, j))

    if not pairs:
        return fg_pts, bg_pts

    pairs.sort(key=lambda x: x[0])

    removed_fg = [False] * n_fg
    removed_bg = [False] * n_bg

    for dist, i, j in pairs:
        if removed_fg[i] or removed_bg[j]:
            continue

        gray_fg = average_gray_circular_neighbor(gray_img, fg_pts[i][0], fg_pts[i][1], neighbour_l)
        gray_bg = average_gray_circular_neighbor(gray_img, bg_pts[j][0], bg_pts[j][1], neighbour_l)

        if gray_fg < gray_bg * ratio:
            removed_bg[j] = True
        else:
            removed_fg[i] = True
            removed_bg[j] = True

    new_fg = [pt for i, pt in enumerate(fg_pts) if not removed_fg[i]]
    new_bg = [pt for j, pt in enumerate(bg_pts) if not removed_bg[j]]
    return new_fg, new_bg


def ensure_points_in_regions(binary_mask, points, dist_transform, label_value):
    """
    确保 binary_mask 中的每个连通组件内至少有一个点。
    若组件内没有已有点，则选取距离变换值最大的像素添加。
    """
    if not points:
        points = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    new_points = list(points)
    new_labels = [label_value] * len(points)

    for lab in range(1, num_labels):
        mask_comp = (labels == lab)
        has_point = False
        for pt in points:
            if mask_comp[pt[1], pt[0]]:
                has_point = True
                break
        if has_point:
            continue

        ys, xs = np.where(mask_comp)
        if len(xs) == 0:
            continue

        dist_vals = dist_transform[ys, xs]
        max_idx = np.argmax(dist_vals)
        if dist_vals[max_idx] > 0:
            cy = ys[max_idx]
            cx = xs[max_idx]
        else:
            cx = int(stats[lab, cv2.CC_STAT_LEFT] + stats[lab, cv2.CC_STAT_WIDTH] / 2)
            cy = int(stats[lab, cv2.CC_STAT_TOP] + stats[lab, cv2.CC_STAT_HEIGHT] / 2)

        new_points.append((cx, cy))
        new_labels.append(label_value)

    return new_points, new_labels


def draw_grids(img, grid_size_fore, grid_size_back):
    """
    在图像副本上绘制均匀分布的网格线：
    - 前景网格线为红色（单像素）
    - 背景网格线为绿色（单像素）
    网格数量由图像宽高和 grid_size 决定：横向 int(w/grid_size) 条，纵向 int(h/grid_size) 条。
    """
    img_copy = img.copy()
    h, w = img_copy.shape[:2]

    if grid_size_fore > 0:
        n_cols_fore = max(1, int(w / grid_size_fore))
        n_rows_fore = max(1, int(h / grid_size_fore))
        for i in range(1, n_cols_fore):
            x = int(round(i * w / n_cols_fore))
            cv2.line(img_copy, (x, 0), (x, h), (0, 0, 255), 1, cv2.LINE_AA)
        for j in range(1, n_rows_fore):
            y = int(round(j * h / n_rows_fore))
            cv2.line(img_copy, (0, y), (w, y), (0, 0, 255), 1, cv2.LINE_AA)

    if grid_size_back > 0:
        n_cols_back = max(1, int(w / grid_size_back))
        n_rows_back = max(1, int(h / grid_size_back))
        for i in range(1, n_cols_back):
            x = int(round(i * w / n_cols_back))
            cv2.line(img_copy, (x, 0), (x, h), (0, 255, 0), 1, cv2.LINE_AA)
        for j in range(1, n_rows_back):
            y = int(round(j * h / n_rows_back))
            cv2.line(img_copy, (0, y), (w, y), (0, 255, 0), 1, cv2.LINE_AA)

    return img_copy


def process_ap_images(original_folder, foreground_folder, background_folder, output_folder,
                      fg_x_ratio=0.2, fg_ratio=0.05,
                      bg_min_area=200, bg_margin=5,
                      bg_ratio=0.05, prefix_len=10,
                      dis_transf_perce_low=60, dis_transf_per_high=80,
                      x_lratio=1, x_hratio=3,
                      distance_transform_percentile_dir=None,
                      x_multiplier=0.2, grid_size_fore=0, grid_size_back=0,
                      dis_foreback=0, grid_final=1000, neighbour_l=0,
                      distransf_area_select_ratio=0.5,
                      distranfirstnum=10,               # 新增：取面积最大的前 distranfirstnum 个前景距离变换区域
                      first_condition_id=2, second_condition_id=3,
                      final_condition_id=3,
                      grid_size_fore_indice=4,
                      grid_size_backforeratio=1.5):
    """
    AP 提示生成，同时保存合并后的距离变换分位数区域图（前景白色，背景灰色），包含小区域和大区域。
    grid_size_fore 根据 distance_transform_percentile 中前景连通区域面积最大的前 distranfirstnum 个区域的平均面积计算；
    grid_size_back 为 grid_size_fore 乘以 grid_size_backforeratio。
    新增参数：
        distranfirstnum: 用于计算前景网格尺寸的连通区域数量（取面积最大的前 N 个），默认10。
        distransf_area_select_ratio: 用于全局面积过滤的比例（grid_final > 0 时生效）。
        first_condition_id, second_condition_id: 等级2需要满足的条件编号（1=面积最大，2=区域灰度最优，3=邻域灰度最优）。
        final_condition_id: 等级3需要满足的条件编号。
        grid_size_fore_indice: 前景网格尺寸系数（平均面积平方根乘以该系数）。
        grid_size_backforeratio: 背景网格尺寸相对于前景的比例。
    """
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
        orig_path = os.path.join(original_folder, orig_fname) if orig_fname else None

        fg_path = os.path.join(foreground_folder, fg_fname)
        bg_path = os.path.join(background_folder, bg_fname)

        # 读取前景二值图，获取尺寸
        fg_bin_initial = cv2.imread(fg_path, cv2.IMREAD_GRAYSCALE)
        if fg_bin_initial is None:
            print(f"    警告: 无法读取前景图像 {fg_fname}，跳过。")
            continue
        h, w = fg_bin_initial.shape[:2]
        percentile_mask_full = np.zeros((h, w), dtype=np.uint8)

        # 提前加载原图（用于筛选时的灰度计算）
        orig_img = None
        orig_gray = None
        if orig_path is not None and os.path.exists(orig_path):
            orig_img = cv2.imread(orig_path)
            if orig_img is not None:
                if len(orig_img.shape) == 3:
                    orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
                else:
                    orig_gray = orig_img.copy()

        # 提取前景提示，同时填充分位数区域（白色，包括小区域和大区域）
        try:
            fg_pts, fg_lbls, fg_boxes = extract_foreground_prompts(
                fg_path,
                x_ratio=fg_x_ratio,
                fore_ratio=fg_ratio,
                dis_transf_perce_low=dis_transf_perce_low,
                dis_transf_per_high=dis_transf_per_high,
                x_lratio=x_lratio,
                x_hratio=x_hratio,
                percentile_mask_full=percentile_mask_full
            )
        except Exception as e:
            print(f"    前景提取失败 {fg_fname}: {e}")
            continue

        # 提取背景提示，同时填充分位数区域（灰色，包括小区域和大区域）
        try:
            bg_pts, bg_lbls = extract_background_prompts(
                bg_path, min_area=bg_min_area, margin=bg_margin,
                ratio=bg_ratio,
                dis_transf_perce_low=dis_transf_perce_low,
                dis_transf_per_high=dis_transf_per_high,
                x_lratio=x_lratio,
                x_hratio=x_hratio,
                percentile_mask_full=percentile_mask_full
            )
        except Exception as e:
            print(f"    背景提取失败 {bg_fname}: {e}")
            bg_pts, bg_lbls = [], []

        # ========== 确保每个连通区域至少有一个提示点 ==========
        fg_bin_for_dist = cv2.imread(fg_path, cv2.IMREAD_GRAYSCALE)
        if fg_bin_for_dist is not None:
            _, fg_bin_for_dist = cv2.threshold(fg_bin_for_dist, 127, 255, cv2.THRESH_BINARY)
            fg_dist_transform = cv2.distanceTransform(fg_bin_for_dist, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        else:
            fg_dist_transform = np.zeros_like(fg_bin_initial, dtype=np.float32)

        bg_bin_for_dist = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
        if bg_bin_for_dist is not None:
            _, bg_bin_for_dist = cv2.threshold(bg_bin_for_dist, 127, 255, cv2.THRESH_BINARY)
            bg_bin_for_dist[:bg_margin, :] = 0
            bg_bin_for_dist[-bg_margin:, :] = 0
            bg_bin_for_dist[:, :bg_margin] = 0
            bg_bin_for_dist[:, -bg_margin:] = 0
            bg_dist_transform = cv2.distanceTransform(bg_bin_for_dist, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        else:
            bg_dist_transform = np.zeros_like(fg_bin_initial, dtype=np.float32)

        fg_mask_full = (percentile_mask_full == 255).astype(np.uint8) * 255
        fg_pts, fg_lbls = ensure_points_in_regions(fg_mask_full, fg_pts, fg_dist_transform, 1)

        bg_mask_full = (percentile_mask_full == 128).astype(np.uint8) * 255
        bg_pts, bg_lbls = ensure_points_in_regions(bg_mask_full, bg_pts, bg_dist_transform, -1)

        # ===== 动态设置 grid_size_fore / grid_size_back =====
        # 基于 distance_transform_percentile 中前景连通区域面积最大的前 distranfirstnum 个区域的平均面积计算
        fg_mask_for_size = (percentile_mask_full == 255).astype(np.uint8) * 255
        grid_size_fore = compute_grid_size_from_mask(
            fg_mask_for_size,
            top_n=distranfirstnum,
            grid_size_fore_indice=grid_size_fore_indice
        )

        # 背景网格尺寸 = 前景网格尺寸 * 比例
        grid_size_back = int(grid_size_backforeratio * grid_size_fore)

        # ===== 构建区域标签、平均灰度和面积，用于距离筛选和网格筛选 =====
        labels_fg = np.zeros_like(percentile_mask_full, dtype=np.int32)
        region_avg_fg = {}
        region_area_fg = {}
        if orig_gray is not None and np.any(percentile_mask_full == 255):
            fg_mask_for_cc = (percentile_mask_full == 255).astype(np.uint8) * 255
            num_fg, labels_fg_temp, stats_fg, _ = cv2.connectedComponentsWithStats(fg_mask_for_cc, connectivity=8)
            labels_fg = labels_fg_temp
            for lab in range(1, num_fg):
                region_pixels = orig_gray[labels_fg == lab]
                if region_pixels.size > 0:
                    region_avg_fg[lab] = float(np.mean(region_pixels))
                else:
                    region_avg_fg[lab] = 0.0
                region_area_fg[lab] = int(stats_fg[lab, cv2.CC_STAT_AREA])

        labels_bg = np.zeros_like(percentile_mask_full, dtype=np.int32)
        region_avg_bg = {}
        region_area_bg = {}
        if orig_gray is not None and np.any(percentile_mask_full == 128):
            bg_mask_for_cc = (percentile_mask_full == 128).astype(np.uint8) * 255
            num_bg, labels_bg_temp, stats_bg, _ = cv2.connectedComponentsWithStats(bg_mask_for_cc, connectivity=8)
            labels_bg = labels_bg_temp
            for lab in range(1, num_bg):
                region_pixels = orig_gray[labels_bg == lab]
                if region_pixels.size > 0:
                    region_avg_bg[lab] = float(np.mean(region_pixels))
                else:
                    region_avg_bg[lab] = 0.0
                region_area_bg[lab] = int(stats_bg[lab, cv2.CC_STAT_AREA])

        # ===== 前景筛选 =====
        if orig_gray is not None:
            # 1. 全局面积前 distransf_area_select_ratio 过滤
            if grid_final > 0:
                print(f"    执行前景全局面积前 {distransf_area_select_ratio*100:.0f}% 过滤 (grid_final={grid_final})...")
                fg_pts = filter_points_by_grid_area(fg_pts, percentile_mask_full, grid_final,
                                                    is_foreground=True,
                                                    keep_ratio=distransf_area_select_ratio)
                fg_lbls = [1] * len(fg_pts)
                print(f"    过滤后前景点: {len(fg_pts)}")
            # 2. 网格条件筛选
            if grid_size_fore > 0 and fg_pts:
                print(f"    执行前景网格条件筛选 (grid_size={grid_size_fore}, 网格数量={int(w/grid_size_fore)}x{int(h/grid_size_fore)})...")
                fg_pts = filter_points_by_grid_conditions(
                    fg_pts, orig_gray, grid_size_fore, keep_low=True,
                    neighbour_l=neighbour_l, region_label_img=labels_fg,
                    region_area=region_area_fg, region_avg_gray=region_avg_fg,
                    first_condition_id=first_condition_id,
                    second_condition_id=second_condition_id,
                    final_condition_id=final_condition_id
                )
                fg_lbls = [1] * len(fg_pts)
                print(f"    网格条件筛选后前景点: {len(fg_pts)}")

            # ===== 背景筛选 =====
            # 1. 全局面积前 distransf_area_select_ratio 过滤
            if grid_final > 0:
                print(f"    执行背景全局面积前 {distransf_area_select_ratio*100:.0f}% 过滤 (grid_final={grid_final})...")
                bg_pts = filter_points_by_grid_area(bg_pts, percentile_mask_full, grid_final,
                                                    is_foreground=False,
                                                    keep_ratio=distransf_area_select_ratio)
                bg_lbls = [-1] * len(bg_pts)
                print(f"    过滤后背景点: {len(bg_pts)}")
            # 2. 网格条件筛选
            if grid_size_back > 0 and bg_pts:
                print(f"    执行背景网格条件筛选 (grid_size={grid_size_back}, 网格数量={int(w/grid_size_back)}x{int(h/grid_size_back)})...")
                bg_pts = filter_points_by_grid_conditions(
                    bg_pts, orig_gray, grid_size_back, keep_low=False,
                    neighbour_l=neighbour_l, region_label_img=labels_bg,
                    region_area=region_area_bg, region_avg_gray=region_avg_bg,
                    first_condition_id=first_condition_id,
                    second_condition_id=second_condition_id,
                    final_condition_id=final_condition_id
                )
                bg_lbls = [-1] * len(bg_pts)
                print(f"    网格条件筛选后背景点: {len(bg_pts)}")

            # ===== 跨类筛选 =====
            if dis_foreback > 0 and fg_pts and bg_pts:
                print(f"    执行跨类筛选 (dis_foreback={dis_foreback})...")
                fg_pts, bg_pts = filter_cross_fg_bg(fg_pts, bg_pts, orig_gray, dis_foreback, 0.8, neighbour_l)
                fg_lbls = [1] * len(fg_pts)
                bg_lbls = [-1] * len(bg_pts)
                print(f"    跨类筛选后前景点: {len(fg_pts)}, 背景点: {len(bg_pts)}")
        else:
            if grid_size_fore > 0 or grid_size_back > 0 or dis_foreback > 0 or grid_final > 0:
                print("    警告: 原图缺失，无法执行筛选，保留原始提示点。")

        # ========== 构建筛选后的距离变换区域掩膜（仅保留包含最终提示点的连通区域） ==========
        filtered_percentile_mask = np.zeros_like(percentile_mask_full)

        # 前景区域：保留包含最终前景点的连通区域
        if fg_pts:
            fg_binary = (percentile_mask_full == 255).astype(np.uint8) * 255
            if np.count_nonzero(fg_binary) > 0:
                num_fg_cc, labels_fg_cc, stats_fg_cc, _ = cv2.connectedComponentsWithStats(fg_binary, connectivity=8)
                for lab in range(1, num_fg_cc):
                    ys, xs = np.where(labels_fg_cc == lab)
                    if any((x, y) in fg_pts for y, x in zip(ys, xs)):
                        filtered_percentile_mask[labels_fg_cc == lab] = 255

        # 背景区域：保留包含最终背景点的连通区域
        if bg_pts:
            bg_binary = (percentile_mask_full == 128).astype(np.uint8) * 255
            if np.count_nonzero(bg_binary) > 0:
                num_bg_cc, labels_bg_cc, stats_bg_cc, _ = cv2.connectedComponentsWithStats(bg_binary, connectivity=8)
                for lab in range(1, num_bg_cc):
                    ys, xs = np.where(labels_bg_cc == lab)
                    if any((x, y) in bg_pts for y, x in zip(ys, xs)):
                        filtered_percentile_mask[labels_bg_cc == lab] = 128

        # 保存合并后的分位数区域图像（原始，未筛选）
        if distance_transform_percentile_dir is not None:
            os.makedirs(distance_transform_percentile_dir, exist_ok=True)
            out_percentile_path = os.path.join(distance_transform_percentile_dir, f"{base}.png")
            cv2.imwrite(out_percentile_path, percentile_mask_full)

            # 保存最终距离变换叠加提示点的图像（使用筛选后的区域）
            dt_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            dt_overlay[filtered_percentile_mask == 255] = (255, 255, 255)  # 前景白色
            dt_overlay[filtered_percentile_mask == 128] = (128, 128, 128)  # 背景灰色
            for pt in fg_pts:
                cv2.circle(dt_overlay, tuple(pt), 2, (0, 0, 255), -1)
            for pt in bg_pts:
                cv2.circle(dt_overlay, tuple(pt), 2, (0, 255, 0), -1)
            out_dt_overlay_path = os.path.join(output_folder, f"{base}_distance_transform_overlay.png")
            cv2.imwrite(out_dt_overlay_path, dt_overlay)

            # 保存带网格的版本
            if grid_size_fore > 0 or grid_size_back > 0:
                dt_overlay_grid = draw_grids(dt_overlay, grid_size_fore, grid_size_back)
                out_dt_overlay_grid = os.path.join(output_folder, f"{base}_distance_transform_overlay_grid.png")
                cv2.imwrite(out_dt_overlay_grid, dt_overlay_grid)

        # 后续可视化与文本输出
        all_points = fg_pts + bg_pts
        all_labels = fg_lbls + bg_lbls

        if orig_img is None:
            orig_img = cv2.imread(fg_path)
            if orig_img is None:
                orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
        if len(orig_img.shape) == 2:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

        img_overlay = orig_img.copy()
        for box in fg_boxes:
            cv2.rectangle(img_overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        for pt in fg_pts:
            cv2.circle(img_overlay, tuple(pt), 1, (0, 0, 255), -1)
        for pt in bg_pts:
            cv2.circle(img_overlay, tuple(pt), 1, (0, 255, 0), -1)
        out_overlay = os.path.join(output_folder, f"{base}_overlay.png")
        cv2.imwrite(out_overlay, img_overlay)

        # 保存带网格的 overlay
        if grid_size_fore > 0 or grid_size_back > 0:
            img_overlay_grid = draw_grids(img_overlay, grid_size_fore, grid_size_back)
            out_overlay_grid = os.path.join(output_folder, f"{base}_overlay_grid.png")
            cv2.imwrite(out_overlay_grid, img_overlay_grid)

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
                cv2.circle(mask_canvas, tuple(pt), 2, (0, 0, 255), -1)
            for pt in bg_pts:
                cv2.circle(mask_canvas, tuple(pt), 2, (0, 255, 0), -1)
            out_mask = os.path.join(output_folder, f"{base}_mask_overlay.png")
            cv2.imwrite(out_mask, mask_canvas)

            # 保存带网格的 mask_overlay
            if grid_size_fore > 0 or grid_size_back > 0:
                mask_overlay_grid = draw_grids(mask_canvas, grid_size_fore, grid_size_back)
                out_mask_grid = os.path.join(output_folder, f"{base}_mask_overlay_grid.png")
                cv2.imwrite(out_mask_grid, mask_overlay_grid)

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


def run_pipeline(input_folder, output_folder, temp_dir=None):
    """
    执行完整流水线：低通滤波 → 轮廓提取（前景/背景两套参数）→ 衬度提取 → 逻辑操作 → 小连通区域过滤 → AP生成
    所有中间文件保存在 temp_dir（默认输出文件夹下的 pipeline_temp），且不会被自动删除。
    """
    start_time = time.time()
    print(f"开始处理文件夹: {input_folder}")

    if temp_dir is None:
        temp_dir = os.path.join(output_folder, "pipeline_temp")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"临时目录（所有中间结果）: {temp_dir}")

    # 定义子目录
    lpf_dir = os.path.join(temp_dir, "lpf")
    contour_after_fore = os.path.join(temp_dir, "contour_after_fore")
    contour_morph_fore = os.path.join(temp_dir, "contour_morph_fore")
    contour_after_back = os.path.join(temp_dir, "contour_after_back")
    contour_morph_back = os.path.join(temp_dir, "contour_morph_back")

    contrast_fore = os.path.join(temp_dir, "contrast_fore")
    contrast_back = os.path.join(temp_dir, "contrast_back")
    forecoarse = os.path.join(temp_dir, "forecoarse")
    backcoarse = os.path.join(temp_dir, "backcoarse")
    distance_transform_percentile = os.path.join(temp_dir, "distance_transform_percentile")

    # 1. 低通滤波
    print("执行低通滤波...")
    success = process_lowpass_filter(input_folder, lpf_dir, r=1, r_noise=100)
    if not success:
        print("低通滤波失败，终止。")
        return False
    print(f"  低通滤波生成 {len(glob.glob(os.path.join(lpf_dir, '*')))} 个文件")

    # 2. 轮廓提取（分前景和背景两套参数）
    print("执行轮廓提取...")

    # ===== 前景轮廓提取参数 =====
    fg_x_multiplier = 0.2
    fg_morph_shape = 'rect'
    fg_morph_size = 2
    fg_morph_iter = 1

    # ===== 背景轮廓提取参数（可独立调整） =====
    bg_x_multiplier = -1
    bg_morph_shape = 'ellipse'
    bg_morph_size = 4
    bg_morph_iter = 1

    # 前景轮廓提取
    process_contour_folder(
        lpf_dir, contour_after_fore, contour_morph_fore,
        0.9, 0.8,
        fg_x_multiplier, fg_morph_shape, fg_morph_size, fg_morph_iter,
        output_before_dir=None
    )
    print(f"  前景轮廓提取完成，形态学结果 {len(glob.glob(os.path.join(contour_morph_fore, '*')))} 个文件")

    # 背景轮廓提取
    process_contour_folder(
        lpf_dir, contour_after_back, contour_morph_back,
        0.9, 0.8,
        bg_x_multiplier, bg_morph_shape, bg_morph_size, bg_morph_iter,
        output_before_dir=None
    )
    print(f"  背景轮廓提取完成，形态学结果 {len(glob.glob(os.path.join(contour_morph_back, '*')))} 个文件")

    # 3. 衬度提取
    print("执行衬度提取...")
    process_contrast_images(lpf_dir, contrast_fore, contrast_back,
                            1.0, 1.3, invert_fore=True, invert_back=False)
    print(f"  衬度提取完成，前景 {len(glob.glob(os.path.join(contrast_fore, '*')))} 个文件")
    # # 3. 衬度提取
    # print("执行衬度提取...")
    # process_contrast_images(lpf_dir, contrast_fore, contrast_back,
    #                         1.0, 1.1, invert_fore=True, invert_back=False)
    # print(f"  衬度提取完成，前景 {len(glob.glob(os.path.join(contrast_fore, '*')))} 个文件")

    # 4. 逻辑操作
    print("执行逻辑操作...")
    prefix_len = 2

    # 前景：contrast_fore - 前景轮廓形态学结果 = forecoarse
    count1, _ = logic_operation_folder(contrast_fore, contour_morph_fore, "差集(A-B)",
                                       forecoarse, prefix_len)
    print(f"  前景差集生成 {count1} 个文件")

    # 背景：contrast_back - 背景轮廓形态学结果 = backcoarse
    count2, _ = logic_operation_folder(contrast_back, contour_morph_back, "差集(A-B)",
                                       backcoarse, prefix_len)
    print(f"  背景差集生成 {count2} 个文件")

    # 5. 过滤前景 coarse 的小连通区域
    print("过滤前景 coarse 的小连通区域...")
    coarse_filtered = os.path.join(temp_dir, "coarse_filtered")
    filter_small_connected(forecoarse, coarse_filtered, x_val=3.0)

    filtered_files = glob.glob(os.path.join(coarse_filtered, '*'))
    print(f"  过滤后前景图像数: {len(filtered_files)}")
    if len(filtered_files) == 0:
        print("  ⚠️ 警告：过滤后无图像，将 forecoarse 复制到 coarse_filtered")
        fore_files = glob.glob(os.path.join(forecoarse, '*'))
        for f in fore_files:
            shutil.copy2(f, coarse_filtered)
        filtered_files = glob.glob(os.path.join(coarse_filtered, '*'))
        print(f"  复制后前景图像数: {len(filtered_files)}")

    foreground_ap = coarse_filtered

    # 6. AP Generate
    print("执行 AP Generate...")
    ap_output = output_folder
    os.makedirs(ap_output, exist_ok=True)
    os.makedirs(distance_transform_percentile, exist_ok=True)

    basenames = process_ap_images(
        original_folder=input_folder,
        foreground_folder=foreground_ap,
        background_folder=backcoarse,
        output_folder=ap_output,
        fg_x_ratio=0.2,
        fg_ratio=0.05,
        bg_min_area=100,
        bg_margin=5,
        bg_ratio=0.01,
        prefix_len=2,
        dis_transf_perce_low=10,
        dis_transf_per_high=80,
        x_lratio=1,
        x_hratio=5,
        distance_transform_percentile_dir=distance_transform_percentile,
        x_multiplier=0.2,
        dis_foreback=5,
        grid_final=1000,
        neighbour_l=3,
        distransf_area_select_ratio=0.8,
        distranfirstnum=5,               # 新增：取面积最大的前5个距离变换连通区域
        first_condition_id=1,
        second_condition_id=2,
        final_condition_id=2,
        grid_size_fore_indice=1,       # # grid_size = grid_size_fore_indice * int(np.sqrt(avg_area))，默认为4
        grid_size_backforeratio=0.5    # grid_size_back = int(grid_size_backforeratio * grid_size_fore)，默认为1.5

    )

#         # 等级1：同时满足条件①面积最大、②区域平均灰度最优、③邻域灰度最优
#         # 等级2：同时满足 first_condition_id 和 second_condition_id 指定的两个条件
#         #        （条件编号：1=面积最大，2=区域平均灰度最优，3=邻域灰度最优）
#         # 等级3：满足 final_condition_id 指定的单个条件
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
