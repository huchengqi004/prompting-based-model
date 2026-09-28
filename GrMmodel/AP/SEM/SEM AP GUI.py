import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import glob
import shutil
import threading

# ==================== 纹理提取核心算法 ====================
def auto_canny_threshold_texture(gray_img, high_percentile=0.9, low_percentile=0.5):
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = np.uint8(np.clip(grad_mag, 0, 255))
    non_zero = grad_mag[grad_mag > 0]
    if len(non_zero) == 0:
        return 0, 0, grad_mag
    sorted_mag = np.sort(non_zero)
    n = len(sorted_mag)
    high_idx = min(int(high_percentile * n), n - 1)
    low_idx  = min(int(low_percentile * n), n - 1)
    return int(sorted_mag[low_idx]), int(sorted_mag[high_idx]), grad_mag

def gradient_info(img_gray):
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)
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
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = np.uint8(np.clip(grad_mag, 0, 255))
    non_zero = grad_mag[grad_mag > 0]
    if len(non_zero) == 0:
        return 0, 0, grad_mag
    sorted_mag = np.sort(non_zero)
    n = len(sorted_mag)
    high_idx = min(int(high_percentile * n), n - 1)
    low_idx  = min(int(low_percentile * n), n - 1)
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
    keep_count = 0
    for cnt in contours:
        if cv2.arcLength(cnt, False) > threshold:
            cv2.drawContours(filtered, [cnt], -1, 255, 1)
            keep_count += 1
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
        msg = f"在 {input_dir} 中未找到任何图像。"
        if log_callback: log_callback(msg)
        return

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        log_msg = f"[{idx}/{total}] 轮廓处理: {base}"
        if log_callback: log_callback(log_msg)

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

        # 形态学处理
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
        msg = f"在 {input_dir} 中未找到任何图像。"
        if log_callback: log_callback(msg)
        return

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        log_msg = f"[{idx}/{total}] 衬度处理: {base}"
        if log_callback: log_callback(log_msg)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        otsu_th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        new_th = np.clip(ratio_fore * otsu_th, 0, 255)
        _, binary = cv2.threshold(img, new_th, 255, cv2.THRESH_BINARY)
        if invert_fore:
            binary = cv2.bitwise_not(binary)
        cv2.imwrite(os.path.join(fore_out_dir, base+".png"), binary)

        new_th = np.clip(ratio_back * otsu_th, 0, 255)
        _, binary = cv2.threshold(img, new_th, 255, cv2.THRESH_BINARY)
        if invert_back:
            binary = cv2.bitwise_not(binary)
        cv2.imwrite(os.path.join(back_out_dir, base+".png"), binary)

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
        msg = f"在 {input_dir} 中未找到任何图像。"
        if log_callback: log_callback(msg)
        return

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        log_msg = f"[{idx}/{total}] 纹理处理: {base}"
        if log_callback: log_callback(log_msg)

        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fg_tex, fg_conn = extract_texture_connected(gray, fg_params)
        bg_tex, bg_conn = extract_texture_connected(gray, bg_params)

        cv2.imwrite(os.path.join(output_fore_conn_dir, base+".png"), fg_conn)
        cv2.imwrite(os.path.join(output_back_conn_dir, base+".png"), bg_conn)
        cv2.imwrite(os.path.join(output_fore_tex_dir, base+".png"), fg_tex)
        cv2.imwrite(os.path.join(output_back_tex_dir, base+".png"), bg_tex)

# ==================== 逻辑操作 ====================
def logic_operation_folder(folderA, folderB, operation, output_dir, prefix_len=0, log_callback=None):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    filesA = [f for f in os.listdir(folderA) if f.lower().endswith(exts)]
    filesB = [f for f in os.listdir(folderB) if f.lower().endswith(exts)]
    if not filesA or not filesB:
        if log_callback: log_callback("某个文件夹中没有图像文件。")
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
            if log_callback: log_callback("没有相同前缀的文件！")
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
            if log_callback: log_callback("没有同名图像文件！")
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
        msg = f"在 {input_dir} 中未找到任何图像。"
        if log_callback: log_callback(msg)
        return False

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        log_msg = f"[{idx}/{total}] 低通滤波: {base}"
        if log_callback: log_callback(log_msg)

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

    bg_files = [f for f in os.listdir(background_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
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
            print(f"  前景提取失败: {e}")
            continue

        try:
            bg_pts, bg_lbls = extract_background_prompts(
                bg_path, min_area=bg_min_area, margin=bg_margin,
                ratio=bg_ratio, top_x=bg_top_x
            )
        except Exception as e:
            print(f"  背景提取失败: {e}")
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

# ==================== 结果查看器 ====================
class TextureResultViewer:
    def __init__(self, master, app):
        self.app = app
        self.window = None
        self.labels = {}
        self.photo_refs = []

    def show(self):
        if self.window is None or not self.window.winfo_exists():
            self.window = tk.Toplevel(self.app.master)
            self.window.title("纹理 - 连接图结果")
            self.window.geometry("1000x600")
            self.window.minsize(600, 400)
            self.window.protocol("WM_DELETE_WINDOW", self.on_close)
            self.build_ui()
        else:
            self.window.deiconify()
            self.window.lift()

    def build_ui(self):
        info_frame = tk.Frame(self.window)
        info_frame.pack(fill=tk.X, pady=5)
        self.file_label = tk.Label(info_frame, text="", font=("Arial", 12))
        self.file_label.pack(side=tk.LEFT, padx=10)
        self.param_label = tk.Label(info_frame, text="", font=("Arial", 9), fg="blue")
        self.param_label.pack(side=tk.RIGHT, padx=10)

        display_frame = tk.Frame(self.window)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        fg_subframe = tk.LabelFrame(display_frame, text="前景连接")
        fg_subframe.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        fg_label = tk.Label(fg_subframe, bg="lightgray", relief=tk.RIDGE)
        fg_label.pack(fill=tk.BOTH, expand=True)
        self.labels['fg_conn'] = fg_label

        bg_subframe = tk.LabelFrame(display_frame, text="背景连接")
        bg_subframe.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        bg_label = tk.Label(bg_subframe, bg="lightgray", relief=tk.RIDGE)
        bg_label.pack(fill=tk.BOTH, expand=True)
        self.labels['bg_conn'] = bg_label

        nav_frame = tk.Frame(self.window)
        nav_frame.pack(fill=tk.X, pady=5)
        btn_prev = tk.Button(nav_frame, text="◀ 上一张", command=self.app.prev_texture_image, width=10)
        btn_prev.pack(side=tk.LEFT, padx=10)
        self.index_label = tk.Label(nav_frame, text="0 / 0", font=("Arial", 10))
        self.index_label.pack(side=tk.LEFT, padx=20)
        btn_next = tk.Button(nav_frame, text="下一张 ▶", command=self.app.next_texture_image, width=10)
        btn_next.pack(side=tk.LEFT, padx=10)

        self.window.bind("<Configure>", self.on_resize)

    def on_resize(self, event=None):
        if hasattr(self, 'current_data'):
            self.update_images(self.current_data)

    def on_close(self):
        self.window.destroy()
        self.window = None

    def update(self, data, idx, total, filename, param_info=""):
        self.current_data = data
        self.file_label.config(text=f"当前图像: {filename}")
        self.index_label.config(text=f"{idx+1} / {total}")
        self.param_label.config(text=param_info)
        self.update_images(data)

    def update_images(self, data):
        images = {
            'fg_conn': data['fg_conn'],
            'bg_conn': data['bg_conn']
        }
        for key, img_array in images.items():
            label = self.labels.get(key)
            if label is None or img_array is None:
                continue
            w = label.winfo_width()
            h = label.winfo_height()
            if w <= 1 or h <= 1:
                win_w = self.window.winfo_width()
                win_h = self.window.winfo_height()
                w = (win_w - 30) // 2 if win_w > 100 else 300
                h = win_h - 100 if win_h > 100 else 200
            pil_img = Image.fromarray(img_array)
            pil_img.thumbnail((w-10, h-10), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            label.config(image=photo)
            label.image = photo

class ContourResultViewer:
    def __init__(self, master, app):
        self.app = app
        self.window = None
        self.labels = {}
        self.photo_refs = []

    def show(self):
        if self.window is None or not self.window.winfo_exists():
            self.window = tk.Toplevel(self.app.master)
            self.window.title("轮廓 - 形态学处理结果")
            self.window.geometry("1000x600")
            self.window.minsize(600, 400)
            self.window.protocol("WM_DELETE_WINDOW", self.on_close)
            self.build_ui()
        else:
            self.window.deiconify()
            self.window.lift()

    def build_ui(self):
        info_frame = tk.Frame(self.window)
        info_frame.pack(fill=tk.X, pady=5)
        self.file_label = tk.Label(info_frame, text="", font=("Arial", 12))
        self.file_label.pack(side=tk.LEFT, padx=10)
        self.param_label = tk.Label(info_frame, text="", font=("Arial", 9), fg="blue")
        self.param_label.pack(side=tk.RIGHT, padx=10)

        display_frame = tk.Frame(self.window)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_subframe = tk.LabelFrame(display_frame, text="原始过滤边缘")
        left_subframe.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_label = tk.Label(left_subframe, bg="lightgray", relief=tk.RIDGE)
        left_label.pack(fill=tk.BOTH, expand=True)
        self.labels['original'] = left_label

        right_subframe = tk.LabelFrame(display_frame, text="形态学处理边缘")
        right_subframe.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        right_label = tk.Label(right_subframe, bg="lightgray", relief=tk.RIDGE)
        right_label.pack(fill=tk.BOTH, expand=True)
        self.labels['morph'] = right_label

        nav_frame = tk.Frame(self.window)
        nav_frame.pack(fill=tk.X, pady=5)
        btn_prev = tk.Button(nav_frame, text="◀ 上一张", command=self.app.prev_contour_image, width=10)
        btn_prev.pack(side=tk.LEFT, padx=10)
        self.index_label = tk.Label(nav_frame, text="0 / 0", font=("Arial", 10))
        self.index_label.pack(side=tk.LEFT, padx=20)
        btn_next = tk.Button(nav_frame, text="下一张 ▶", command=self.app.next_contour_image, width=10)
        btn_next.pack(side=tk.LEFT, padx=10)

        self.window.bind("<Configure>", self.on_resize)

    def on_resize(self, event=None):
        if hasattr(self, 'current_data'):
            self.update_images(self.current_data)

    def on_close(self):
        self.window.destroy()
        self.window = None

    def update(self, data, idx, total, filename, param_info=""):
        self.current_data = data
        self.file_label.config(text=f"当前图像: {filename}")
        self.index_label.config(text=f"{idx+1} / {total}")
        self.param_label.config(text=param_info)
        self.update_images(data)

    def update_images(self, data):
        images = {
            'original': data['original'],
            'morph': data['morph']
        }
        for key, img_array in images.items():
            label = self.labels.get(key)
            if label is None or img_array is None:
                continue
            w = label.winfo_width()
            h = label.winfo_height()
            if w <= 1 or h <= 1:
                win_w = self.window.winfo_width()
                win_h = self.window.winfo_height()
                w = (win_w - 30) // 2 if win_w > 100 else 300
                h = win_h - 100 if win_h > 100 else 200
            pil_img = Image.fromarray(img_array)
            pil_img.thumbnail((w-10, h-10), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            label.config(image=photo)
            label.image = photo

class ContrastMorphViewer:
    def __init__(self, master, app):
        self.app = app
        self.window = None
        self.labels = {}
        self.photo_refs = []

    def show(self):
        if self.window is None or not self.window.winfo_exists():
            self.window = tk.Toplevel(self.app.master)
            self.window.title("衬度 - 形态学处理结果")
            self.window.geometry("1200x700")
            self.window.minsize(800, 500)
            self.window.protocol("WM_DELETE_WINDOW", self.on_close)
            self.build_ui()
        else:
            self.window.deiconify()
            self.window.lift()

    def build_ui(self):
        info_frame = tk.Frame(self.window)
        info_frame.pack(fill=tk.X, pady=5)
        self.file_label = tk.Label(info_frame, text="", font=("Arial", 12))
        self.file_label.pack(side=tk.LEFT, padx=10)
        self.param_label = tk.Label(info_frame, text="", font=("Arial", 9), fg="blue")
        self.param_label.pack(side=tk.RIGHT, padx=10)

        display_frame = tk.Frame(self.window)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        titles = [("前景原始", "fore_orig"), ("前景形态", "fore_morph"),
                  ("背景原始", "back_orig"), ("背景形态", "back_morph")]
        for i, (title, key) in enumerate(titles):
            subframe = tk.LabelFrame(display_frame, text=title)
            row, col = divmod(i, 2)
            subframe.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            label = tk.Label(subframe, bg="lightgray", relief=tk.RIDGE)
            label.pack(fill=tk.BOTH, expand=True)
            self.labels[key] = label

        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_rowconfigure(1, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_columnconfigure(1, weight=1)

        nav_frame = tk.Frame(self.window)
        nav_frame.pack(fill=tk.X, pady=5)
        btn_prev = tk.Button(nav_frame, text="◀ 上一张", command=self.app.prev_contrast_image, width=10)
        btn_prev.pack(side=tk.LEFT, padx=10)
        self.index_label = tk.Label(nav_frame, text="0 / 0", font=("Arial", 10))
        self.index_label.pack(side=tk.LEFT, padx=20)
        btn_next = tk.Button(nav_frame, text="下一张 ▶", command=self.app.next_contrast_image, width=10)
        btn_next.pack(side=tk.LEFT, padx=10)

        self.window.bind("<Configure>", self.on_resize)

    def on_resize(self, event=None):
        if hasattr(self, 'current_data'):
            self.update_images(self.current_data)

    def on_close(self):
        self.window.destroy()
        self.window = None

    def update(self, data, idx, total, filename, param_info=""):
        self.current_data = data
        self.file_label.config(text=f"当前图像: {filename}")
        self.index_label.config(text=f"{idx+1} / {total}")
        self.param_label.config(text=param_info)
        self.update_images(data)

    def update_images(self, data):
        images = {
            'fore_orig': data['fore_orig'],
            'fore_morph': data['fore_morph'],
            'back_orig': data['back_orig'],
            'back_morph': data['back_morph']
        }
        for key, img_array in images.items():
            label = self.labels.get(key)
            if label is None or img_array is None:
                continue
            w = label.winfo_width()
            h = label.winfo_height()
            if w <= 1 or h <= 1:
                win_w = self.window.winfo_width()
                win_h = self.window.winfo_height()
                w = (win_w - 30) // 2 if win_w > 100 else 300
                h = (win_h - 100) // 2 if win_h > 100 else 200
            pil_img = Image.fromarray(img_array)
            pil_img.thumbnail((w-10, h-10), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            label.config(image=photo)
            label.image = photo

# ==================== AP 结果查看器 ====================
class APResultViewer:
    def __init__(self, master, app):
        self.app = app
        self.window = None
        self.labels = {}
        self.photo_refs = []

    def show(self):
        if self.window is None or not self.window.winfo_exists():
            self.window = tk.Toplevel(self.app.master)
            self.window.title("AP Generate - 提示叠加结果")
            self.window.geometry("1200x700")
            self.window.minsize(800, 500)
            self.window.protocol("WM_DELETE_WINDOW", self.on_close)
            self.build_ui()
        else:
            self.window.deiconify()
            self.window.lift()

    def build_ui(self):
        info_frame = tk.Frame(self.window)
        info_frame.pack(fill=tk.X, pady=5)
        self.file_label = tk.Label(info_frame, text="", font=("Arial", 12))
        self.file_label.pack(side=tk.LEFT, padx=10)
        self.param_label = tk.Label(info_frame, text="", font=("Arial", 9), fg="blue")
        self.param_label.pack(side=tk.RIGHT, padx=10)

        display_frame = tk.Frame(self.window)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        titles = [("原图叠加 (红色点+黄色框, 绿色点)", "overlay"),
                  ("蒙版叠加 (白色前景+灰色背景)", "mask")]
        for i, (title, key) in enumerate(titles):
            subframe = tk.LabelFrame(display_frame, text=title)
            row, col = divmod(i, 2)
            subframe.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            label = tk.Label(subframe, bg="lightgray", relief=tk.RIDGE)
            label.pack(fill=tk.BOTH, expand=True)
            self.labels[key] = label

        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_rowconfigure(1, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_columnconfigure(1, weight=1)

        nav_frame = tk.Frame(self.window)
        nav_frame.pack(fill=tk.X, pady=5)
        btn_prev = tk.Button(nav_frame, text="◀ 上一张", command=self.app.prev_ap_image, width=10)
        btn_prev.pack(side=tk.LEFT, padx=10)
        self.index_label = tk.Label(nav_frame, text="0 / 0", font=("Arial", 10))
        self.index_label.pack(side=tk.LEFT, padx=20)
        btn_next = tk.Button(nav_frame, text="下一张 ▶", command=self.app.next_ap_image, width=10)
        btn_next.pack(side=tk.LEFT, padx=10)

        self.window.bind("<Configure>", self.on_resize)

    def on_resize(self, event=None):
        if hasattr(self, 'current_data'):
            self.update_images(self.current_data)

    def on_close(self):
        self.window.destroy()
        self.window = None

    def update(self, data, idx, total, filename, param_info=""):
        self.current_data = data
        self.file_label.config(text=f"当前图像: {filename}")
        self.index_label.config(text=f"{idx+1} / {total}")
        self.param_label.config(text=param_info)
        self.update_images(data)

    def update_images(self, data):
        images = {
            'overlay': data['overlay'],
            'mask': data['mask']
        }
        for key, img_array in images.items():
            label = self.labels.get(key)
            if label is None or img_array is None:
                continue
            w = label.winfo_width()
            h = label.winfo_height()
            if w <= 1 or h <= 1:
                win_w = self.window.winfo_width()
                win_h = self.window.winfo_height()
                w = (win_w - 30) // 2 if win_w > 100 else 300
                h = (win_h - 100) // 2 if win_h > 100 else 200
            pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            pil_img.thumbnail((w-10, h-10), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            label.config(image=photo)
            label.image = photo

# ==================== 主应用 ====================
class TextureExtractionApp:
    def __init__(self, master):
        self.master = master
        master.title("图像处理工具集")
        master.geometry("900x750")
        master.resizable(True, True)

        # 路径变量
        self.lowpass_folder_path = tk.StringVar()
        self.texture_folder_path = tk.StringVar()
        self.texture_image_files = []
        self.texture_current_idx = 0
        self.texture_cache = {}

        self.contour_folder_path = tk.StringVar()
        self.contour_image_files = []
        self.contour_current_idx = 0
        self.contour_cache = {}
        self.contour_output_base = tk.StringVar(value=os.path.dirname(os.path.abspath(__file__)))

        self.contrast_folder_path = tk.StringVar()
        self.contrast_image_files = []
        self.contrast_current_idx = 0
        self.contrast_cache = {}
        self.contrast_output_base = tk.StringVar(value=os.path.dirname(os.path.abspath(__file__)))

        # 逻辑操作路径
        self.logic_step1_A = tk.StringVar()
        self.logic_step1_B = tk.StringVar()
        self.logic_step2_A = tk.StringVar()
        self.logic_step2_B = tk.StringVar()
        self.logic_back_A = tk.StringVar()
        self.logic_back_B = tk.StringVar()
        self.logic_step1_out = tk.StringVar()
        self.logic_step2_out = tk.StringVar()
        self.logic_back_out = tk.StringVar()
        self.logic_prefix_len = tk.IntVar(value=2)

        # ===== AP Generate 变量 =====
        self.ap_original_folder = tk.StringVar()
        self.ap_foreground_folder = tk.StringVar()
        self.ap_background_folder = tk.StringVar()
        self.ap_output_folder = tk.StringVar()
        self.ap_fg_x_ratio = tk.DoubleVar(value=0.2)
        self.ap_fg_ratio = tk.DoubleVar(value=0.05)
        self.ap_bg_min_area = tk.IntVar(value=200)
        self.ap_bg_margin = tk.IntVar(value=5)
        self.ap_bg_ratio = tk.DoubleVar(value=0.05)
        self.ap_bg_top_x = tk.IntVar(value=3)
        self.ap_prefix_len = tk.IntVar(value=2)
        self.ap_image_files = []
        self.ap_current_idx = 0
        self.ap_cache = {}

        # 查看器
        self.texture_viewer = TextureResultViewer(master, self)
        self.contour_viewer = ContourResultViewer(master, self)
        self.contrast_viewer = ContrastMorphViewer(master, self)
        self.ap_viewer = APResultViewer(master, self)

        self.build_ui()
        self.show_lowpass_tab()

    def build_ui(self):
        top_frame = tk.Frame(self.master, bg="#e0e0e0")
        top_frame.pack(fill=tk.X, side=tk.TOP)

        tk.Button(top_frame, text="▶ Run", command=self.run_all,
                  bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=8).pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(top_frame, text="低通滤波", command=self.show_lowpass_tab, width=15).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top_frame, text="纹理提取", command=self.show_texture_tab, width=15).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top_frame, text="轮廓提取", command=self.show_contour_tab, width=15).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top_frame, text="衬度提取", command=self.show_contrast_tab, width=15).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top_frame, text="逻辑操作", command=self.show_logic_tab, width=15).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top_frame, text="AP generate", command=self.show_ap_tab, width=15).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top_frame, text="其他功能", state=tk.DISABLED, width=15).pack(side=tk.LEFT, padx=5, pady=5)

        self.main_container = tk.Frame(self.master)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.lowpass_frame = tk.Frame(self.main_container)
        self.build_lowpass_tab()
        self.texture_frame = tk.Frame(self.main_container)
        self.build_texture_tab()
        self.contour_frame = tk.Frame(self.main_container)
        self.build_contour_tab()
        self.contrast_frame = tk.Frame(self.main_container)
        self.build_contrast_tab()
        self.logic_frame = tk.Frame(self.main_container)
        self.build_logic_tab()
        self.ap_frame = tk.Frame(self.main_container)
        self.build_ap_tab()

    def hide_all_frames(self):
        for frame in [self.lowpass_frame, self.texture_frame, self.contour_frame,
                      self.contrast_frame, self.logic_frame, self.ap_frame]:
            frame.pack_forget()

    def show_lowpass_tab(self):
        self.hide_all_frames()
        self.lowpass_frame.pack(fill=tk.BOTH, expand=True)

    def show_texture_tab(self):
        self.hide_all_frames()
        self.texture_frame.pack(fill=tk.BOTH, expand=True)

    def show_contour_tab(self):
        self.hide_all_frames()
        self.contour_frame.pack(fill=tk.BOTH, expand=True)

    def show_contrast_tab(self):
        self.hide_all_frames()
        self.contrast_frame.pack(fill=tk.BOTH, expand=True)

    def show_logic_tab(self):
        self.hide_all_frames()
        self.logic_frame.pack(fill=tk.BOTH, expand=True)

    def show_ap_tab(self):
        self.hide_all_frames()
        self.ap_frame.pack(fill=tk.BOTH, expand=True)

    # ---------- 低通滤波 ----------
    def build_lowpass_tab(self):
        parent = self.lowpass_frame
        common_frame = tk.LabelFrame(parent, text="公共设置", padx=5, pady=5)
        common_frame.pack(fill=tk.X, pady=5)
        tk.Label(common_frame, text="图片文件夹：").grid(row=0, column=0, sticky='w', padx=5)
        tk.Entry(common_frame, textvariable=self.lowpass_folder_path, width=50).grid(row=0, column=1, padx=5)
        tk.Button(common_frame, text="浏览...", command=self.browse_lowpass_folder).grid(row=0, column=2, padx=5)

        param_frame = tk.LabelFrame(parent, text="滤波参数", padx=5, pady=5)
        param_frame.pack(fill=tk.X, pady=5)
        tk.Label(param_frame, text="r (低通半径):").grid(row=0, column=0, sticky='e', padx=5, pady=2)
        self.lowpass_r = tk.IntVar(value=1)
        tk.Entry(param_frame, textvariable=self.lowpass_r, width=8).grid(row=0, column=1, sticky='w', padx=5)
        tk.Label(param_frame, text="(默认: 1)", fg="gray", font=("Arial", 8)).grid(row=0, column=2, sticky='w', padx=2)
        tk.Label(param_frame, text="r_noise (滤噪半径):").grid(row=0, column=3, sticky='e', padx=5, pady=2)
        self.lowpass_r_noise = tk.IntVar(value=100)
        tk.Entry(param_frame, textvariable=self.lowpass_r_noise, width=8).grid(row=0, column=4, sticky='w', padx=5)
        tk.Label(param_frame, text="(默认: 100)", fg="gray", font=("Arial", 8)).grid(row=0, column=5, sticky='w', padx=2)

        btn_frame = tk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="执行低通滤波", command=self.run_lowpass,
                  bg="#2196F3", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.lowpass_status = tk.Label(parent, text="就绪", anchor='w')
        self.lowpass_status.pack(fill=tk.X, pady=5)

    def browse_lowpass_folder(self):
        folder = filedialog.askdirectory(title="选择包含图片的文件夹")
        if folder:
            self.lowpass_folder_path.set(folder)
            self.texture_folder_path.set(folder)

    def run_lowpass(self):
        input_dir = self.lowpass_folder_path.get().strip()
        if not input_dir:
            messagebox.showerror("错误", "请先选择图片文件夹。")
            return
        try:
            r = self.lowpass_r.get()
            r_noise = self.lowpass_r_noise.get()
        except:
            messagebox.showerror("参数错误", "请输入有效数值。")
            return
        output_dir = input_dir + "-FFT"
        self.lowpass_status.config(text="正在处理，请稍候...")
        success = process_lowpass_filter(input_dir, output_dir, r, r_noise,
                                         log_callback=lambda msg: self.lowpass_status.config(text=msg))
        if success:
            self.contour_folder_path.set(output_dir)
            self.contrast_folder_path.set(output_dir)
            self.lowpass_status.config(text=f"低通滤波完成，已更新轮廓和衬度输入路径")

    # ---------- 纹理提取 ----------
    def build_texture_tab(self):
        parent = self.texture_frame
        common_frame = tk.LabelFrame(parent, text="公共设置", padx=5, pady=5)
        common_frame.pack(fill=tk.X, pady=5)
        tk.Label(common_frame, text="图片文件夹：").grid(row=0, column=0, sticky='w', padx=5)
        tk.Entry(common_frame, textvariable=self.texture_folder_path, width=50).grid(row=0, column=1, padx=5)
        tk.Button(common_frame, text="浏览...", command=self.browse_texture_folder).grid(row=0, column=2, padx=5)
        tk.Button(common_frame, text="加载图像列表", command=self.load_texture_image_list).grid(row=0, column=3, padx=5)

        param_container = tk.Frame(parent)
        param_container.pack(fill=tk.BOTH, expand=True, pady=5)
        self.foreground_frame = tk.LabelFrame(param_container, text="用于提取前景", padx=5, pady=5)
        self.foreground_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        self.background_frame = tk.LabelFrame(param_container, text="用于获取背景", padx=5, pady=5)
        self.background_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5,0))

        self.fg_vars = self.build_texture_param_widgets(self.foreground_frame, "foreground")
        self.bg_vars = self.build_texture_param_widgets(self.background_frame, "background")

        control_frame = tk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="形态学操作", command=self.apply_texture_morphology, bg="#FF9800", fg="white", width=12).pack(side=tk.LEFT, padx=5)
        self.texture_status = tk.Label(parent, text="就绪", anchor='w')
        self.texture_status.pack(fill=tk.X, pady=5)

    def build_texture_param_widgets(self, parent, mode):
        defaults = {
            'foreground': {
                'percent_high': 0.9, 'percent_low': 0.8, 'contour_tex_ratio': 0,
                'min_contour_length': 0, 'tex_density1_enable': True,
                'tex_density_win1': 40, 'tex_density_thr1': 22,
                'tex_density2_enable': True, 'tex_density_win2': 16,
                'tex_density_thr2': 10, 'symmetry_half_width': 2,
                'symmetry_threshold_deg': 135,
                'morph_shape': 'rect', 'morph_size': 2, 'morph_iter': 1
            },
            'background': {
                'percent_high': 0.9, 'percent_low': 0.7, 'contour_tex_ratio': 0,
                'min_contour_length': 0, 'tex_density1_enable': True,
                'tex_density_win1': 40, 'tex_density_thr1': 20,
                'tex_density2_enable': True, 'tex_density_win2': 16,
                'tex_density_thr2': 5, 'symmetry_half_width': 2,
                'symmetry_threshold_deg': 135,
                'morph_shape': 'rect', 'morph_size': 2, 'morph_iter': 1
            }
        }
        d = defaults[mode]
        vars_dict = {}
        param_frame = tk.Frame(parent)
        param_frame.pack(fill=tk.X, pady=2)
        fields = [
            ("Canny高百分位:", tk.DoubleVar, d['percent_high'], 'percent_high'),
            ("Canny低百分位:", tk.DoubleVar, d['percent_low'], 'percent_low'),
            ("纹理点比例:", tk.DoubleVar, d['contour_tex_ratio'], 'contour_tex_ratio'),
            ("最小边缘长度:", tk.IntVar, d['min_contour_length'], 'min_contour_length'),
            ("密度过滤1启用:", tk.BooleanVar, d['tex_density1_enable'], 'tex_density1_enable'),
            ("窗口1:", tk.IntVar, d['tex_density_win1'], 'tex_density_win1'),
            ("阈值1:", tk.IntVar, d['tex_density_thr1'], 'tex_density_thr1'),
            ("密度过滤2启用:", tk.BooleanVar, d['tex_density2_enable'], 'tex_density2_enable'),
            ("窗口2:", tk.IntVar, d['tex_density_win2'], 'tex_density_win2'),
            ("阈值2:", tk.IntVar, d['tex_density_thr2'], 'tex_density_thr2'),
            ("对称半宽:", tk.IntVar, d['symmetry_half_width'], 'symmetry_half_width'),
            ("对称角度阈值(°):", tk.DoubleVar, d['symmetry_threshold_deg'], 'symmetry_threshold_deg'),
        ]
        row = 0
        for label_text, var_type, default_val, key in fields:
            tk.Label(param_frame, text=label_text).grid(row=row, column=0, sticky='e', padx=5, pady=1)
            var = var_type(value=default_val)
            if isinstance(var, tk.BooleanVar):
                tk.Checkbutton(param_frame, variable=var).grid(row=row, column=1, sticky='w', padx=5, pady=1)
            else:
                tk.Entry(param_frame, textvariable=var, width=8).grid(row=row, column=1, sticky='w', padx=5, pady=1)
            vars_dict[key] = var
            row += 1

        morph_frame = tk.LabelFrame(parent, text="连接图膨胀参数", padx=5, pady=5)
        morph_frame.pack(fill=tk.X, pady=5)
        tk.Label(morph_frame, text="kernel形状:").grid(row=0, column=0, sticky='e', padx=5)
        shape_var = tk.StringVar(value=d['morph_shape'])
        ttk.Combobox(morph_frame, textvariable=shape_var,
                     values=["rect", "ellipse", "cross", "None"],
                     width=10, state="readonly").grid(row=0, column=1, sticky='w', padx=5)
        tk.Label(morph_frame, text=f"(默认: {d['morph_shape']})", fg="gray", font=("Arial", 8)).grid(row=0, column=2, sticky='w', padx=2)
        vars_dict['morph_shape'] = shape_var
        tk.Label(morph_frame, text="kernel大小:").grid(row=0, column=3, sticky='e', padx=5)
        size_var = tk.IntVar(value=d['morph_size'])
        tk.Entry(morph_frame, textvariable=size_var, width=5).grid(row=0, column=4, sticky='w', padx=5)
        tk.Label(morph_frame, text=f"(默认: {d['morph_size']})", fg="gray", font=("Arial", 8)).grid(row=0, column=5, sticky='w', padx=2)
        vars_dict['morph_size'] = size_var
        tk.Label(morph_frame, text="迭代次数:").grid(row=0, column=6, sticky='e', padx=5)
        iter_var = tk.IntVar(value=d['morph_iter'])
        tk.Entry(morph_frame, textvariable=iter_var, width=5).grid(row=0, column=7, sticky='w', padx=5)
        tk.Label(morph_frame, text=f"(默认: {d['morph_iter']})", fg="gray", font=("Arial", 8)).grid(row=0, column=8, sticky='w', padx=2)
        vars_dict['morph_iter'] = iter_var
        return vars_dict

    def get_texture_params_from_vars(self, vars_dict):
        params = {}
        for key, var in vars_dict.items():
            params[key] = var.get()
        return params

    def browse_texture_folder(self):
        folder = filedialog.askdirectory(title="选择包含图片的文件夹")
        if folder:
            self.texture_folder_path.set(folder)

    def load_texture_image_list(self):
        folder = self.texture_folder_path.get().strip()
        if not folder:
            messagebox.showerror("错误", "请先选择图片文件夹。")
            return
        if not os.path.exists(folder):
            messagebox.showerror("错误", f"文件夹不存在：{folder}")
            return
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        files = [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]
        if not files:
            self.texture_status.config(text="文件夹中没有图片文件。")
            self.texture_image_files = []
            self.texture_current_idx = 0
            return
        self.texture_image_files = sorted(files)
        self.texture_current_idx = 0
        self.texture_cache.clear()
        self.texture_status.config(text=f"加载了 {len(self.texture_image_files)} 张图片")
        self.apply_texture_morphology()

    def get_texture_current_image_path(self):
        if not self.texture_image_files or self.texture_current_idx >= len(self.texture_image_files):
            return None
        folder = self.texture_folder_path.get().strip()
        return os.path.join(folder, self.texture_image_files[self.texture_current_idx])

    def prev_texture_image(self):
        if not self.texture_image_files:
            return
        self.texture_current_idx = (self.texture_current_idx - 1) % len(self.texture_image_files)
        self.update_texture_viewer()

    def next_texture_image(self):
        if not self.texture_image_files:
            return
        self.texture_current_idx = (self.texture_current_idx + 1) % len(self.texture_image_files)
        self.update_texture_viewer()

    def update_texture_viewer(self):
        idx = self.texture_current_idx
        if idx in self.texture_cache:
            data = self.texture_cache[idx]
            self.texture_viewer.show()
            param_info = self.texture_cache.get(idx, {}).get('param_info', '')
            self.texture_viewer.update(data, idx, len(self.texture_image_files),
                                       self.texture_image_files[idx], param_info)
        else:
            self.apply_texture_morphology()

    def apply_texture_morphology(self):
        img_path = self.get_texture_current_image_path()
        if img_path is None or not os.path.exists(img_path):
            self.texture_status.config(text="无效的图像路径。")
            return
        img = cv2.imread(img_path)
        if img is None:
            self.texture_status.config(text="无法读取图像。")
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fg_params = self.get_texture_params_from_vars(self.fg_vars)
        bg_params = self.get_texture_params_from_vars(self.bg_vars)

        fg_morph_str = f"FG: {fg_params['morph_shape']}, size={fg_params['morph_size']}, iter={fg_params['morph_iter']}"
        bg_morph_str = f"BG: {bg_params['morph_shape']}, size={bg_params['morph_size']}, iter={bg_params['morph_iter']}"
        param_info = f"{fg_morph_str} | {bg_morph_str}"

        try:
            _, fg_conn = extract_texture_connected(gray, fg_params)
            _, bg_conn = extract_texture_connected(gray, bg_params)
        except Exception as e:
            self.texture_status.config(text=f"提取失败: {str(e)}")
            return

        self.texture_cache[self.texture_current_idx] = {
            'fg_conn': fg_conn,
            'bg_conn': bg_conn,
            'param_info': param_info
        }
        self.texture_status.config(text=f"已提取: {self.texture_image_files[self.texture_current_idx]}")
        self.texture_viewer.show()
        self.texture_viewer.update(self.texture_cache[self.texture_current_idx],
                                   self.texture_current_idx, len(self.texture_image_files),
                                   self.texture_image_files[self.texture_current_idx], param_info)

    # ---------- 轮廓提取 ----------
    def build_contour_tab(self):
        parent = self.contour_frame
        common_frame = tk.LabelFrame(parent, text="公共设置", padx=5, pady=5)
        common_frame.pack(fill=tk.X, pady=5)
        tk.Label(common_frame, text="图片文件夹：").grid(row=0, column=0, sticky='w', padx=5)
        tk.Entry(common_frame, textvariable=self.contour_folder_path, width=50).grid(row=0, column=1, padx=5)
        tk.Button(common_frame, text="浏览...", command=self.browse_contour_folder).grid(row=0, column=2, padx=5)
        tk.Button(common_frame, text="加载图像列表", command=self.load_contour_image_list).grid(row=0, column=3, padx=5)
        tk.Label(common_frame, text="输出根目录：").grid(row=1, column=0, sticky='w', padx=5)
        tk.Entry(common_frame, textvariable=self.contour_output_base, width=50).grid(row=1, column=1, padx=5)
        tk.Button(common_frame, text="浏览...", command=self.browse_contour_output).grid(row=1, column=2, padx=5)

        param_frame = tk.LabelFrame(parent, text="轮廓提取参数", padx=5, pady=5)
        param_frame.pack(fill=tk.X, pady=5)
        tk.Label(param_frame, text="Canny高百分位:").grid(row=0, column=0, sticky='e', padx=5, pady=2)
        self.contour_high = tk.DoubleVar(value=0.9)
        tk.Entry(param_frame, textvariable=self.contour_high, width=8).grid(row=0, column=1, sticky='w', padx=5)
        tk.Label(param_frame, text="Canny低百分位:").grid(row=0, column=2, sticky='e', padx=5, pady=2)
        self.contour_low = tk.DoubleVar(value=0.8)
        tk.Entry(param_frame, textvariable=self.contour_low, width=8).grid(row=0, column=3, sticky='w', padx=5)
        tk.Label(param_frame, text="长轮廓X值:").grid(row=0, column=4, sticky='e', padx=5, pady=2)
        self.contour_x = tk.DoubleVar(value=0.2)
        tk.Entry(param_frame, textvariable=self.contour_x, width=8).grid(row=0, column=5, sticky='w', padx=5)
        tk.Label(param_frame, text="(阈值=Q1+X*(Q3-Q1))", fg="gray", font=("Arial", 8)).grid(row=0, column=6, sticky='w', padx=2)

        morph_frame = tk.LabelFrame(parent, text="形态学参数（边缘后处理）", padx=5, pady=5)
        morph_frame.pack(fill=tk.X, pady=5)
        tk.Label(morph_frame, text="kernel形状:").grid(row=0, column=0, sticky='e', padx=5)
        self.contour_morph_shape = tk.StringVar(value="rect")
        ttk.Combobox(morph_frame, textvariable=self.contour_morph_shape,
                     values=["rect", "ellipse", "cross", "None"],
                     width=10, state="readonly").grid(row=0, column=1, sticky='w', padx=5)
        tk.Label(morph_frame, text="(默认: rect)", fg="gray", font=("Arial", 8)).grid(row=0, column=2, sticky='w', padx=2)
        tk.Label(morph_frame, text="kernel大小:").grid(row=0, column=3, sticky='e', padx=5)
        self.contour_morph_size = tk.IntVar(value=4)
        tk.Entry(morph_frame, textvariable=self.contour_morph_size, width=5).grid(row=0, column=4, sticky='w', padx=5)
        tk.Label(morph_frame, text="(默认: 4)", fg="gray", font=("Arial", 8)).grid(row=0, column=5, sticky='w', padx=2)
        tk.Label(morph_frame, text="迭代次数:").grid(row=0, column=6, sticky='e', padx=5)
        self.contour_morph_iter = tk.IntVar(value=1)
        tk.Entry(morph_frame, textvariable=self.contour_morph_iter, width=5).grid(row=0, column=7, sticky='w', padx=5)
        tk.Label(morph_frame, text="(默认: 1)", fg="gray", font=("Arial", 8)).grid(row=0, column=8, sticky='w', padx=2)

        btn_frame = tk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="运行轮廓提取(批量)", command=self.run_contour_extraction,
                  bg="#2196F3", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="形态学操作(预览)", command=self.apply_contour_morphology,
                  bg="#FF9800", fg="white", width=15).pack(side=tk.LEFT, padx=5)
        self.contour_status = tk.Label(parent, text="就绪", anchor='w')
        self.contour_status.pack(fill=tk.X, pady=5)

    def browse_contour_folder(self):
        folder = filedialog.askdirectory(title="选择包含图片的文件夹")
        if folder:
            self.contour_folder_path.set(folder)

    def browse_contour_output(self):
        folder = filedialog.askdirectory(title="选择输出根目录")
        if folder:
            self.contour_output_base.set(folder)

    def load_contour_image_list(self):
        folder = self.contour_folder_path.get().strip()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("错误", f"文件夹不存在：{folder}")
            return
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        files = [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]
        if not files:
            self.contour_status.config(text="文件夹中没有图片文件。")
            self.contour_image_files = []
            self.contour_current_idx = 0
            return
        self.contour_image_files = sorted(files)
        self.contour_current_idx = 0
        self.contour_cache.clear()
        self.contour_status.config(text=f"加载了 {len(self.contour_image_files)} 张图片")
        self.apply_contour_morphology()

    def get_contour_current_image_path(self):
        if not self.contour_image_files or self.contour_current_idx >= len(self.contour_image_files):
            return None
        folder = self.contour_folder_path.get().strip()
        return os.path.join(folder, self.contour_image_files[self.contour_current_idx])

    def prev_contour_image(self):
        if not self.contour_image_files:
            return
        self.contour_current_idx = (self.contour_current_idx - 1) % len(self.contour_image_files)
        self.update_contour_viewer()

    def next_contour_image(self):
        if not self.contour_image_files:
            return
        self.contour_current_idx = (self.contour_current_idx + 1) % len(self.contour_image_files)
        self.update_contour_viewer()

    def update_contour_viewer(self):
        idx = self.contour_current_idx
        if idx in self.contour_cache:
            data = self.contour_cache[idx]
            self.contour_viewer.show()
            param_info = self.contour_cache.get(idx, {}).get('param_info', '')
            self.contour_viewer.update(data, idx, len(self.contour_image_files),
                                       self.contour_image_files[idx], param_info)
        else:
            self.apply_contour_morphology()

    def apply_contour_morphology(self):
        if not self.contour_image_files:
            messagebox.showwarning("提示", "请先加载图像列表（点击'加载图像列表'按钮）。")
            self.contour_status.config(text="请先加载图像列表")
            return
        img_path = self.get_contour_current_image_path()
        if img_path is None or not os.path.exists(img_path):
            self.contour_status.config(text="无效的图像路径。")
            return
        img = cv2.imread(img_path)
        if img is None:
            self.contour_status.config(text="无法读取图像。")
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        high = self.contour_high.get()
        low = self.contour_low.get()
        x_val = self.contour_x.get()
        morph_shape = self.contour_morph_shape.get()
        morph_size = self.contour_morph_size.get()
        morph_iter = self.contour_morph_iter.get()

        low_p, high_p, _ = auto_canny_threshold_contour(gray, high, low)
        edges = cv2.Canny(gray, low_p, high_p)
        filtered = filter_long_contours(edges, x_multiplier=x_val)
        morph_edges = apply_morphology_to_edge(filtered, morph_shape, morph_size, morph_iter)

        param_info = f"X={x_val}, morph={morph_shape}, size={morph_size}, iter={morph_iter}"
        self.contour_cache[self.contour_current_idx] = {
            'original': filtered,
            'morph': morph_edges,
            'param_info': param_info
        }
        self.contour_status.config(text=f"已处理: {self.contour_image_files[self.contour_current_idx]}")
        self.contour_viewer.show()
        self.contour_viewer.update(self.contour_cache[self.contour_current_idx],
                                   self.contour_current_idx, len(self.contour_image_files),
                                   self.contour_image_files[self.contour_current_idx], param_info)

    def run_contour_extraction(self):
        input_dir = self.contour_folder_path.get().strip()
        if not input_dir:
            messagebox.showerror("错误", "请先选择图片文件夹。")
            return
        output_base = self.contour_output_base.get().strip()
        if not output_base:
            messagebox.showerror("错误", "请指定输出根目录。")
            return

        try:
            high = self.contour_high.get()
            low = self.contour_low.get()
            x = self.contour_x.get()
            morph_shape = self.contour_morph_shape.get()
            morph_size = self.contour_morph_size.get()
            morph_iter = self.contour_morph_iter.get()
        except:
            messagebox.showerror("参数错误", "请输入有效数值。")
            return

        self.contour_status.config(text="正在批量处理，请稍候...")
        after_dir = os.path.join(output_base, f"afterfilter-xratio{x}")
        morph_dir = os.path.join(output_base, f"afterfilter-xratio{x}_morph")
        before_dir = os.path.join(output_base, "beforefilter")
        process_contour_folder(input_dir, after_dir, morph_dir,
                               high, low, x,
                               morph_shape, morph_size, morph_iter,
                               output_before_dir=before_dir,
                               log_callback=lambda msg: self.contour_status.config(text=msg))
        self.contour_status.config(text="批量轮廓提取完成！")
        self.logic_step1_B.set(morph_dir)

    # ---------- 衬度提取 ----------
    def build_contrast_tab(self):
        parent = self.contrast_frame
        common_frame = tk.LabelFrame(parent, text="公共设置", padx=5, pady=5)
        common_frame.pack(fill=tk.X, pady=5)
        tk.Label(common_frame, text="图片文件夹：").grid(row=0, column=0, sticky='w', padx=5)
        tk.Entry(common_frame, textvariable=self.contrast_folder_path, width=50).grid(row=0, column=1, padx=5)
        tk.Button(common_frame, text="浏览...", command=self.browse_contrast_folder).grid(row=0, column=2, padx=5)
        tk.Button(common_frame, text="加载图像列表", command=self.load_contrast_image_list).grid(row=0, column=3, padx=5)
        tk.Label(common_frame, text="输出根目录：").grid(row=1, column=0, sticky='w', padx=5)
        tk.Entry(common_frame, textvariable=self.contrast_output_base, width=50).grid(row=1, column=1, padx=5)
        tk.Button(common_frame, text="浏览...", command=self.browse_contrast_output).grid(row=1, column=2, padx=5)

        param_container = tk.Frame(parent)
        param_container.pack(fill=tk.BOTH, expand=True, pady=5)

        fore_frame = tk.LabelFrame(param_container, text="用于提取前景", padx=5, pady=5)
        fore_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        tk.Label(fore_frame, text="Otsu阈值比例:").grid(row=0, column=0, sticky='e', padx=5, pady=2)
        self.contrast_fore_ratio = tk.DoubleVar(value=0.9)
        tk.Entry(fore_frame, textvariable=self.contrast_fore_ratio, width=8).grid(row=0, column=1, sticky='w', padx=5)
        tk.Label(fore_frame, text="(默认: 0.9)", fg="gray", font=("Arial", 8)).grid(row=0, column=2, sticky='w', padx=2)
        self.contrast_fore_invert = tk.BooleanVar(value=True)
        tk.Checkbutton(fore_frame, text="黑白反转", variable=self.contrast_fore_invert).grid(row=1, column=0, columnspan=3, sticky='w', padx=5, pady=2)

        tk.Label(fore_frame, text="kernel形状:").grid(row=2, column=0, sticky='e', padx=5, pady=2)
        self.contrast_fore_morph_shape = tk.StringVar(value="None")
        ttk.Combobox(fore_frame, textvariable=self.contrast_fore_morph_shape,
                     values=["rect", "ellipse", "cross", "None"],
                     width=10, state="readonly").grid(row=2, column=1, sticky='w', padx=5)
        tk.Label(fore_frame, text="(默认: None)", fg="gray", font=("Arial", 8)).grid(row=2, column=2, sticky='w', padx=2)
        tk.Label(fore_frame, text="kernel大小:").grid(row=3, column=0, sticky='e', padx=5, pady=2)
        self.contrast_fore_morph_size = tk.IntVar(value=3)
        tk.Entry(fore_frame, textvariable=self.contrast_fore_morph_size, width=5).grid(row=3, column=1, sticky='w', padx=5)
        tk.Label(fore_frame, text="(默认: 3)", fg="gray", font=("Arial", 8)).grid(row=3, column=2, sticky='w', padx=2)
        tk.Label(fore_frame, text="迭代次数:").grid(row=4, column=0, sticky='e', padx=5, pady=2)
        self.contrast_fore_morph_iter = tk.IntVar(value=1)
        tk.Entry(fore_frame, textvariable=self.contrast_fore_morph_iter, width=5).grid(row=4, column=1, sticky='w', padx=5)
        tk.Label(fore_frame, text="(默认: 1)", fg="gray", font=("Arial", 8)).grid(row=4, column=2, sticky='w', padx=2)

        back_frame = tk.LabelFrame(param_container, text="用于获取背景", padx=5, pady=5)
        back_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5,0))
        tk.Label(back_frame, text="Otsu阈值比例:").grid(row=0, column=0, sticky='e', padx=5, pady=2)
        self.contrast_back_ratio = tk.DoubleVar(value=1.4)
        tk.Entry(back_frame, textvariable=self.contrast_back_ratio, width=8).grid(row=0, column=1, sticky='w', padx=5)
        tk.Label(back_frame, text="(默认: 1.4)", fg="gray", font=("Arial", 8)).grid(row=0, column=2, sticky='w', padx=2)
        self.contrast_back_invert = tk.BooleanVar(value=False)
        tk.Checkbutton(back_frame, text="黑白反转", variable=self.contrast_back_invert).grid(row=1, column=0, columnspan=3, sticky='w', padx=5, pady=2)

        tk.Label(back_frame, text="kernel形状:").grid(row=2, column=0, sticky='e', padx=5, pady=2)
        self.contrast_back_morph_shape = tk.StringVar(value="None")
        ttk.Combobox(back_frame, textvariable=self.contrast_back_morph_shape,
                     values=["rect", "ellipse", "cross", "None"],
                     width=10, state="readonly").grid(row=2, column=1, sticky='w', padx=5)
        tk.Label(back_frame, text="(默认: None)", fg="gray", font=("Arial", 8)).grid(row=2, column=2, sticky='w', padx=2)
        tk.Label(back_frame, text="kernel大小:").grid(row=3, column=0, sticky='e', padx=5, pady=2)
        self.contrast_back_morph_size = tk.IntVar(value=3)
        tk.Entry(back_frame, textvariable=self.contrast_back_morph_size, width=5).grid(row=3, column=1, sticky='w', padx=5)
        tk.Label(back_frame, text="(默认: 3)", fg="gray", font=("Arial", 8)).grid(row=3, column=2, sticky='w', padx=2)
        tk.Label(back_frame, text="迭代次数:").grid(row=4, column=0, sticky='e', padx=5, pady=2)
        self.contrast_back_morph_iter = tk.IntVar(value=1)
        tk.Entry(back_frame, textvariable=self.contrast_back_morph_iter, width=5).grid(row=4, column=1, sticky='w', padx=5)
        tk.Label(back_frame, text="(默认: 1)", fg="gray", font=("Arial", 8)).grid(row=4, column=2, sticky='w', padx=2)

        btn_frame = tk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="运行衬度提取(批量)", command=self.run_contrast_extraction,
                  bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="形态学操作(预览)", command=self.apply_contrast_morphology,
                  bg="#FF9800", fg="white", width=15).pack(side=tk.LEFT, padx=5)
        self.contrast_status = tk.Label(parent, text="就绪", anchor='w')
        self.contrast_status.pack(fill=tk.X, pady=5)

    def browse_contrast_folder(self):
        folder = filedialog.askdirectory(title="选择包含图片的文件夹")
        if folder:
            self.contrast_folder_path.set(folder)

    def browse_contrast_output(self):
        folder = filedialog.askdirectory(title="选择输出根目录")
        if folder:
            self.contrast_output_base.set(folder)

    def load_contrast_image_list(self):
        folder = self.contrast_folder_path.get().strip()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("错误", f"文件夹不存在：{folder}")
            return
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        files = [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]
        if not files:
            self.contrast_status.config(text="文件夹中没有图片文件。")
            self.contrast_image_files = []
            self.contrast_current_idx = 0
            return
        self.contrast_image_files = sorted(files)
        self.contrast_current_idx = 0
        self.contrast_cache.clear()
        self.contrast_status.config(text=f"加载了 {len(self.contrast_image_files)} 张图片")
        self.apply_contrast_morphology()

    def get_contrast_current_image_path(self):
        if not self.contrast_image_files or self.contrast_current_idx >= len(self.contrast_image_files):
            return None
        folder = self.contrast_folder_path.get().strip()
        return os.path.join(folder, self.contrast_image_files[self.contrast_current_idx])

    def prev_contrast_image(self):
        if not self.contrast_image_files:
            return
        self.contrast_current_idx = (self.contrast_current_idx - 1) % len(self.contrast_image_files)
        self.update_contrast_viewer()

    def next_contrast_image(self):
        if not self.contrast_image_files:
            return
        self.contrast_current_idx = (self.contrast_current_idx + 1) % len(self.contrast_image_files)
        self.update_contrast_viewer()

    def update_contrast_viewer(self):
        idx = self.contrast_current_idx
        if idx in self.contrast_cache:
            data = self.contrast_cache[idx]
            self.contrast_viewer.show()
            param_info = self.contrast_cache.get(idx, {}).get('param_info', '')
            self.contrast_viewer.update(data, idx, len(self.contrast_image_files),
                                        self.contrast_image_files[idx], param_info)
        else:
            self.apply_contrast_morphology()

    def apply_contrast_morphology(self):
        if not self.contrast_image_files:
            messagebox.showwarning("提示", "请先加载图像列表（点击'加载图像列表'按钮）。")
            self.contrast_status.config(text="请先加载图像列表")
            return
        img_path = self.get_contrast_current_image_path()
        if img_path is None or not os.path.exists(img_path):
            self.contrast_status.config(text="无效的图像路径。")
            return
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.contrast_status.config(text="无法读取图像。")
            return

        fore_ratio = self.contrast_fore_ratio.get()
        back_ratio = self.contrast_back_ratio.get()
        invert_fore = self.contrast_fore_invert.get()
        invert_back = self.contrast_back_invert.get()
        fore_shape = self.contrast_fore_morph_shape.get()
        fore_size = self.contrast_fore_morph_size.get()
        fore_iter = self.contrast_fore_morph_iter.get()
        back_shape = self.contrast_back_morph_shape.get()
        back_size = self.contrast_back_morph_size.get()
        back_iter = self.contrast_back_morph_iter.get()

        fore_orig, fore_morph = process_contrast_image_single(
            img, fore_ratio, invert_fore, fore_shape, fore_size, fore_iter)
        back_orig, back_morph = process_contrast_image_single(
            img, back_ratio, invert_back, back_shape, back_size, back_iter)

        param_info = (f"前景: ratio={fore_ratio}, invert={invert_fore}, morph={fore_shape},{fore_size},{fore_iter} | "
                      f"背景: ratio={back_ratio}, invert={invert_back}, morph={back_shape},{back_size},{back_iter}")

        self.contrast_cache[self.contrast_current_idx] = {
            'fore_orig': fore_orig,
            'fore_morph': fore_morph,
            'back_orig': back_orig,
            'back_morph': back_morph,
            'param_info': param_info
        }
        self.contrast_status.config(text=f"已处理: {self.contrast_image_files[self.contrast_current_idx]}")
        self.contrast_viewer.show()
        self.contrast_viewer.update(self.contrast_cache[self.contrast_current_idx],
                                    self.contrast_current_idx, len(self.contrast_image_files),
                                    self.contrast_image_files[self.contrast_current_idx], param_info)

    def run_contrast_extraction(self):
        input_dir = self.contrast_folder_path.get().strip()
        if not input_dir:
            messagebox.showerror("错误", "请先选择图片文件夹。")
            return
        output_base = self.contrast_output_base.get().strip()
        if not output_base:
            messagebox.showerror("错误", "请指定输出根目录。")
            return

        try:
            fore_ratio = self.contrast_fore_ratio.get()
            back_ratio = self.contrast_back_ratio.get()
            invert_fore = self.contrast_fore_invert.get()
            invert_back = self.contrast_back_invert.get()
        except:
            messagebox.showerror("参数错误", "请输入有效数值。")
            return

        self.contrast_status.config(text="正在批量处理，请稍候...")
        fore_out = os.path.join(output_base, f"otsu_{fore_ratio}_fore")
        back_out = os.path.join(output_base, f"otsu_{back_ratio}_back")
        process_contrast_images(input_dir, fore_out, back_out,
                                fore_ratio, back_ratio, invert_fore, invert_back,
                                log_callback=lambda msg: self.contrast_status.config(text=msg))
        self.contrast_status.config(text="衬度提取完成！")
        self.logic_step1_A.set(fore_out)
        self.logic_back_A.set(back_out)

    # ---------- 逻辑操作 ----------
    def build_logic_tab(self):
        parent = self.logic_frame
        prefix_frame = tk.LabelFrame(parent, text="前缀匹配设置", padx=5, pady=5)
        prefix_frame.pack(fill=tk.X, pady=5)
        tk.Label(prefix_frame, text="前缀匹配位数 (0表示完全匹配):").pack(side=tk.LEFT, padx=5)
        tk.Entry(prefix_frame, textvariable=self.logic_prefix_len, width=6).pack(side=tk.LEFT, padx=5)
        tk.Label(prefix_frame, text="(默认: 2)", fg="gray").pack(side=tk.LEFT, padx=5)

        fore_frame = tk.LabelFrame(parent, text="前景提取", padx=5, pady=5, font=('Arial', 10, 'bold'))
        fore_frame.pack(fill=tk.X, pady=5)

        step1_frame = tk.LabelFrame(fore_frame, text="步骤1: 差集(A-B) - A为衬度前景，B为轮廓形态学结果", padx=5, pady=5)
        step1_frame.pack(fill=tk.X, pady=2)
        tk.Label(step1_frame, text="A (衬度前景):").grid(row=0, column=0, sticky='w', padx=5)
        tk.Entry(step1_frame, textvariable=self.logic_step1_A, width=40).grid(row=0, column=1, padx=5)
        tk.Button(step1_frame, text="浏览...", command=lambda: self.browse_logic_folder(self.logic_step1_A)).grid(row=0, column=2, padx=5)
        tk.Label(step1_frame, text="B (轮廓形态学):").grid(row=1, column=0, sticky='w', padx=5)
        tk.Entry(step1_frame, textvariable=self.logic_step1_B, width=40).grid(row=1, column=1, padx=5)
        tk.Button(step1_frame, text="浏览...", command=lambda: self.browse_logic_folder(self.logic_step1_B)).grid(row=1, column=2, padx=5)
        tk.Label(step1_frame, text="输出目录:").grid(row=2, column=0, sticky='w', padx=5)
        tk.Entry(step1_frame, textvariable=self.logic_step1_out, width=40).grid(row=2, column=1, padx=5)
        tk.Button(step1_frame, text="浏览...", command=lambda: self.browse_logic_folder(self.logic_step1_out)).grid(row=2, column=2, padx=5)

        step2_frame = tk.LabelFrame(fore_frame, text="步骤2: 并集(Union) - A为步骤1结果，B为纹理前景连接", padx=5, pady=5)
        step2_frame.pack(fill=tk.X, pady=2)
        tk.Label(step2_frame, text="A (步骤1结果):").grid(row=0, column=0, sticky='w', padx=5)
        tk.Entry(step2_frame, textvariable=self.logic_step2_A, width=40).grid(row=0, column=1, padx=5)
        tk.Button(step2_frame, text="浏览...", command=lambda: self.browse_logic_folder(self.logic_step2_A)).grid(row=0, column=2, padx=5)
        tk.Label(step2_frame, text="B (纹理前景连接):").grid(row=1, column=0, sticky='w', padx=5)
        tk.Entry(step2_frame, textvariable=self.logic_step2_B, width=40).grid(row=1, column=1, padx=5)
        tk.Button(step2_frame, text="浏览...", command=lambda: self.browse_logic_folder(self.logic_step2_B)).grid(row=1, column=2, padx=5)
        tk.Label(step2_frame, text="输出目录:").grid(row=2, column=0, sticky='w', padx=5)
        tk.Entry(step2_frame, textvariable=self.logic_step2_out, width=40).grid(row=2, column=1, padx=5)
        tk.Button(step2_frame, text="浏览...", command=lambda: self.browse_logic_folder(self.logic_step2_out)).grid(row=2, column=2, padx=5)

        back_frame = tk.LabelFrame(parent, text="背景提取", padx=5, pady=5, font=('Arial', 10, 'bold'))
        back_frame.pack(fill=tk.X, pady=5)
        back_step_frame = tk.LabelFrame(back_frame, text="差集(A-B) - A为衬度背景，B为纹理背景连接", padx=5, pady=5)
        back_step_frame.pack(fill=tk.X, pady=2)
        tk.Label(back_step_frame, text="A (衬度背景):").grid(row=0, column=0, sticky='w', padx=5)
        tk.Entry(back_step_frame, textvariable=self.logic_back_A, width=40).grid(row=0, column=1, padx=5)
        tk.Button(back_step_frame, text="浏览...", command=lambda: self.browse_logic_folder(self.logic_back_A)).grid(row=0, column=2, padx=5)
        tk.Label(back_step_frame, text="B (纹理背景连接):").grid(row=1, column=0, sticky='w', padx=5)
        tk.Entry(back_step_frame, textvariable=self.logic_back_B, width=40).grid(row=1, column=1, padx=5)
        tk.Button(back_step_frame, text="浏览...", command=lambda: self.browse_logic_folder(self.logic_back_B)).grid(row=1, column=2, padx=5)
        tk.Label(back_step_frame, text="输出目录:").grid(row=2, column=0, sticky='w', padx=5)
        tk.Entry(back_step_frame, textvariable=self.logic_back_out, width=40).grid(row=2, column=1, padx=5)
        tk.Button(back_step_frame, text="浏览...", command=lambda: self.browse_logic_folder(self.logic_back_out)).grid(row=2, column=2, padx=5)

        btn_frame = tk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=10)
        tk.Button(btn_frame, text="执行逻辑操作", command=self.run_logic_ops,
                  bg="#FF5722", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.logic_status = tk.Label(parent, text="就绪", anchor='w')
        self.logic_status.pack(fill=tk.X, pady=5)

    def browse_logic_folder(self, var):
        folder = filedialog.askdirectory(title="选择文件夹")
        if folder:
            var.set(folder)

    def run_logic_ops(self):
        prefix_len = self.logic_prefix_len.get()
        step1_A = self.logic_step1_A.get().strip()
        step1_B = self.logic_step1_B.get().strip()
        step1_out = self.logic_step1_out.get().strip()
        step2_A = self.logic_step2_A.get().strip()
        step2_B = self.logic_step2_B.get().strip()
        step2_out = self.logic_step2_out.get().strip()
        back_A = self.logic_back_A.get().strip()
        back_B = self.logic_back_B.get().strip()
        back_out = self.logic_back_out.get().strip()

        if not all([step1_A, step1_B, step1_out, step2_A, step2_B, step2_out, back_A, back_B, back_out]):
            messagebox.showerror("错误", "请填写所有路径（包括输出目录）。")
            return

        self.logic_status.config(text="执行步骤1: 差集(A-B)...")
        count1, skip1 = logic_operation_folder(step1_A, step1_B, "差集(A-B)", step1_out, prefix_len, log_callback=lambda msg: self.logic_status.config(text=msg))
        if count1 == 0 and skip1 == 0:
            messagebox.showwarning("警告", "步骤1未处理任何图像，请检查路径和图像匹配。")
            return

        self.logic_step2_A.set(step1_out)
        self.logic_status.config(text="执行步骤2: 并集(Union)...")
        count2, skip2 = logic_operation_folder(step1_out, step2_B, "并集", step2_out, prefix_len, log_callback=lambda msg: self.logic_status.config(text=msg))

        self.logic_status.config(text="执行背景提取: 差集(A-B)...")
        count3, skip3 = logic_operation_folder(back_A, back_B, "差集(A-B)", back_out, prefix_len, log_callback=lambda msg: self.logic_status.config(text=msg))

        msg = f"全部完成！\n步骤1: {count1} 张，跳过 {skip1} 张\n步骤2: {count2} 张，跳过 {skip2} 张\n背景: {count3} 张，跳过 {skip3} 张"
        messagebox.showinfo("完成", msg)
        self.logic_status.config(text="逻辑操作全部完成！")

    # ---------- AP Generate 选项卡 ----------
    def build_ap_tab(self):
        parent = self.ap_frame
        # 文件夹设置
        common_frame = tk.LabelFrame(parent, text="文件夹设置", padx=5, pady=5)
        common_frame.pack(fill=tk.X, pady=5)
        row = 0
        for label, var in [("原始图像文件夹：", self.ap_original_folder),
                           ("前景二值文件夹：", self.ap_foreground_folder),
                           ("背景二值文件夹：", self.ap_background_folder),
                           ("输出文件夹：", self.ap_output_folder)]:
            tk.Label(common_frame, text=label).grid(row=row, column=0, sticky='w', padx=5)
            tk.Entry(common_frame, textvariable=var, width=50).grid(row=row, column=1, padx=5)
            tk.Button(common_frame, text="浏览...", command=lambda v=var: self.browse_ap_folder(v)).grid(row=row, column=2, padx=5)
            row += 1

        # 参数设置
        param_frame = tk.LabelFrame(parent, text="提取参数", padx=5, pady=5)
        param_frame.pack(fill=tk.X, pady=5)

        fg_frame = tk.LabelFrame(param_frame, text="前景参数", padx=5, pady=5)
        fg_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        tk.Label(fg_frame, text="fg_x_ratio:").grid(row=0, column=0, sticky='e', padx=5)
        tk.Entry(fg_frame, textvariable=self.ap_fg_x_ratio, width=8).grid(row=0, column=1, sticky='w', padx=5)
        tk.Label(fg_frame, text="fg_ratio:").grid(row=1, column=0, sticky='e', padx=5)
        tk.Entry(fg_frame, textvariable=self.ap_fg_ratio, width=8).grid(row=1, column=1, sticky='w', padx=5)

        bg_frame = tk.LabelFrame(param_frame, text="背景参数", padx=5, pady=5)
        bg_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5,0))
        tk.Label(bg_frame, text="bg_min_area:").grid(row=0, column=0, sticky='e', padx=5)
        tk.Entry(bg_frame, textvariable=self.ap_bg_min_area, width=8).grid(row=0, column=1, sticky='w', padx=5)
        tk.Label(bg_frame, text="bg_margin:").grid(row=1, column=0, sticky='e', padx=5)
        tk.Entry(bg_frame, textvariable=self.ap_bg_margin, width=8).grid(row=1, column=1, sticky='w', padx=5)
        tk.Label(bg_frame, text="bg_ratio:").grid(row=2, column=0, sticky='e', padx=5)
        tk.Entry(bg_frame, textvariable=self.ap_bg_ratio, width=8).grid(row=2, column=1, sticky='w', padx=5)
        tk.Label(bg_frame, text="bg_top_x:").grid(row=3, column=0, sticky='e', padx=5)
        tk.Entry(bg_frame, textvariable=self.ap_bg_top_x, width=8).grid(row=3, column=1, sticky='w', padx=5)

        prefix_frame = tk.LabelFrame(parent, text="前缀匹配设置", padx=5, pady=5)
        prefix_frame.pack(fill=tk.X, pady=5)
        tk.Label(prefix_frame, text="前缀匹配位数:").pack(side=tk.LEFT, padx=5)
        tk.Entry(prefix_frame, textvariable=self.ap_prefix_len, width=6).pack(side=tk.LEFT, padx=5)
        tk.Label(prefix_frame, text="(默认2)").pack(side=tk.LEFT, padx=5)

        btn_frame = tk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="生成提示", command=self.run_ap_generate,
                  bg="#9C27B0", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.ap_status = tk.Label(parent, text="就绪", anchor='w')
        self.ap_status.pack(fill=tk.X, pady=5)

    def browse_ap_folder(self, var):
        folder = filedialog.askdirectory(title="选择文件夹")
        if folder:
            var.set(folder)

    def run_ap_generate(self):
        orig_folder = self.ap_original_folder.get().strip()
        fg_folder = self.ap_foreground_folder.get().strip()
        bg_folder = self.ap_background_folder.get().strip()
        out_folder = self.ap_output_folder.get().strip()

        if not all([orig_folder, fg_folder, bg_folder, out_folder]):
            messagebox.showerror("错误", "请完整填写所有文件夹路径。")
            return

        try:
            fg_x = self.ap_fg_x_ratio.get()
            fg_r = self.ap_fg_ratio.get()
            bg_min = self.ap_bg_min_area.get()
            bg_margin = self.ap_bg_margin.get()
            bg_r = self.ap_bg_ratio.get()
            bg_top = self.ap_bg_top_x.get()
            prefix_len = self.ap_prefix_len.get()
        except Exception as e:
            messagebox.showerror("参数错误", f"参数格式错误：{e}")
            return

        self.ap_status.config(text="正在处理，请稍候...")
        def task():
            try:
                basenames = process_ap_images(
                    original_folder=orig_folder,
                    foreground_folder=fg_folder,
                    background_folder=bg_folder,
                    output_folder=out_folder,
                    fg_x_ratio=fg_x,
                    fg_ratio=fg_r,
                    bg_min_area=bg_min,
                    bg_margin=bg_margin,
                    bg_ratio=bg_r,
                    bg_top_x=bg_top,
                    prefix_len=prefix_len
                )
                self.master.after(0, lambda: self.ap_status.config(text=f"处理完成！生成了 {len(basenames)} 组图像"))
                if basenames:
                    self.master.after(0, lambda: self.load_ap_image_list(out_folder, basenames))
                else:
                    self.master.after(0, lambda: messagebox.showwarning("警告", "未生成任何结果，请检查输入。"))
            except Exception as e:
                self.master.after(0, lambda: self.ap_status.config(text=f"错误: {str(e)}"))
                import traceback
                traceback.print_exc()

        threading.Thread(target=task, daemon=True).start()

    def load_ap_image_list(self, output_folder, basenames):
        self.ap_image_files = sorted(basenames)
        self.ap_current_idx = 0
        self.ap_cache.clear()
        if not self.ap_image_files:
            self.ap_status.config(text="没有找到生成的图像。")
            return
        self.ap_status.config(text=f"加载了 {len(self.ap_image_files)} 组图像")
        self.update_ap_viewer()

    def get_ap_current_paths(self):
        if not self.ap_image_files or self.ap_current_idx >= len(self.ap_image_files):
            return None, None
        base = self.ap_image_files[self.ap_current_idx]
        out_folder = self.ap_output_folder.get().strip()
        overlay_path = os.path.join(out_folder, f"{base}_overlay.png")
        mask_path = os.path.join(out_folder, f"{base}_mask_overlay.png")
        return overlay_path, mask_path

    def prev_ap_image(self):
        if not self.ap_image_files:
            return
        self.ap_current_idx = (self.ap_current_idx - 1) % len(self.ap_image_files)
        self.update_ap_viewer()

    def next_ap_image(self):
        if not self.ap_image_files:
            return
        self.ap_current_idx = (self.ap_current_idx + 1) % len(self.ap_image_files)
        self.update_ap_viewer()

    def update_ap_viewer(self):
        idx = self.ap_current_idx
        if idx in self.ap_cache:
            data = self.ap_cache[idx]
            self.ap_viewer.show()
            param_info = self.ap_cache.get(idx, {}).get('param_info', '')
            self.ap_viewer.update(data, idx, len(self.ap_image_files),
                                  self.ap_image_files[idx], param_info)
        else:
            overlay_path, mask_path = self.get_ap_current_paths()
            if overlay_path is None or not os.path.exists(overlay_path):
                self.ap_status.config(text="图像文件不存在。")
                return
            overlay = cv2.imread(overlay_path)
            mask = cv2.imread(mask_path) if os.path.exists(mask_path) else np.zeros_like(overlay)
            if overlay is None:
                self.ap_status.config(text="无法读取图像。")
                return
            fg_x = self.ap_fg_x_ratio.get()
            fg_r = self.ap_fg_ratio.get()
            bg_min = self.ap_bg_min_area.get()
            bg_margin = self.ap_bg_margin.get()
            bg_r = self.ap_bg_ratio.get()
            bg_top = self.ap_bg_top_x.get()
            param_info = f"fg_x={fg_x}, fg_ratio={fg_r}, bg_min={bg_min}, bg_margin={bg_margin}, bg_ratio={bg_r}, bg_top={bg_top}"
            data = {
                'overlay': overlay,
                'mask': mask,
                'param_info': param_info
            }
            self.ap_cache[idx] = data
            self.ap_viewer.show()
            self.ap_viewer.update(data, idx, len(self.ap_image_files),
                                  self.ap_image_files[idx], param_info)

    # ==================== Run 按钮 ====================
    def run_all(self):
        input_dir = self.lowpass_folder_path.get().strip()
        if not input_dir:
            messagebox.showerror("错误", "请先在低通滤波功能区选择图片文件夹。")
            return
        if not os.path.exists(input_dir):
            messagebox.showerror("错误", f"输入文件夹不存在：{input_dir}")
            return

        for child in self.master.winfo_children():
            if isinstance(child, tk.Frame):
                for btn in child.winfo_children():
                    if isinstance(btn, tk.Button) and btn['text'] == '▶ Run':
                        btn.config(state='disabled')
                        break

        def task():
            try:
                self._run_all_impl(input_dir)
            except Exception as e:
                messagebox.showerror("错误", f"处理过程中发生异常：\n{str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                for child in self.master.winfo_children():
                    if isinstance(child, tk.Frame):
                        for btn in child.winfo_children():
                            if isinstance(btn, tk.Button) and btn['text'] == '▶ Run':
                                btn.config(state='normal')
                                break

        threading.Thread(target=task, daemon=True).start()

    def _run_all_impl(self, input_dir):
        lpf_output_dir = input_dir + "-FFT"
        r = self.lowpass_r.get()
        r_noise = self.lowpass_r_noise.get()

        def log(msg):
            self.master.after(0, lambda: self.logic_status.config(text=msg))

        log("执行低通滤波...")
        success = process_lowpass_filter(input_dir, lpf_output_dir, r, r_noise, log_callback=log)
        if not success:
            log("低通滤波失败，终止处理。")
            messagebox.showerror("错误", "低通滤波失败，请检查输入图像。")
            return

        self.master.after(0, lambda: self.contour_folder_path.set(lpf_output_dir))
        self.master.after(0, lambda: self.contrast_folder_path.set(lpf_output_dir))
        self.master.after(0, lambda: self.texture_folder_path.set(input_dir))

        results_root = os.path.join(os.path.dirname(input_dir), os.path.basename(input_dir) + "_results")
        tex_fore_conn_dir = os.path.join(results_root, "tex_fore")
        tex_back_conn_dir = os.path.join(results_root, "tex_back")
        tex_fore_tex_dir = os.path.join(results_root, "tex_fore_single")
        tex_back_tex_dir = os.path.join(results_root, "tex_back_single")
        fg_params = self.get_texture_params_from_vars(self.fg_vars)
        bg_params = self.get_texture_params_from_vars(self.bg_vars)
        log("开始纹理提取...")
        process_texture_batch(input_dir,
                              tex_fore_conn_dir, tex_back_conn_dir,
                              tex_fore_tex_dir, tex_back_tex_dir,
                              fg_params, bg_params, log_callback=log)

        contour_dir = os.path.join(results_root, "contour")
        contour_morph_dir = os.path.join(results_root, "contour_morph")
        contour_high = self.contour_high.get()
        contour_low = self.contour_low.get()
        contour_x = self.contour_x.get()
        morph_shape = self.contour_morph_shape.get()
        morph_size = self.contour_morph_size.get()
        morph_iter = self.contour_morph_iter.get()
        log("开始轮廓提取...")
        process_contour_folder(lpf_output_dir, contour_dir, contour_morph_dir,
                               contour_high, contour_low, contour_x,
                               morph_shape, morph_size, morph_iter,
                               output_before_dir=None, log_callback=log)

        contrast_fore_dir = os.path.join(results_root, "contrast_fore")
        contrast_back_dir = os.path.join(results_root, "contrast_back")
        fore_ratio = self.contrast_fore_ratio.get()
        back_ratio = self.contrast_back_ratio.get()
        invert_fore = self.contrast_fore_invert.get()
        invert_back = self.contrast_back_invert.get()
        log("开始衬度提取...")
        process_contrast_images(lpf_output_dir, contrast_fore_dir, contrast_back_dir,
                                fore_ratio, back_ratio, invert_fore, invert_back, log_callback=log)

        prefix_len = self.logic_prefix_len.get()
        contrast_contour_dir = os.path.join(results_root, "contrast_contour")
        forecoarse_dir = os.path.join(results_root, "forecoarse")
        backcoarse_dir = os.path.join(results_root, "backcoarse")

        log("执行前景第一步: 差集(A-B) (A=contrast_fore, B=contour_morph)")
        count1, skip1 = logic_operation_folder(contrast_fore_dir, contour_morph_dir, "差集(A-B)", contrast_contour_dir, prefix_len, log_callback=log)
        if count1 == 0 and skip1 == 0:
            messagebox.showwarning("警告", "前景第一步未处理任何图像，请检查路径。")

        log("执行前景第二步: 并集(Union) (A=contrast_contour, B=tex_fore_conn)")
        count2, skip2 = logic_operation_folder(contrast_contour_dir, tex_fore_conn_dir, "并集", forecoarse_dir, prefix_len, log_callback=log)

        log("执行背景: 差集(A-B) (A=contrast_back, B=tex_back_conn)")
        count3, skip3 = logic_operation_folder(contrast_back_dir, tex_back_conn_dir, "差集(A-B)", backcoarse_dir, prefix_len, log_callback=log)

        self.master.after(0, lambda: self.logic_step1_A.set(contrast_fore_dir))
        self.master.after(0, lambda: self.logic_step1_B.set(contour_morph_dir))
        self.master.after(0, lambda: self.logic_step1_out.set(contrast_contour_dir))
        self.master.after(0, lambda: self.logic_step2_A.set(contrast_contour_dir))
        self.master.after(0, lambda: self.logic_step2_B.set(tex_fore_conn_dir))
        self.master.after(0, lambda: self.logic_step2_out.set(forecoarse_dir))
        self.master.after(0, lambda: self.logic_back_A.set(contrast_back_dir))
        self.master.after(0, lambda: self.logic_back_B.set(tex_back_conn_dir))
        self.master.after(0, lambda: self.logic_back_out.set(backcoarse_dir))

        # ===== AP Generate =====
        log("执行 AP Generate...")
        ap_output_dir = os.path.join(results_root, "AP")
        ap_fg_x = self.ap_fg_x_ratio.get()
        ap_fg_r = self.ap_fg_ratio.get()
        ap_bg_min = self.ap_bg_min_area.get()
        ap_bg_margin = self.ap_bg_margin.get()
        ap_bg_r = self.ap_bg_ratio.get()
        ap_bg_top = self.ap_bg_top_x.get()
        ap_pref = self.ap_prefix_len.get()

        try:
            basenames = process_ap_images(
                original_folder=input_dir,
                foreground_folder=forecoarse_dir,
                background_folder=backcoarse_dir,
                output_folder=ap_output_dir,
                fg_x_ratio=ap_fg_x,
                fg_ratio=ap_fg_r,
                bg_min_area=ap_bg_min,
                bg_margin=ap_bg_margin,
                bg_ratio=ap_bg_r,
                bg_top_x=ap_bg_top,
                prefix_len=ap_pref
            )
            log(f"AP Generate 完成，生成了 {len(basenames)} 组图像")
            self.master.after(0, lambda: self.ap_original_folder.set(input_dir))
            self.master.after(0, lambda: self.ap_foreground_folder.set(forecoarse_dir))
            self.master.after(0, lambda: self.ap_background_folder.set(backcoarse_dir))
            self.master.after(0, lambda: self.ap_output_folder.set(ap_output_dir))
            if basenames:
                self.master.after(0, lambda: self.load_ap_image_list(ap_output_dir, basenames))
        except Exception as e:
            log(f"AP Generate 失败: {str(e)}")
            import traceback
            traceback.print_exc()

        msg = f"全部处理完成！\n纹理、轮廓、衬度已保存。\n逻辑操作结果：\n前景步骤1: {count1} 张\n前景步骤2: {count2} 张\n背景: {count3} 张"
        self.master.after(0, lambda: messagebox.showinfo("完成", msg))
        log("全部处理完成！")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = TextureExtractionApp(root)
    root.mainloop()