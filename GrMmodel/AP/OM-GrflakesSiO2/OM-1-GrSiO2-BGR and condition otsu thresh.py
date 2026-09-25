import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
#
# # ------------------ SLG单层石墨烯AP配置参数 ------------------
# # BGR各通道百分位范围（例如 5 和 95 表示保留该通道中间90%的像素）
# B_LOW_PERCENTILE, B_HIGH_PERCENTILE =0.1, 9
# G_LOW_PERCENTILE, G_HIGH_PERCENTILE = 0, 9
# R_LOW_PERCENTILE, R_HIGH_PERCENTILE = 0, 9
# EXCLUDE_UPPER_THRESH = 130  ##（排除绝对灰度值高于此值的像素）
# EXCLUDE_Lower_THRESH = 55   ##（排除绝对灰度值低于此值的像素）
# # # 条件Otsu排除百分位（排除灰度值高于此百分位的像素）
# # OTSU_EXCLUDE_PERCENTILE = 90
# MIN_AREA = 100               # 前景种子连通区域最小面积（仅用于显示，不影响掩膜生成）
# DILATE_KERNEL_SIZE = 3       # 膨胀核大小（奇数）
# DILATE_ITERATIONS = 1        # 膨胀次数
# ----------------------------------------------
# -------------------------------------------


##1-4层石墨烯
# BGR各通道百分位范围（例如 5 和 95 表示保留该通道中间90%的像素）
B_LOW_PERCENTILE, B_HIGH_PERCENTILE =0, 9
G_LOW_PERCENTILE, G_HIGH_PERCENTILE = 0, 9
R_LOW_PERCENTILE, R_HIGH_PERCENTILE = 0, 9
EXCLUDE_UPPER_THRESH = 130  ##（排除绝对灰度值高于此值的像素）
EXCLUDE_Lower_THRESH = 40   ##（排除绝对灰度值低于此值的像素）
# # 条件Otsu排除百分位（排除灰度值高于此百分位的像素）
# OTSU_EXCLUDE_PERCENTILE = 90
MIN_AREA = 100               # 前景种子连通区域最小面积（仅用于显示，不影响掩膜生成）
DILATE_KERNEL_SIZE = 3       # 膨胀核大小（奇数）
DILATE_ITERATIONS = 1        # 膨胀次数

# 排除绝对灰度高的像素

# def otsu_after_excluding_high(image_gray, upper_thresh=EXCLUDE_UPPER_THRESH):
#     """条件Otsu分割（排除高亮像素后计算Otsu阈值）"""
#     valid_mask = image_gray <= upper_thresh
#     valid_pixels = image_gray[valid_mask]
#
#     if len(valid_pixels) == 0:
#         return np.zeros_like(image_gray), 0
#
#     hist = cv2.calcHist([valid_pixels], [0], None, [256], [0, 256])
#     hist = hist.ravel() / hist.sum()
#
#     best_thresh = 0
#     max_variance = 0
#     total_mean = np.sum(np.arange(256) * hist)
#     weight_bg = 0.0
#     mean_bg = 0.0
#
#     for t in range(256):
#         w_b = weight_bg
#         w_f = 1.0 - w_b
#         if w_b == 0 or w_f == 0:
#             weight_bg += hist[t]
#             mean_bg += t * hist[t]
#             continue
#         mean_b = mean_bg / w_b
#         mean_f = (total_mean - mean_bg) / w_f
#         variance_between = w_b * w_f * (mean_b - mean_f) ** 2
#         if variance_between > max_variance:
#             max_variance = variance_between
#             best_thresh = t
#         weight_bg += hist[t]
#         mean_bg += t * hist[t]
#
#     binary_mask = np.zeros_like(image_gray)
#     binary_mask[(image_gray <= best_thresh) & valid_mask] = 255
#     return binary_mask, best_thresh

def otsu_with_range(image_gray, lower_thresh=10, upper_thresh=200):
    """
    排除灰度值低于 lower_thresh 或高于 upper_thresh 的像素，
    对中间范围内的像素计算 Otsu 阈值并分割。
    """
    # 有效像素：灰度值在 [lower_thresh, upper_thresh] 内
    valid_mask = (image_gray >= lower_thresh) & (image_gray <= upper_thresh)
    valid_pixels = image_gray[valid_mask]

    if len(valid_pixels) == 0:
        return np.zeros_like(image_gray), 0

    # 后续 Otsu 计算与之前相同
    hist = cv2.calcHist([valid_pixels], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()

    best_thresh = 0
    max_variance = 0
    total_mean = np.sum(np.arange(256) * hist)
    weight_bg = 0.0
    mean_bg = 0.0

    for t in range(256):
        w_b = weight_bg
        w_f = 1.0 - w_b
        if w_b == 0 or w_f == 0:
            weight_bg += hist[t]
            mean_bg += t * hist[t]
            continue
        mean_b = mean_bg / w_b
        mean_f = (total_mean - mean_bg) / w_f
        variance_between = w_b * w_f * (mean_b - mean_f) ** 2
        if variance_between > max_variance:
            max_variance = variance_between
            best_thresh = t
        weight_bg += hist[t]
        mean_bg += t * hist[t]

    # 应用阈值：最终前景为灰度在 [lower_thresh, best_thresh] 之间的像素
    # 注意：根据前景是暗还是亮，可调整条件
    binary_mask = np.zeros_like(image_gray)
    binary_mask[(image_gray >= lower_thresh) & (image_gray <= best_thresh)] = 255
    return binary_mask, best_thresh

def process_folder(input_dir):
    """批量处理文件夹中的图像，保存Otsu、动态BGR掩膜及种子区域"""
    output_dir = os.path.join(input_dir, "BGRwithGraycontrastotsu")
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(valid_exts):
            continue
        img_path = os.path.join(input_dir, filename)
        print(f"处理图像: {img_path}")

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("  无法读取，跳过")
            continue

        base_name = os.path.splitext(filename)[0]

        # 1. 条件Otsu得到A
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        A_binary, otsu_thresh = otsu_with_range(gray, EXCLUDE_Lower_THRESH, EXCLUDE_UPPER_THRESH)
        A_bool = A_binary > 0

        # 2. 动态BGR范围掩膜B
        B_vals = img[:, :, 0]
        G_vals = img[:, :, 1]
        R_vals = img[:, :, 2]

        # 分别计算每个通道的百分位阈值
        b_low = np.percentile(B_vals, B_LOW_PERCENTILE)
        b_high = np.percentile(B_vals, B_HIGH_PERCENTILE)
        g_low = np.percentile(G_vals, G_LOW_PERCENTILE)
        g_high = np.percentile(G_vals, G_HIGH_PERCENTILE)
        r_low = np.percentile(R_vals, R_LOW_PERCENTILE)
        r_high = np.percentile(R_vals, R_HIGH_PERCENTILE)

        print(f"  动态BGR范围: B[{b_low:.0f}, {b_high:.0f}], G[{g_low:.0f}, {g_high:.0f}], R[{r_low:.0f}, {r_high:.0f}]")

        B_bool = (B_vals >= b_low) & (B_vals <= b_high) & \
                 (G_vals >= g_low) & (G_vals <= g_high) & \
                 (R_vals >= r_low) & (R_vals <= r_high)

        # 保存Otsu和BGR掩膜

        otsupath=os.path.join(output_dir, 'otsu')
        os.makedirs(otsupath, exist_ok=True)
        cv2.imwrite(os.path.join(otsupath ,f"{base_name}.jpg"), A_binary)

        BGRpath=os.path.join(output_dir, 'BGR')
        os.makedirs(BGRpath, exist_ok=True)
        cv2.imwrite(os.path.join(BGRpath ,f"{base_name}.jpg"), (B_bool.astype(np.uint8) * 255))



        # 3. 前景种子区域（A∩B）
        coarseforeregion = A_bool & B_bool
        coreseforepath=os.path.join(output_dir, 'coresefore')
        os.makedirs(coreseforepath, exist_ok=True)
        cv2.imwrite(os.path.join(coreseforepath ,f"{base_name}.jpg"),
                    (coarseforeregion.astype(np.uint8) * 255))


        coarseforeregionerode = cv2.erode(coarseforeregion.astype(np.uint8), kernel, iterations=DILATE_ITERATIONS).astype(bool)
        coreseforeerodepath = os.path.join(output_dir, 'coreseforeerode')
        os.makedirs(coreseforeerodepath, exist_ok=True)
        cv2.imwrite(os.path.join(coreseforeerodepath ,f"{base_name}.jpg"),
                    (coarseforeregionerode.astype(np.uint8) * 255))

        # coarseforeregionerodeslight = cv2.erode(coarseforeregion.astype(np.uint8),  (2,2),  1)
        # coarsebackregion = ~coarseforeregionerodeslight

        coarsebackregion = ~coarseforeregion
        coresebackpath = os.path.join(output_dir, 'coreseback')
        os.makedirs(coresebackpath, exist_ok=True)
        cv2.imwrite(os.path.join(coresebackpath ,f"{base_name}.jpg"),
                    (coarsebackregion.astype(np.uint8) * 255))
        coarsebackregionerode = cv2.morphologyEx(coarsebackregion.astype(np.uint8), cv2.MORPH_ERODE, kernel, iterations=DILATE_ITERATIONS).astype(bool)

        coarsebackerode = os.path.join(output_dir, 'coarsebackerode')
        os.makedirs(coarsebackerode, exist_ok=True)
        cv2.imwrite(os.path.join(coarsebackerode, f"{base_name}.jpg"),
                    (coarsebackregionerode.astype(np.uint8) * 255))




        # 4. 背景种子区域（初始B∩~A，可能被膨胀修正）
        # backseed_region = B_bool & (~A_bool)
        #
        # # 5. 膨胀规则：若B∩~A像素数 < A∩B像素数，则对B膨胀后取与~A的交集
        # if np.count_nonzero(backseed_region) < np.count_nonzero(foreseed_region):
        #     print(f"  触发膨胀规则: B∩~A像素({np.count_nonzero(backseed_region)}) < A∩B像素({np.count_nonzero(foreseed_region)})")
        #     B_dilated = cv2.dilate(B_bool.astype(np.uint8), kernel, iterations=DILATE_ITERATIONS).astype(bool)
        #     backseed_region = B_dilated & (~A_bool)
        #     print(f"  膨胀后背景种子像素数: {np.count_nonzero(backseed_region)}")
        # else:
        #     print(f"  无需膨胀: B∩~A像素({np.count_nonzero(backseed_region)}) >= A∩B像素({np.count_nonzero(foreseed_region)})")
        #
        # # 保存背景种子区域（可能已更新）
        # cv2.imwrite(os.path.join(output_dir, f"{base_name}_backseedregion.jpg"),
        #             (backseed_region.astype(np.uint8) * 255))

        print(f"  已保存Otsu、BGR、种子区域结果")

    print("全部处理完成！")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title="请选择包含原始图像的文件夹")
    if input_dir:
        print(f"输入文件夹: {input_dir}")
        process_folder(input_dir)
    else:
        print("未选择文件夹，程序退出。")