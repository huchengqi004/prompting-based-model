#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立脚本：带比例的 Otsu 阈值分割
- 用户通过 filedialog 选择输入图像文件夹
- 用户输入一个或多个比例系数（逗号分隔），例如：1.0,1.3
- 对每张图像执行：Otsu 阈值 * 比例 → 二值化 → 可选反转 → 可选形态学膨胀
- 结果自动保存到输入文件夹下的 scaled_otsu_results/<ratio>/ 子文件夹中
"""

import os
import sys
import cv2
import glob
import tkinter as tk
from tkinter import filedialog


# ==================== 用户可调参数 ====================
DEFAULT_RATIOS = [1.0, 1.3]      # 默认比例系数（用户可在控制台输入覆盖）
INVERT = True                    # 是否反转二值化结果（True: 白色为背景；False: 白色为前景）
APPLY_MORPH = False              # 是否对二值结果做形态学膨胀
MORPH_SHAPE = 'rect'             # 'rect' / 'ellipse' / 'cross'
MORPH_SIZE = 3                   # 形态学核尺寸
MORPH_ITER = 1                   # 形态学迭代次数
OUTPUT_SUBFOLDER = "scaled_otsu_results"
# =====================================================


def select_folder(title="请选择文件夹"):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder


def apply_morphology(binary_img, morph_shape, morph_size, morph_iter):
    if morph_shape is None or morph_shape == 'None':
        return binary_img
    shape_map = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS,
    }
    shape = shape_map.get(morph_shape, cv2.MORPH_RECT)
    kernel = cv2.getStructuringElement(shape, (morph_size, morph_size))
    return cv2.dilate(binary_img, kernel, iterations=morph_iter)


def process_scaled_otsu(input_dir, output_dir, ratios, invert=True,
                        apply_morph=False, morph_shape='rect',
                        morph_size=3, morph_iter=1):
    """
    对 input_dir 中所有图像执行带比例的 Otsu 阈值分割。
    每个比例对应 output_dir 下的一个子文件夹。
    """
    os.makedirs(output_dir, exist_ok=True)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"输入文件夹 {input_dir} 中未找到图像文件。")
        return

    print(f"共找到 {len(image_paths)} 张图像，将处理 {len(ratios)} 个比例系数。")

    # 为每个比例创建子文件夹
    for ratio in ratios:
        ratio_dir = os.path.join(output_dir, f"ratio_{ratio}")
        os.makedirs(ratio_dir, exist_ok=True)

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx}/{total}] 处理: {os.path.basename(img_path)}")

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"    警告：无法读取 {img_path}，跳过。")
            continue

        otsu_th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        for ratio in ratios:
            new_th = float(np.clip(ratio * otsu_th, 0, 255))
            _, binary = cv2.threshold(img, new_th, 255, cv2.THRESH_BINARY)

            if invert:
                binary = cv2.bitwise_not(binary)

            if apply_morph:
                binary = apply_morphology(binary, morph_shape, morph_size, morph_iter)

            out_name = f"{base}_th{int(round(new_th))}.png"
            out_path = os.path.join(output_dir, f"ratio_{ratio}", out_name)
            cv2.imwrite(out_path, binary)

    print(f"\n处理完成，结果已保存至: {output_dir}")


def main():
    print("请在弹出的对话框中选择输入图像文件夹...")
    input_dir = select_folder("请选择输入图像文件夹")
    if not input_dir:
        print("未选择文件夹，程序退出。")
        sys.exit(1)

    print(f"输入文件夹: {input_dir}")

    # 用户输入比例系数
    user_input = input(
        f"请输入比例系数（逗号分隔，直接回车使用默认 {DEFAULT_RATIOS}）："
    ).strip()
    if user_input:
        try:
            ratios = [float(x.strip()) for x in user_input.split(',') if x.strip()]
        except ValueError:
            print("输入无效，使用默认比例系数。")
            ratios = DEFAULT_RATIOS
    else:
        ratios = DEFAULT_RATIOS

    print(f"使用的比例系数: {ratios}")

    # 输出文件夹：输入文件夹下的 scaled_otsu_results
    output_dir = os.path.join(input_dir, OUTPUT_SUBFOLDER)
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出文件夹: {output_dir}")

    process_scaled_otsu(
        input_dir=input_dir,
        output_dir=output_dir,
        ratios=ratios,
        invert=INVERT,
        apply_morph=APPLY_MORPH,
        morph_shape=MORPH_SHAPE,
        morph_size=MORPH_SIZE,
        morph_iter=MORPH_ITER,
    )


if __name__ == "__main__":
    import numpy as np  # 放在这里避免顶层 import 未使用告警
    main()