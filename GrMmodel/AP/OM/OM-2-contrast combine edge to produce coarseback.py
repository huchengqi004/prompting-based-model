# # # edgeimagepath = "B:/software-patent/graphenefraction/OM/AP-22_rn3_l120_w13_rectsink50/e1_n_low20_high100_BPF contour extraction.jpg"
# # # contrastimagepath = 'B:/software-patent/graphenefraction/OM/AP-22-0/22-HSVthreshold148.png'
# #
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")
#
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def retain_inside_region(target_img, restrict_mask):
#     """
#     保留位于限制掩膜连通区域内部的像素
#
#     参数:
#         target_img: 目标图像 (要处理的图像)
#         restrict_mask: 限制掩膜 (二值图像，白色表示要保留的区域)
#
#     返回:
#         处理后的图像 (只保留限制区域内部的像素)
#     """
#     # 确保限制掩膜是二值图像
#     if len(restrict_mask.shape) > 2:
#         restrict_mask = cv2.cvtColor(restrict_mask, cv2.COLOR_BGR2GRAY)
#
#     # 二值化限制掩膜
#     _, binary_mask = cv2.threshold(restrict_mask, 1, 255, cv2.THRESH_BINARY)
#
#     # 查找连通区域
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
#
#     # 创建结果图像
#     if len(target_img.shape) == 3:  # 彩色图像
#         result = np.zeros_like(target_img)
#         mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
#     else:  # 灰度图像
#         result = np.zeros_like(target_img)
#         mask_3ch = binary_mask
#
#     # 遍历所有连通区域（跳过背景区域，索引0）
#     for i in range(1, num_labels):
#         # 创建当前区域的掩膜
#         component_mask = np.uint8(labels == i) * 255
#
#         # 对当前区域应用掩膜
#         if len(target_img.shape) == 3:
#             component_mask_3ch = cv2.cvtColor(component_mask, cv2.COLOR_GRAY2BGR)
#             component_result = cv2.bitwise_and(target_img, component_mask_3ch)
#         else:
#             component_result = cv2.bitwise_and(target_img, component_mask)
#
#         # 将当前区域结果添加到最终结果
#         result = cv2.add(result, component_result)
#
#     return result, binary_mask
#
#
# def visualize_results(target_img, restrict_mask, result_img):
#     """可视化结果"""
#     plt.figure(figsize=(15, 10))
#
#     plt.subplot(231)
#     plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB) if len(target_img.shape) == 3 else target_img,
#                cmap='gray')
#     plt.title('目标图像')
#     plt.axis('off')
#
#     plt.subplot(232)
#     plt.imshow(restrict_mask, cmap='gray')
#     plt.title('限制掩膜')
#     plt.axis('off')
#
#     plt.subplot(233)
#     plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB) if len(result_img.shape) == 3 else result_img, cmap='gray')
#     plt.title('保留内部像素的结果')
#     plt.axis('off')
#
#     # 显示叠加效果
#     overlay = cv2.addWeighted(
#         cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB) if len(target_img.shape) == 3 else target_img,
#         0.7,
#         cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB) if len(result_img.shape) == 3 else result_img,
#         0.3,
#         0
#     )
#
#     plt.subplot(234)
#     plt.imshow(overlay, cmap='gray')
#     plt.title('叠加效果')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.savefig('inside_region_result.jpg', dpi=300, bbox_inches='tight')
#     plt.show()
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 1. 读取图像
#
#     # #轮廓提取结果
#     # target_img = cv2.imread(
#     #     # "B:/software-patent/graphenefraction/OM/AP-22_rn3_l120_w13_rectsink50/e1_n_low20_high100_BPF contour extraction.jpg")
#     #     "B:/software-patent/graphenefraction/OM/AP-22-0/e1_n100_low20_high100_BPF contour extraction.jpg")
#     # #衬度提取结果
#     # restrict_mask = cv2.imread('B:/software-patent/graphenefraction/OM/AP-22-0/22-HSVthreshold97.png',
#     #                            cv2.IMREAD_GRAYSCALE)
#     # savepath = "B:/software-patent/graphenefraction/OM/AP-22_rn3_l120_w13_rectsink50/restricted_result.jpg"
#     # #把边缘提取结果限制在HSV根据饱和度阈值提取结果的膨胀结果内部，防止褶皱衬度干扰。
#     # combinepath = "B:/software-patent/graphenefraction/OM/AP-22_rn3_l120_w13_rectsink50/combined_result.jpg"
#     # combinemorphpath = "B:/software-patent/graphenefraction/OM/AP-22_rn3_l120_w13_rectsink50/combinedmorph_result.jpg"
#     # # combinefillpath = "B:/software-patent/graphenefraction/OM/AP-22_rn3_l120_w13_rectsink50/combinedfill_result.jpg"
#
#
#     #轮廓提取结果
#     target_img = cv2.imread(
#         # "B:/software-patent/graphenefraction/OM/AP-22_rn3_l120_w13_rectsink50/e1_n_low20_high100_BPF contour extraction.jpg")
#         r"A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\22_LPF_r1_n100_edge.jpg")
#     #衬度提取结果
#     restrict_mask = cv2.imread(r'A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\22saturationfore100back150.png',
#                                cv2.IMREAD_GRAYSCALE)
#     savepath = r"A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\22restricted_result.jpg"
#     #把边缘提取结果限制在HSV根据饱和度阈值提取结果的膨胀结果内部，防止褶皱衬度干扰。
#     combinepath = r"A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\22combined_result.jpg"
#     combinemorphpath = r"A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\22combinedmorph_result.jpg"
#     # combinefillpath = "B:/software-patent/graphenefraction/OM/AP-22_rn3_l120_w13_rectsink50/combinedfill_result.jpg"
#
#
#
#     # 2. 对限制掩膜进行膨胀（可选）
#     iter = 5
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#
#     dilatecontrast = cv2.dilate(restrict_mask, kernel, iterations=1)
#     contrastdilatepath = "B:/software-patent/graphenefraction/OM/AP-22_rn3_l120_w13_rectsink50/contrastdilate.jpg"
#     cv2.imwrite(contrastdilatepath, dilatecontrast)
#
#
#
#     dilated_mask = cv2.dilate(restrict_mask, kernel, iterations=iter)
#
#     # 3. 保留限制区域内部的像素
#     result_img, binary_mask = retain_inside_region(target_img, dilated_mask)    #保留HSV饱和度阈值结果内部的边缘特征，定义为result_img.
#
#     # 4. 可视化结果
#     visualize_results(target_img, dilated_mask, result_img)
#     cv2.imwrite(savepath, result_img)
#     print(f"轮廓位于目标区域的结果已保存至: {savepath}")
#
#     # 5. 保存结果
#
#     # 确保图像尺寸相同
#     if restrict_mask.shape[:2] != result_img.shape[:2]:
#         print(f"调整图像尺寸: restrict_mask {restrict_mask.shape} -> result_img {result_img.shape}")
#         img1 = cv2.resize(restrict_mask, (result_img.shape[1], result_img.shape[0]))
#
#     # 确保通道数相同
#     if len(restrict_mask.shape) != len(result_img.shape):
#         print(f"调整通道数: restrict_mask {len(restrict_mask.shape)}通道 -> result_img {len(result_img.shape)}通道")
#         if len(restrict_mask.shape) == 2:  # restrict_mask是灰度图，result_img是彩色图
#             restrict_mask = cv2.cvtColor(restrict_mask, cv2.COLOR_GRAY2BGR)
#         else:  # restrict_mask是彩色图，result_img是灰度图
#             result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
#     # 简单像素相加
#     combineresult = cv2.add(restrict_mask, result_img)
# #HSV饱和度阈值分割结果restrict_mask限制的边缘特征result_img与阈值分割结果restrict_mask进行叠加，作为基础前景。对前景进行形态学操作作为最终前景。
#
#
#     # 1. 转换为灰度图（如果是三通道）
#     if len(combineresult.shape) == 3:
#         print("转换为三通道。")
#         combineresult = cv2.cvtColor(combineresult, cv2.COLOR_BGR2GRAY)
#
#     # # 2. 转换为 uint8 并二值化
#     # combineresult = cv2.convertScaleAbs(combineresult)  # 自动映射到0-255
#     # _, combineresult = cv2.threshold(combineresult, 127, 255, cv2.THRESH_BINARY)
#     combineresult_fill = combineresult.astype(np.uint8)
#
#     # 3. 调用 findContours
#     contours, _ = cv2.findContours(combineresult_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # 创建掩膜并填充Contour内部
#     contourmask = np.zeros(restrict_mask.shape, dtype=np.uint8)
#     cv2.drawContours(contourmask, contours[1:], -1, 255, thickness=-1)
#
#     # cv2.imwrite(combinefillpath, contourmask)
#     # print(f"结合结果填充图像已保存至: {combinefillpath}")
#
#     iter2 = 1
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     morph_mask = cv2.dilate(combineresult, kernel2, iterations=4)
#     # morph_mask = cv2.erode(morph_mask, kernel2, iterations=2)
#
#     # morph_mask = cv2.morphologyEx(combineresult, cv2.MORPH_CLOSE, kernel, iterations=2)
#
#     cv2.imwrite(combinepath, combineresult)
#
#     print(f"结合结果已保存至: {combinepath}")
#
#     cv2.imwrite(combinemorphpath, morph_mask)
#     print(f"结合结果膨胀图像已保存至: {combinemorphpath}")
#
#

import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import re
import tkinter as tk
from tkinter import filedialog

warnings.filterwarnings("ignore")


def retain_inside_region(target_img, restrict_mask):
    """
    保留位于限制掩膜连通区域内部的像素
    """
    if len(restrict_mask.shape) > 2:
        restrict_mask = cv2.cvtColor(restrict_mask, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(restrict_mask, 1, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    if len(target_img.shape) == 3:
        result = np.zeros_like(target_img)
    else:
        result = np.zeros_like(target_img)

    for i in range(1, num_labels):
        component_mask = np.uint8(labels == i) * 255
        if len(target_img.shape) == 3:
            component_mask_3ch = cv2.cvtColor(component_mask, cv2.COLOR_GRAY2BGR)
            component_result = cv2.bitwise_and(target_img, component_mask_3ch)
        else:
            component_result = cv2.bitwise_and(target_img, component_mask)
        result = cv2.add(result, component_result)

    return result, binary_mask


def extract_key(filename, n):
    """
    提取文件名中第一个连续数字串的前 n 位作为匹配键。
    """
    digits = re.findall(r'\d+', filename)
    if digits:
        first = digits[0]
        if len(first) >= n:
            return first[:n]
    return None

#
# def process_pair(target_path, mask_path, output_dir, key, kernel_comp, iterations_edges, iterations_contouredges):
#     """
#     处理一对图像，执行原脚本操作，并额外生成补集膨胀图。
#     """
#     target_img = cv2.imread(target_path)
#     restrict_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#
#     if target_img is None:
#         print(f"无法读取目标图像: {target_path}")
#         return
#     if restrict_mask is None:
#         print(f"无法读取限制掩膜: {mask_path}")
#         return
#
#     # 确保尺寸一致
#     if target_img.shape[:2] != restrict_mask.shape[:2]:
#         restrict_mask = cv2.resize(restrict_mask, (target_img.shape[1], target_img.shape[0]),
#                                    interpolation=cv2.INTER_NEAREST)
#
#     # ---------- 原脚本核心操作 ----------
#     iter_dilate = 5
#     kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     dilated_mask = cv2.dilate(restrict_mask, kernel_dilate, iterations=iter_dilate)
#
#     result_img, binary_mask = retain_inside_region(target_img, dilated_mask)
#
#     save_result_path = os.path.join(output_dir, f"{key}_restricted_result.jpg")
#     cv2.imwrite(save_result_path, result_img)
#     print(f"轮廓位于目标区域的结果已保存至: {save_result_path}")
#
#     # 调整通道并叠加
#     restrict_mask_bgr = restrict_mask
#     result_img_bgr = result_img
#     if len(restrict_mask.shape) != len(result_img.shape):
#         if len(restrict_mask.shape) == 2:
#             restrict_mask_bgr = cv2.cvtColor(restrict_mask, cv2.COLOR_GRAY2BGR)
#         else:
#             result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
#
#     if restrict_mask_bgr.shape[:2] != result_img_bgr.shape[:2]:
#         restrict_mask_bgr = cv2.resize(restrict_mask_bgr, (result_img_bgr.shape[1], result_img_bgr.shape[0]))
#
#     # combineresult = cv2.add(restrict_mask_bgr, result_img_bgr)
#
#     result_img_bgr_dilate = cv2.dilate(result_img_bgr, kernel_comp, iterations=iterations_edges)
#     union = cv2.bitwise_or(restrict_mask_bgr, result_img_bgr_dilate)
#     complement = cv2.bitwise_not(union)
#     complement = cv2.erode(complement, kernel_comp, iterations=iterations_contouredges)
#
#     comp_path = os.path.join(output_dir, f"{key}_complement.jpg")
#     cv2.imwrite(comp_path, complement)
#     print(f"补集膨胀图像已保存至: {comp_path}")
#
#     #
#     # if len(combineresult.shape) == 3:
#     #     combineresult = cv2.cvtColor(combineresult, cv2.COLOR_BGR2GRAY)
#     #
#     # combineresult_fill = combineresult.astype(np.uint8)
#     #
#     # contours, _ = cv2.findContours(combineresult_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # contourmask = np.zeros(restrict_mask.shape, dtype=np.uint8)
#     # if len(contours) > 1:
#     #     cv2.drawContours(contourmask, contours[1:], -1, 255, thickness=-1)
#     #
#     # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     # morph_mask = cv2.dilate(combineresult, kernel2, iterations=4)
#     #
#     # combine_path = os.path.join(output_dir, f"{key}_combined_result.jpg")
#     # cv2.imwrite(combine_path, combineresult)
#     # print(f"结合结果已保存至: {combine_path}")
#     #
#     # morph_path = os.path.join(output_dir, f"{key}_combinedmorph_result.jpg")
#     # cv2.imwrite(morph_path, morph_mask)
#     # print(f"结合结果膨胀图像已保存至: {morph_path}")
def process_pair(target_path, mask_path, output_dir, key, kernel_size, iterations_contouredges):
    target_img = cv2.imread(target_path)
    restrict_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if target_img is None:
        print(f"无法读取目标图像: {target_path}")
        return
    if restrict_mask is None:
        print(f"无法读取限制掩膜: {mask_path}")
        return

    if target_img.shape[:2] != restrict_mask.shape[:2]:
        restrict_mask = cv2.resize(restrict_mask, (target_img.shape[1], target_img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

    # 原脚本核心操作
    iter_dilate = 5
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(restrict_mask, kernel_dilate, iterations=iter_dilate)

    result_img, binary_mask = retain_inside_region(target_img, dilated_mask)

    save_result_path = os.path.join(output_dir, f"{key}_restricted_result.jpg")
    cv2.imwrite(save_result_path, result_img)
    print(f"轮廓位于目标区域的结果已保存至: {save_result_path}")

    # 统一通道
    if len(result_img.shape) == 2:
        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    else:
        result_img_bgr = result_img

    if len(target_img.shape) == 2:
        target_img_bgr = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
    else:
        target_img_bgr = target_img

    if target_img_bgr.shape[:2] != result_img_bgr.shape[:2]:
        target_img_bgr = cv2.resize(target_img_bgr, (result_img_bgr.shape[1], result_img_bgr.shape[0]))

    # 构造形态学核
    kernel_comp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 1) 轮廓膨胀
    result_img_bgr_dilate = cv2.dilate(result_img_bgr, kernel_dilate, iterations=1)


    # 转为三通道（如果原来是灰度）
    if len(restrict_mask.shape) == 2:
        restrict_mask_bgr = cv2.cvtColor(restrict_mask, cv2.COLOR_GRAY2BGR)
    else:
        restrict_mask_bgr = restrict_mask.copy()
    # 将 result_img_bgr_dilate 也转为三通道（如果原来是灰度）
    if len(result_img_bgr_dilate.shape) == 2:
        result_img_bgr_dilate = cv2.cvtColor(result_img_bgr_dilate, cv2.COLOR_GRAY2BGR)

    # 2) 与 target_dir 中图像取并集（注意：这里用 target_img_bgr，不是 restrict_mask_bgr）
    union = cv2.bitwise_or(restrict_mask_bgr, result_img_bgr_dilate)
    # 3) 取反
    complement = cv2.bitwise_not(union)
    # 4) 腐蚀
    if iterations_contouredges > 0:
        complement = cv2.erode(complement, kernel_comp, iterations=iterations_contouredges)

    comp_path = os.path.join(output_dir, f"{key}_coarseback.jpg")
    cv2.imwrite(comp_path, complement)
    print(f"补集图像已保存至: {comp_path}")
if __name__ == "__main__":
    # 隐藏 tkinter 主窗口
    root = tk.Tk()
    root.withdraw()

    # # 用户通过文件对话框选择文件夹
    # target_dir = filedialog.askdirectory(title="请选择目标图像文件夹（轮廓提取结果）")
    # if not target_dir:
    #     print("未选择目标图像文件夹，程序退出。")
    #     exit()
    #
    # mask_dir = filedialog.askdirectory(title="请选择限制掩膜图像文件夹（衬度提取结果）")
    # if not mask_dir:
    #     print("未选择限制掩膜文件夹，程序退出。")
    #     exit()
    # # 控制台输入匹配位数和膨胀参数
    # n = int(input("请输入匹配位数 N (例如 2): ").strip())
    # kernel_size = int(input("请输入补集膨胀核尺寸 (例如 5): ").strip())
    # iterations = int(input("请输入补集膨胀次数 (例如 4): ").strip())


    ####LPF生成的边缘
    target_dir = r'A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\OMimage\image\edge'

    ####饱和度阈值化提取的衬度结果。
    mask_dir = r'A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\OMimage\image\saturation'

    # 输出文件夹自动设置为 target_dir 下的 "AP generated results"
    output_dir = os.path.join(target_dir, "AP generated results")
    os.makedirs(output_dir, exist_ok=True)
    n = 2   #图像名称前几位字符匹配
    kernel_size = 5
    iterations_contouredges = 2


    # 构建目标图像字典
    target_dict = {}
    for f in os.listdir(target_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            key = extract_key(f, n)
            if key:
                target_dict[key] = os.path.join(target_dir, f)

    # 构建限制掩膜字典
    mask_dict = {}
    for f in os.listdir(mask_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            key = extract_key(f, n)
            if key:
                mask_dict[key] = os.path.join(mask_dir, f)

    # 匹配并处理
    matched_count = 0
    for key, target_path in target_dict.items():
        if key in mask_dict:
            mask_path = mask_dict[key]
            print(f"\n匹配到: {key} -> {os.path.basename(target_path)} + {os.path.basename(mask_path)}")
            process_pair(target_path, mask_path, output_dir, key, kernel_size,  iterations_contouredges)


            matched_count += 1
        else:
            print(f"未找到匹配: {key} (目标文件: {os.path.basename(target_path)})")

    print(f"\n处理完成，共匹配并处理 {matched_count} 对图像。")
    print(f"所有结果已保存至: {output_dir}")