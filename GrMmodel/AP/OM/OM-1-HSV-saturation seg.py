import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def process_images(input_folder, thresh1=100, thresh2=150, line_width=2):
    """
    对输入文件夹中的所有图像进行饱和度分割，生成前景/背景掩膜，并在原图上绘制所有轮廓。
    前景：饱和度 < thresh1 的区域 → 红色轮廓 + 前景掩膜
    背景：饱和度 > thresh2 的区域 → 绿色轮廓 + 背景掩膜
    所有输出文件保存在原图目录下的 coarse 文件夹中。
    """
    # 创建 coarse 子文件夹
    coarse_dir = os.path.join(input_folder, "coarse")
    os.makedirs(coarse_dir, exist_ok=True)

    # 支持的图像扩展名
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # 遍历文件夹
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(valid_exts):
            continue

        img_path = os.path.join(input_folder, filename)
        print(f"处理图像: {img_path}")

        # 读取图像（BGR格式）
        img = cv2.imread(img_path)
        if img is None:
            print(f"  无法读取图像，跳过: {img_path}")
            continue

        # 转换为HSV并提取饱和度通道
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]

        # 生成前景和背景二值掩膜（0或255）
        fore_mask = (s_channel < thresh1).astype(np.uint8) * 255
        back_mask = (s_channel > thresh2).astype(np.uint8) * 255

        # 构造基础文件名（不含扩展名）
        base_name = os.path.splitext(filename)[0]

        # 1. 保存前景掩膜
        fore_path = os.path.join(coarse_dir, f"{base_name}fore.png")
        cv2.imwrite(fore_path, fore_mask)
        print(f"  保存前景掩膜: {fore_path}")

        # 2. 保存背景掩膜
        back_path = os.path.join(coarse_dir, f"{base_name}back.png")
        cv2.imwrite(back_path, back_mask)
        print(f"  保存背景掩膜: {back_path}")

        # 3. 绘制轮廓叠加图
        overlay_img = img.copy()

        # 提取所有轮廓（RETR_LIST 包含所有轮廓，不建立层级关系）
        contours_fore, _ = cv2.findContours(fore_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, contours_fore, -1, (0, 0, 255), line_width)  # 红色

        contours_back, _ = cv2.findContours(back_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, contours_back, -1, (0, 255, 0), line_width)  # 绿色

        overlay_path = os.path.join(coarse_dir, f"{base_name}overlay.png")
        cv2.imwrite(overlay_path, overlay_img)
        print(f"  保存轮廓叠加图: {overlay_path}")

    print("全部处理完成！")

if __name__ == "__main__":
    # 弹出文件夹选择对话框
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="请选择包含图像的文件夹")
    if folder_selected:
        print(f"选择的文件夹: {folder_selected}")
        # 使用默认阈值（阈值1=100，阈值2=120，线宽2）
        process_images(folder_selected)
    else:
        print("未选择文件夹，程序退出。")


# import os
# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import filedialog
#
# # def process_images(input_folder, thresh1=100, thresh2=150, line_width=2):
# def process_images(input_folder, thresh1=140, thresh2=90, line_width=2):
#     """
#     对输入文件夹中的所有图像进行饱和度分割，提取前景和背景区域的轮廓，
#     并在原图上用红色和绿色线条绘制这些轮廓。
#     前景：饱和度 < thresh1 的区域 → 红色轮廓
#     背景：饱和度 > thresh2 的区域 → 绿色轮廓
#     结果保存在原图目录下的 coarse 文件夹中，文件名为“原图名称overlay.png”。
#     """
#     # 创建 coarse 子文件夹
#     coarse_dir = os.path.join(input_folder, "coarse")
#     os.makedirs(coarse_dir, exist_ok=True)
#
#     # 支持的图像扩展名
#     valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
#
#     # 遍历文件夹
#     for filename in os.listdir(input_folder):
#         if not filename.lower().endswith(valid_exts):
#             continue
#
#         img_path = os.path.join(input_folder, filename)
#         print(f"处理图像: {img_path}")
#
#         # 读取图像（BGR格式）
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"  无法读取图像，跳过: {img_path}")
#             continue
#
#         # 转换为HSV并提取饱和度通道
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         s_channel = hsv[:, :, 1]
#
#         # 生成前景和背景二值掩模（0或255）
#         fore_mask = (s_channel < thresh1).astype(np.uint8) * 255
#         back_mask = (s_channel > thresh2).astype(np.uint8) * 255
#
#         # 在原图副本上绘制轮廓
#         overlay_img = img.copy()
#
#         # --- 绘制前景轮廓（红色） ---
#         contours_fore, _ = cv2.findContours(fore_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(overlay_img, contours_fore, -1, (0, 0, 255), line_width)
#
#         # --- 绘制背景轮廓（绿色） ---
#         contours_back, _ = cv2.findContours(back_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(overlay_img, contours_back, -1, (0, 255, 0), line_width)
#
#         # 保存叠加图
#         base_name = os.path.splitext(filename)[0]
#         overlay_path = os.path.join(coarse_dir, f"{base_name}overlay.png")
#         cv2.imwrite(overlay_path, overlay_img)
#         print(f"  保存轮廓叠加图: {overlay_path}")
#
#     print("全部处理完成！")
#
# if __name__ == "__main__":
#     # 弹出文件夹选择对话框
#     root = tk.Tk()
#     root.withdraw()
#     folder_selected = filedialog.askdirectory(title="请选择包含图像的文件夹")
#     if folder_selected:
#         print(f"选择的文件夹: {folder_selected}")
#         # 使用默认阈值（阈值1=100，阈值2=120，线宽2）
#         process_images(folder_selected)
#         print("未选择文件夹，程序退出。")
#
#
# # import os
# # import cv2
# # import numpy as np
# # import tkinter as tk
# # from tkinter import filedialog
# #
# # def process_images(input_folder, thresh1=100, thresh2=150, alpha=0.5):
# #     """
# #     对输入文件夹中的所有图像进行饱和度分割，生成前景/背景掩模及叠加可视化图。
# #     前景：饱和度 < thresh1 的区域（红色半透明叠加）
# #     背景：饱和度 > thresh2 的区域（绿色半透明叠加）
# #     掩模和叠加图均保存在原图目录下的 coarse 文件夹中。
# #     """
# #     # 创建 coarse 子文件夹
# #     coarse_dir = os.path.join(input_folder, "coarse")
# #     os.makedirs(coarse_dir, exist_ok=True)
# #
# #     # 支持的图像扩展名
# #     valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
# #
# #     # 遍历文件夹
# #     for filename in os.listdir(input_folder):
# #         # 检查是否为支持的图像文件
# #         if not filename.lower().endswith(valid_exts):
# #             continue
# #
# #         img_path = os.path.join(input_folder, filename)
# #         print(f"处理图像: {img_path}")
# #
# #         # 读取图像 (BGR格式)
# #         img = cv2.imread(img_path)
# #         if img is None:
# #             print(f"  无法读取图像，跳过: {img_path}")
# #             continue
# #
# #         # 转换为HSV
# #         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# #         # 提取饱和度通道
# #         s_channel = hsv[:, :, 1]
# #
# #         # 前景掩模：饱和度 < thresh1
# #         fore_mask = (s_channel < thresh1).astype(np.uint8) * 255
# #
# #         # 背景掩模：饱和度 > thresh2
# #         back_mask = (s_channel > thresh2).astype(np.uint8) * 255
# #
# #         # 构造保存路径
# #         base_name = os.path.splitext(filename)[0]  # 去掉扩展名
# #         fore_path = os.path.join(coarse_dir, f"{base_name}fore.png")
# #         back_path = os.path.join(coarse_dir, f"{base_name}back.png")
# #
# #         # 保存掩模图像
# #         cv2.imwrite(fore_path, fore_mask)
# #         cv2.imwrite(back_path, back_mask)
# #
# #         print(f"  保存前景掩模: {fore_path}")
# #         print(f"  保存背景掩模: {back_path}")
# #
# #         # 生成叠加可视化图
# #         overlay_img = img.copy()  # 原图副本
# #         # 前景区域叠加红色半透明
# #         fore_bool = fore_mask > 0
# #         overlay_img[fore_bool] = (overlay_img[fore_bool] * (1 - alpha) +
# #                                   np.array([255, 255, 255]) * alpha).astype(np.uint8)
# #         # 背景区域叠加绿色半透明
# #         back_bool = back_mask > 0
# #         overlay_img[back_bool] = (overlay_img[back_bool] * (1 - alpha) +
# #                                   np.array([255, 0, 0]) * alpha).astype(np.uint8)
# #
# #         #
# #         # # 前景区域叠加红色半透明
# #         # fore_bool = fore_mask > 0
# #         # overlay_img[fore_bool] = (overlay_img[fore_bool] * (1 - alpha) +
# #         #                           np.array([0, 0, 255]) * alpha).astype(np.uint8)
# #         # # 背景区域叠加绿色半透明
# #         # back_bool = back_mask > 0
# #         # overlay_img[back_bool] = (overlay_img[back_bool] * (1 - alpha) +
# #         #                           np.array([0, 255, 0]) * alpha).astype(np.uint8)
# #
# #
# #         overlay_path = os.path.join(coarse_dir, f"{base_name}overlay.png")
# #         cv2.imwrite(overlay_path, overlay_img)
# #         print(f"  保存叠加图: {overlay_path}")
# #
# #     print("全部处理完成！")
# #
# # if __name__ == "__main__":
# #     # 弹出文件夹选择对话框
# #     root = tk.Tk()
# #     root.withdraw()  # 隐藏主窗口
# #     folder_selected = filedialog.askdirectory(title="请选择包含图像的文件夹")
# #     if folder_selected:
# #         print(f"选择的文件夹: {folder_selected}")
# #         # 使用默认阈值进行处理（阈值1=100，阈值2=120，透明度0.5）
# #         process_images(folder_selected)
# #     else:
# #         print("未选择文件夹，程序退出。")
# #
# # # import os
# # # import cv2
# # # import numpy as np
# # # import tkinter as tk
# # # from tkinter import filedialog
# # #
# # # def process_images(input_folder, thresh1=100, thresh2=120):
# # #     """
# # #     对输入文件夹中的所有图像进行饱和度分割，生成前景/背景掩模。
# # #     前景：饱和度 < thresh1 的区域
# # #     背景：饱和度 > thresh2 的区域
# # #     掩模保存在原图目录下的 coarse 文件夹中。
# # #     """
# # #     # 创建 coarse 子文件夹
# # #     coarse_dir = os.path.join(input_folder, "coarse")
# # #     os.makedirs(coarse_dir, exist_ok=True)
# # #
# # #     # 支持的图像扩展名
# # #     valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
# # #
# # #     # 遍历文件夹
# # #     for filename in os.listdir(input_folder):
# # #         # 检查是否为支持的图像文件
# # #         if not filename.lower().endswith(valid_exts):
# # #             continue
# # #
# # #         img_path = os.path.join(input_folder, filename)
# # #         print(f"处理图像: {img_path}")
# # #
# # #         # 读取图像 (BGR格式)
# # #         img = cv2.imread(img_path)
# # #         if img is None:
# # #             print(f"  无法读取图像，跳过: {img_path}")
# # #             continue
# # #
# # #         # 转换为HSV
# # #         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # #         # 提取饱和度通道
# # #         s_channel = hsv[:, :, 1]
# # #
# # #         # 前景掩模：饱和度 < thresh1
# # #         fore_mask = (s_channel < thresh1).astype(np.uint8) * 255
# # #
# # #         # 背景掩模：饱和度 > thresh2
# # #         back_mask = (s_channel > thresh2).astype(np.uint8) * 255
# # #
# # #         # 构造保存路径
# # #         base_name = os.path.splitext(filename)[0]  # 去掉扩展名
# # #         fore_path = os.path.join(coarse_dir, f"{base_name}fore.png")
# # #         back_path = os.path.join(coarse_dir, f"{base_name}back.png")
# # #
# # #         # 保存掩模图像
# # #         cv2.imwrite(fore_path, fore_mask)
# # #         cv2.imwrite(back_path, back_mask)
# # #
# # #         print(f"  保存前景掩模: {fore_path}")
# # #         print(f"  保存背景掩模: {back_path}")
# # #
# # #     print("全部处理完成！")
# # #
# # # if __name__ == "__main__":
# # #     # 弹出文件夹选择对话框
# # #     root = tk.Tk()
# # #     root.withdraw()  # 隐藏主窗口
# # #     folder_selected = filedialog.askdirectory(title="请选择包含图像的文件夹")
# # #     if folder_selected:
# # #         print(f"选择的文件夹: {folder_selected}")
# # #         # 使用默认阈值进行处理（阈值1=100，阈值2=120）
# # #         process_images(folder_selected)
# # #     else:
# # #         print("未选择文件夹，程序退出。")