#https://zhuanlan.zhihu.com/p/158742900.

# image_path = "B:/software-patent/graphenefraction/AFM/try/10min-15du.0_00002.jpg"
# image_path = "B:/software-patent/graphenefraction/OM/AP-22-0/22.png"
# image_path = "B:/code/GrSAM/data/Test/label/F1/middle/FT-Filter/F1.jpg"
# imagesave = os.path.join(os.path.dirname(image_path), ('FFT-' + image_path.split("/")[-1].split(".")[0]))




import warnings
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin, radians
warnings.filterwarnings("ignore")
from tkinter import filedialog
import os
from tkinter import filedialog

def auto_canny_threshold_texture(gray_img, high_percentile=0.9, low_percentile=0.8):
    """
    基于梯度幅值分位数自动计算 Canny 阈值。
    返回：低阈值, 高阈值, 梯度幅值图（可选）
    """
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

# 确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

filepath = filedialog.askdirectory(title="请选择需要进行傅里叶变换低通滤波的图像所在的文件夹：")
imagesave = filepath + '-FFT'
if not os.path.exists(imagesave):
    os.makedirs(imagesave)
print("输出文件所在路径为：", imagesave)

r = 1  # 默认为1，遮挡型遮罩半径值,越大，对原图像的平滑效果越明显，更少提取到无用边缘（更不容易提取到目标物内部纹理等边缘）。
r_noise = 100  # 默认为100，频域空间透过型圆形蒙版的半径,用于滤除噪声
# r_noise值越大，过滤的高频噪声越少，图像边缘信息损失得少，edge提取结果较完整；
# r_noise值越小，过滤的噪声越多但是图像边缘信息损失得越多。

for filename in os.listdir(filepath):
    image_path = filepath + "/" + filename
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # center
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 输出为复数形式。
    dft_shift = np.fft.fftshift(dft)

    # 高通滤波器HPF掩码，用于提取边缘轮廓。 Circular HPF mask, center circle is 0, remaining all ones
    mask = np.ones((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    # 将 mask 中圆形区域内的元素置为 0
    mask[mask_area] = 0

    # 低通滤波器LPF掩码，去除背景噪声。
    mask_0 = np.zeros((rows, cols, 2), np.uint8)
    mask_area_0 = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_noise * r_noise
    mask_0[mask_area_0] = 1
    # 对原图进行LPF，结合阈值分割提取衬度特征
    fshift_0 = dft_shift * mask_0
    fshift_mask_mag_0 = 2000 * np.log(cv2.magnitude(fshift_0[:, :, 0], fshift_0[:, :, 1]))
    f_ishift_0 = np.fft.ifftshift(fshift_0)
    img_back_0 = cv2.idft(f_ishift_0)
    img_back_0 = cv2.magnitude(img_back_0[:, :, 0], img_back_0[:, :, 1])  # 滤波滤掉的部分。
    # cv2.imwrite((imagesave + "/" + "e" + str(r) + "_n" + str(r_noise) + "_LPF.jpg"), img_back_0)
    # 将 img_back 转换为 uint8 类型，并将范围归一化到 0 到 255
    img_back_filter_0 = cv2.normalize(img_back_0, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite((imagesave + "/" + filename.split(".")[0] + "-r" + str(r) + "_n" + str(r_noise) + "_LPF.jpg"), img_back_filter_0)  # 过滤噪声，用于衬度特征

    # edges = cv2.Canny(img_back_filter_0, 25, 60, apertureSize=3, L2gradient=False)   #固定参数canny提取

    # 自动计算阈值（默认 high_percentile=0.9, low_percentile=0.5）
    low_thresh, high_thresh, _ = auto_canny_threshold_texture(img_back_filter_0)
    edges_LPF = cv2.Canny(img_back_filter_0, low_thresh, high_thresh, apertureSize=3, L2gradient=False)
    LPFedge_save_path = os.path.join(imagesave, filename.split(".")[0] + f"_r{r}_n{r_noise}_edge.jpg")
    cv2.imwrite(LPFedge_save_path, edges_LPF)
    # 保存边缘图
    imgedge_save_path = os.path.join(imagesave, filename.split(".")[0] + f"spatial_edge.jpg")
    low_thresh_spatial, high_thresh_spatial, _ = auto_canny_threshold_texture(img_back_filter_0)
    edges_image = cv2.Canny(img, low_thresh_spatial, high_thresh_spatial, apertureSize=3, L2gradient=False)
    cv2.imwrite(imgedge_save_path, edges_image)
    print(f"边缘图已保存: {LPFedge_save_path}")



    # # 对原图进行HPF
    # fshift = dft_shift * mask
    # #这个乘法操作是逐元素相乘（element-wise multiplication），也就是对于 dft_shift 和 mask 中相同位置的元素进行相乘。
    # # 当 mask 中某个位置的元素为 0 时，该位置对应的 dft_shift 元素会被置为0，这意味着该频率成分被抑制。
    # # 当 mask 中某个位置的元素为 1 时，该位置对应的 dft_shift 元素保持不变，意味着该频率成分被保留。
    # #dft_shift 是经过傅里叶变换并将零频率分量移到中心后的频域图像。它是一个形状为 (rows, cols, 2) 的数组，其中最后一个维度的长度为 2，通常用于存储复数的实部和虚部。
    # # mask 是一个同样形状为 (rows, cols, 2) 的数组，用于作为频域滤波器的掩码。在高通滤波的情况下，它在频域中心（低频部分）的区域为 0，其余区域为 1。
    # fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    # # 计算频域图像的幅度谱并取对数变换,乘以2000是一个缩放因子，用于调整显示的亮度。
    # f_ishift = np.fft.ifftshift(fshift)
    # img_back = cv2.idft(f_ishift)
    # img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])   #滤波滤掉的部分。
    # # 将 img_back 转换为 uint8 类型，并将范围归一化到 0 到 255
    # img_back_filter = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv2.imwrite((imagesave + "/" + filename.split(".")[0] + "-r" + str(r) + "_HPF.jpg"), img_back_filter)
    # # # Canny 边缘检测
    # # edges = cv2.Canny(img_back_filter, 150, 250, apertureSize=3, L2gradient=False)


