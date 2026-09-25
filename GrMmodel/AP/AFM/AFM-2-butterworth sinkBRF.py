import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# 确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


def generate_periodic_grid_points(x, y, image_shape):
    """
    生成周期性格点，满足 n*x 和 m*y 的条件。

    参数:
        x (int): 水平方向步长
        y (int): 垂直方向步长
        image_shape (tuple): 图像尺寸 (height, width)

    返回:
        grid_points (list): 周期性格点列表，每个点为 (n*x, m*y)
    """
    height, width = image_shape
    half_width = width / 2
    half_height = height / 2

    # 计算 k 和 l
    k = int(np.ceil(half_width / abs(x))) if x != 0 else 0
    l = int(np.ceil(half_height / abs(y))) if y != 0 else 0

    # 生成 n 和 m 的范围
    n_range = range(-k, k + 1)  # 包括负值和正值
    m_range = range(-l, l + 1)  # 包括负值和正值

    # 生成网格点
    grid_points = []
    for n in n_range:
        for m in m_range:
            # 排除原点(0,0)
            if n == 0 and m == 0:
                continue

            point_x = n * x
            point_y = m * y

            # 检查点是否在频谱范围内
            if abs(point_x) < half_width and abs(point_y) < half_height:
                grid_points.append((point_x, point_y))

    return grid_points


def create_periodic_notch_filter(shape, grid_points, radius=10, order=3):
    """
    创建基于周期性格点的巴特沃斯陷波滤波器

    参数:
        shape: 滤波器尺寸 (rows, cols)
        grid_points: 格点坐标列表
        radius: 陷波半径
        order: 巴特沃斯滤波器阶数

    返回:
        陷波滤波器
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    # 创建全通滤波器
    filter = np.ones((rows, cols), np.float32)

    # 对每个格点创建带阻滤波器
    for point in grid_points:
        u, v = point
        # 创建网格坐标
        U, V = np.meshgrid(np.arange(cols), np.arange(rows))

        # 计算到格点的距离
        D = np.sqrt((U - ccol - u) ** 2 + (V - crow - v) ** 2)

        # 创建单个陷波滤波器 (带阻)
        H = 1 / (1 + (radius / (D + 1e-6)) ** (2 * order))  # 避免除以零
        filter = filter * H

    return filter


def visualize_grid_points(shape, grid_points):
    """可视化格点分布"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    # 创建空图像
    vis = np.zeros((rows, cols, 3), dtype=np.uint8)

    # 绘制格点
    for (u, v) in grid_points:
        x = int(ccol + u)
        y = int(crow + v)
        if 0 <= x < cols and 0 <= y < rows:
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)  # 红色点

    # 绘制中心点
    cv2.circle(vis, (ccol, crow), 5, (0, 255, 0), -1)  # 绿色中心点

    # 绘制坐标轴
    cv2.line(vis, (ccol, 0), (ccol, rows - 1), (255, 0, 0), 1)  # 垂直轴
    cv2.line(vis, (0, crow), (cols - 1, crow), (255, 0, 0), 1)  # 水平轴

    return vis


def detect_grid_parameters(magnitude_spectrum, num_peaks=4):
    """
    检测网格参数(水平步长x和垂直步长y)

    参数:
        magnitude_spectrum: 频谱幅度图
        num_peaks: 要检测的峰值数量

    返回:
        x_step, y_step: 网格的水平步长和垂直步长
    """
    # 创建频谱的副本
    mag = magnitude_spectrum.copy()
    rows, cols = mag.shape
    crow, ccol = rows // 2, cols // 2

    # 屏蔽中心低频区域
    mask_size = 30
    mag[crow - mask_size:crow + mask_size, ccol - mask_size:ccol + mask_size] = 0

    # 找到最亮的衍射点
    peaks = []

    for _ in range(num_peaks):
        # 找到最大值位置
        _, max_val, _, max_loc = cv2.minMaxLoc(mag)
        u, v = max_loc

        # 转换为相对于中心的坐标
        u_rel = u - ccol
        v_rel = v - crow

        # 添加到峰值列表
        peaks.append((u_rel, v_rel))

        # 屏蔽已检测区域
        cv2.circle(mag, max_loc, 20, 0, -1)

    # 计算水平步长和垂直步长
    x_values = [abs(p[0]) for p in peaks]
    y_values = [abs(p[1]) for p in peaks]

    # 计算平均步长
    x_step = np.mean(x_values) if x_values else 0
    y_step = np.mean(y_values) if y_values else 0

    return 2*x_step, 2*y_step, peaks


def remove_moire_with_periodic_grid(image_path, radius=15, order=3):
    """
    使用周期性格点状陷波滤波器去除摩尔纹

    参数:
        image_path: 输入图像路径
        radius: 陷波半径
        order: 滤波器阶数
    """
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. 傅里叶变换
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # 计算幅度谱
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. 检测网格参数
    x_step, y_step, base_peaks = detect_grid_parameters(magnitude_spectrum)

    # 如果未检测到有效步长，使用默认值
    if x_step <= 0 or y_step <= 0:
        print("未检测到有效网格参数，使用默认值")
        x_step, y_step = 30, 30

    print(f"检测到的网格参数: x_step={x_step:.2f}, y_step={y_step:.2f}")

    # 4. 生成周期性格点
    grid_points = generate_periodic_grid_points(x_step, y_step, gray.shape)

    # 5. 创建格点状陷波滤波器
    notch_filter = create_periodic_notch_filter(gray.shape, grid_points, radius, order)


    # 6. 应用滤波器
    fshift_filtered = fshift * notch_filter

    # 7. 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back).clip(0, 255).astype(np.uint8)

    # 8. 后处理增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_back)

    # 9. 计算滤波后的频谱
    filtered_magnitude = np.log(np.abs(fshift_filtered) + 1)
    filtered_magnitude = cv2.normalize(filtered_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 10. 可视化格点
    grid_visualization = visualize_grid_points(gray.shape, grid_points)

    return {
        'original': gray,
        'filtered': img_back,   #滤波图像
        'enhanced': enhanced,   #该掩膜增强图像
        'magnitude_spectrum': magnitude_spectrum,
        'filtered_magnitude': filtered_magnitude,
        'notch_filter': notch_filter * 255,    #滤波器可视化
        'grid_points': grid_visualization,
        'base_peaks': base_peaks,
        'x_step': x_step,
        'y_step': y_step
    }

# 使用示例
if __name__ == "__main__":
    image_path = "B:/software-patent/graphenefraction/AFM/AP-10min-15du/try/10min-15du.0_00002-hog.jpg"  # 替换为您的图像路径
    originAFM = cv2.imread("B:/software-patent/graphenefraction/AFM/AP-10min-15du/try/10min-15du.0_00002.jpg")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    if len(originAFM.shape) == 3:  # 3 通道图像（如 RGB）
        originAFM_gray = cv2.cvtColor(originAFM, cv2.COLOR_BGR2GRAY)
    else:
        originAFM_gray = originAFM  # 已经是单通道
    # 对灰度图进行阈值处理，确保 whitespot 是二维
    _, whitespot = cv2.threshold(originAFM_gray, 225, 255, cv2.THRESH_BINARY)


    # cv2.imshow("whitespot", whitespot)
    # cv2.waitKey(0)


    h, w = originAFM.shape[:2]

    thre_fore = 60
    thre_back = 70
    image = cv2.imread(image_path)
    # 应用格点滤波器
    results = remove_moire_with_periodic_grid(image_path, radius=15, order=3)

    # 可视化结果
    plt.figure(figsize=(20, 15))
    border_width = 3

    '''
    #cv2.MORPH_RECT（矩形）
    形状：矩形，所有像素点均为1（有效区域）。
    特点：覆盖范围规则，适合大面积操作，如膨胀或腐蚀时对图像边缘进行均匀扩展或收缩。
    
    cv2.MORPH_ELLIPSE（椭圆形）
    形状：椭圆形，边缘平滑，中心区域为1，向外逐渐过渡为0。
    特点：覆盖范围介于矩形和十字形之间，适合平滑处理或保留曲线结构。
        
    cv2.MORPH_CROSS（十字形）
    形状：十字形，仅中心行和中心列的像素为1，其余为0。
    特点：覆盖范围较小，适合精细操作，如连接相邻像素或处理细长结构。
    
    cv2.MORPH_RECT	矩形	大面积均匀处理（如去除直角）	腐蚀/膨胀
    cv2.MORPH_CROSS	十字形	精细连接或细长结构处理	开运算去除小物体
    cv2.MORPH_ELLIPSE	椭圆形	平滑边缘、去除尖角或圆弧	闭运算平滑轮廓
    '''

    ori = results['original']
    cv2.imwrite(image_path.split(".")[0] + "HOG.png", ori)
    _, ori = cv2.threshold(ori, thre_fore, 255, 1)  # 取阈值低于thre的作为背景

    ori = cv2.erode(ori, kernel1, iterations=1)

    ori = cv2.dilate(ori, kernel, iterations=1)
    ori = cv2.erode(ori, kernel1, iterations=1)

    enhanced = results['enhanced']
    _, enhanced = cv2.threshold(enhanced, thre_fore, 255, 1)  # 取阈值低于thre的作为背景
    # enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
    enhanced = cv2.dilate(enhanced, kernel, iterations=1)


    filter = results['filtered']
    cv2.imwrite(image_path.split(".")[0]+"HOGsinkBPF.png", filter)

    #对背景进行阈值处理及形态学操作
    _, filter = cv2.threshold(filter, thre_back, 255, cv2.THRESH_BINARY)  # 取阈值低于thre的作为背景
    filter_0 = filter
    filter = cv2.dilate(filter, kernel, iterations = 1)

    filter_1 = filter
    filter = cv2.erode(filter, kernel, iterations = 1)
    filter_2 = filter
    filter = cv2.dilate(filter, kernel3, iterations = 1)
    filter_3 = filter
    filter = cv2.erode(filter,kernel, iterations=3)
    filter_4 = filter

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filter, connectivity=4)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <  10000:
            print("背景连通区域面积：", area)
            filter[labels ==label] = 0


    filter_inverse = 255 - filter
    cv2.rectangle(filter_inverse, (0,0), (w, border_width), (0,0,0), -1)    # 矩形左上角坐标 (0,0), 矩形右下角坐标(w, border_width)
    cv2.rectangle(filter_inverse, (0,h-border_width), (w,h), (0,0,0), -1)
    cv2.rectangle(filter_inverse, (0,0), (border_width, h), (0,0,0), -1)
    cv2.rectangle(filter_inverse, (w-border_width,0), (w, h), (0,0,0), -1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filter_inverse, connectivity=4)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <  10000:
            print("背景反转结果连通区域面积：", area)
            filter[labels ==label] = 255    #将背景图像中黑色小连通区域填充为白色。

    # #用于限制前景的背景区域要大一些比较安全，防止前景超过真正界线
    filter_foreinverse = cv2.dilate(filter,kernel1, iterations = 2)

    filter = cv2.erode(filter, kernel, iterations=5)
    filter = cv2.dilate(filter, kernel3, iterations=1)
    filter = cv2.erode(filter, kernel1, iterations=5)
    filter  = cv2.morphologyEx(filter, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filter, connectivity=4)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <  10000:
            print("背景连通区域面积：", area)
            filter[labels ==label] = 0
    filter_inverse = 255 - filter
    cv2.rectangle(filter_inverse, (0,0), (w, border_width), (0,0,0), -1)    # 矩形左上角坐标 (0,0), 矩形右下角坐标(w, border_width)
    cv2.rectangle(filter_inverse, (0,h-border_width), (w,h), (0,0,0), -1)
    cv2.rectangle(filter_inverse, (0,0), (border_width, h), (0,0,0), -1)
    cv2.rectangle(filter_inverse, (w-border_width,0), (w, h), (0,0,0), -1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filter_inverse, connectivity=4)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <  10000:
            print("背景反转结果连通区域面积：", area)
            filter[labels ==label] = 255    #将背景图像中黑色小连通区域填充为白色。

    filter_copy = filter.copy()       #背景要缩小一些比较安全，防止背景超过真正界线

    filter_copy = cv2.erode(filter_copy, kernel2, iterations = 8)
    # filter_copy = cv2.erode(filter_copy, kernel1, iterations = 2)
    filter_copy = cv2.dilate(filter_copy, kernel, iterations = 1)
    filter_copy = cv2.morphologyEx(filter_copy, cv2.MORPH_OPEN, kernel1, iterations =2)
    # filter_copy = cv2.dilate(filter_copy, kernel1, iterations = 1)



    border = 2
    h, w = filter_copy.shape[:2]
    # 上边：从第0行到第border-1行
    filter_copy[:border, :] = 0
    # 下边：从第(h - border)行到最后一行
    filter_copy[h - border:, :] = 0
    # 左边：从第0列到第border-1列
    filter_copy[:, :border] = 0
    # 右边：从第(w - border)列到最后一列
    filter_copy[:, w - border:] = 0

    contours_back, hierarchy = cv2.findContours(filter_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    output = originAFM.copy()


    cv2.drawContours(image = output,            # 目标图像（在此图上绘制）
                     contours = contours_back,  # 轮廓列表（从findContours获得）
                     contourIdx=-1,             # -1表示绘制所有轮廓
                     color = (0, 255, 0),       # 颜色（BGR格式，绿色）
                     thickness=2,               # 线宽
                     lineType = cv2.LINE_AA     # 抗锯齿线型
                     )

    # 上边：从第0行到第border-1行
    filter_copy [:border, :] = 0
    # 下边：从第(h - border)行到最后一行
    filter_copy [h - border:, :] = 0
    # 左边：从第0列到第border-1列
    filter_copy [:, :border] = 0
    # 右边：从第(w - border)列到最后一列
    filter_copy [:, w - border:] = 0

    whitespot_erode = cv2.morphologyEx(whitespot, cv2.MORPH_ERODE, kernel2, iterations=1)
    filter_copy[whitespot_erode == 255] = 255
    cv2.imwrite(image_path.split(".")[0]+"back.png", filter_copy)



    filter2_copy = 255-filter_foreinverse
    filter2_copy = cv2.morphologyEx(filter2_copy, cv2.MORPH_OPEN, kernel, iterations = 3)
    num_labels, labels, stats, centroids  = cv2.connectedComponentsWithStats(filter2_copy, connectivity=4)

    for label in range(1, num_labels):
        area = stats[label, 4]
        print("前景连通区域面积:", area)
        if area < 20000:   #200000000
            filter2_copy[labels ==label] = 0


    filter_fore_inverse = 255 - filter2_copy

    # 上边：从第0行到第border-1行
    filter_fore_inverse[:border, :] = 0
    # 下边：从第(h - border)行到最后一行
    filter_fore_inverse[h - border:, :] = 0
    # 左边：从第0列到第border-1列
    filter_fore_inverse[:, :border] = 0
    # 右边：从第(w - border)列到最后一列
    filter_fore_inverse[:, w - border:] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filter_fore_inverse, connectivity=4)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <  20000:
            print("背景反转结果连通区域面积：", area)
            filter2_copy[labels ==label] = 255    #将背景图像中黑色小连通区域填充为白色。

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    filter2_copy = cv2.erode(filter2_copy, kernel2, iterations=5)
    filter2_copy = cv2.dilate(filter2_copy, kernel, iterations=1)
    filter2_copy = cv2.morphologyEx(filter2_copy, cv2.MORPH_OPEN, kernel3, iterations = 3)
    
    # 上边：从第0行到第border-1行
    filter2_copy [:border, :] = 0
    # 下边：从第(h - border)行到最后一行
    filter2_copy [h - border:, :] = 0
    # 左边：从第0列到第border-1列
    filter2_copy [:, :border] = 0
    # 右边：从第(w - border)列到最后一列
    filter2_copy [:, w - border:] = 0

    print("filter2_copy.shape:", filter2_copy.shape)
    print("whitespot.shape:", whitespot.shape)
    whitespot_dilate= cv2.morphologyEx(whitespot, cv2.MORPH_DILATE, kernel2, iterations=15)
    filter2_copy[whitespot_dilate == 255] = 0
    cv2.imwrite(image_path.split(".")[0]+"fore.png", filter2_copy)

    contours_fore, _ = cv2.findContours(filter2_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image = output,            # 目标图像（在此图上绘制）
                     contours = contours_fore,  # 轮廓列表（从findContours获得）
                     contourIdx=-1,             # -1表示绘制所有轮廓
                     color = (0, 0, 255),       # 颜色（BGR格式）
                     thickness=2,               # 线宽
                     lineType = cv2.LINE_AA     # 抗锯齿线型
                     )
    cv2.imwrite(image_path.split(".")[0]+"contour.png", output)
    print("前背景轮廓图像保存在：", image_path.split(".")[0] + "contour.png")

    cv2.imwrite(image_path.split(".")[0] + "sinkBPFspectrum.png", results['filtered_magnitude'])
    cv2.imwrite(image_path.split(".")[0] + "spectrum.png", results['magnitude_spectrum'])

    # 标记检测到的基础峰值
    mag_vis = cv2.cvtColor(results['magnitude_spectrum'], cv2.COLOR_GRAY2BGR)
    rows, cols = results['original'].shape
    crow, ccol = rows // 2, cols // 2
    for (u, v) in results['base_peaks']:
        x = int(ccol + u)
        y = int(crow + v)
        cv2.circle(mag_vis, (x, y), 5, (0, 0, 255), -1)

    # 保存结果
    cv2.imwrite(image_path.split(".")[0]+"enhanced_image.jpg", results['enhanced'])
