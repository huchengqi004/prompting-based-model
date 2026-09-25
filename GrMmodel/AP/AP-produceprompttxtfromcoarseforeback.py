import os
import cv2
import numpy as np
from tkinter import filedialog
import math

# ---------------------- 用户可调参数 ----------------------
Areafilter_fore = 2500  # 前景连通区域最小面积阈值（用于生成点提示）
Areafilter_back = 2500  # 背景连通区域最小面积阈值（用于生成点提示）
gridspace = 50  # 背景格点采样间距
foreareaforbox = 1000  # 前景连通区域面积大于该值时生成框提示
boxprompt = 1  # 1：生成框提示；0：不生成框提示
prefix_length = 2  # 文件名前缀匹配位数（例如前5个字符相同视为同一组）
output_subfolder = "prompt_results"  # 输出子文件夹名称（创建于原图文件夹下）
distance_centroid_distpoint = 20  # 已废弃，保留以兼容旧参数
foreregionarea = 2500  # 黑色连通区域面积阈值，大于该值时在其外接矩形内密集采样
ratio = 1  # 密集采样倍数，实际间距为 gridspace / ratio
min_background_point_distance = 20  # 提示点之间的最小空间距离阈值（前景和背景均使用）
n_division = 10  # 外接矩形等分数（横向和纵向各 n 等分）
backbox = 0  # 1：绘制小矩形；0：不绘制小矩形
# ---------------------------------------------------------


def select_directory(title):
    """使用文件夹选择对话框，返回路径"""
    folder = filedialog.askdirectory(title=title)
    if not folder:
        raise ValueError("未选择文件夹！")
    return folder


def get_image_files(folder):
    """获取文件夹下所有常见图像文件路径，返回列表"""
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith(extensions):
            files.append(os.path.join(folder, f))
    return sorted(files)


def read_binary_mask(path):
    """读取图像并转换为二值掩码（0/255）"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


def read_image_bgr(path):
    """读取彩色图像（BGR格式）"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    return img


def compute_distance_transform(binary_mask):
    """计算二值掩码的距离变换图（浮点型）"""
    return cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)


def get_connected_components(binary_mask):
    """获取连通组件信息"""
    return cv2.connectedComponentsWithStats(binary_mask, connectivity=8)


def is_white(mask, point):
    """检查点是否在掩码中为白色"""
    x, y = point
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        return mask[y, x] == 255
    return False


def get_max_distance_point(dist_transform, labels, label):
    """获取指定连通区域内的距离变换最大值点"""
    region_mask = np.where(labels == label, 255, 0).astype(np.uint8)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform, mask=region_mask)
    return max_loc  # (x, y)


def get_grid_points_in_region(labels, label, gridspace):
    """获取指定连通区域内的格点（标签等于 label 的点）"""
    points = []
    x, y, w, h = cv2.boundingRect(np.where(labels == label, 255, 0).astype(np.uint8))
    for i in range(x, x + w, gridspace):
        for j in range(y, y + h, gridspace):
            if labels[j, i] == label:
                points.append((i, j))
    return points


def get_grid_points_centered(labels, label, center, gridspace):
    """
    以 center 为中心，在指定连通区域内进行均匀格点采样。
    参数:
        labels: 连通区域标签图
        label: 当前连通区域的标签
        center: (cx, cy)，采样中心（通常是距离变换最大值点）
        gridspace: 格点间距
    返回:
        points: list of (x, y)
    """
    points = []
    cx, cy = center
    h, w = labels.shape

    # 获取该连通区域的边界框
    ys, xs = np.where(labels == label)
    if len(xs) == 0:
        return points
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # 计算覆盖整个边界框所需的步数范围
    i_min = int(np.floor((x_min - cx) / gridspace))
    i_max = int(np.ceil((x_max - cx) / gridspace))
    j_min = int(np.floor((y_min - cy) / gridspace))
    j_max = int(np.ceil((y_max - cy) / gridspace))

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            x = cx + i * gridspace
            y = cy + j * gridspace
            if 0 <= x < w and 0 <= y < h:
                if labels[y, x] == label:
                    points.append((x, y))
    return points


def filter_foreground_points(points, labels, fore_mask, min_distance):
    """
    筛选前景提示点：
    - 优先保留距离变换值更大的点（更靠近区域中心）
    - 点之间空间距离不低于 min_distance

    参数：
        points: list of (x, y)
        labels: list of int
        fore_mask: np.ndarray - 前景掩码（二值图，白色为前景）
        min_distance: int - 前景点之间的最小空间距离阈值

    返回：
        filtered_points, filtered_labels
    """
    if not points:
        return points, labels

    dist_transform = cv2.distanceTransform(fore_mask, cv2.DIST_L2, 5)

    # 分离前景点和背景点
    fg_points = [(p, l) for p, l in zip(points, labels) if l == 1]
    bg_points = [(p, l) for p, l in zip(points, labels) if l == 0]

    if not fg_points:
        return points, labels

    # 计算每个前景点的距离变换值
    fg_with_dist = []
    for pt, lb in fg_points:
        x, y = pt
        if 0 <= x < fore_mask.shape[1] and 0 <= y < fore_mask.shape[0]:
            dist_val = dist_transform[y, x]
        else:
            dist_val = float('-inf')
        fg_with_dist.append((pt, lb, dist_val))

    # 按距离变换值从大到小排序（优先保留靠近区域中心的点）
    fg_with_dist.sort(key=lambda item: item[2], reverse=True)

    # 贪心筛选
    selected_fg = []
    for pt, lb, dist_val in fg_with_dist:
        too_close = False
        for sel_pt, sel_lb, sel_dist in selected_fg:
            d = math.hypot(pt[0] - sel_pt[0], pt[1] - sel_pt[1])
            if d < min_distance:
                too_close = True
                break
        if not too_close:
            selected_fg.append((pt, lb, dist_val))

    # 组合筛选后的前景点 + 原始背景点
    filtered_points = [p for p, l in bg_points]
    filtered_labels = [l for p, l in bg_points]

    for pt, lb, dist_val in selected_fg:
        filtered_points.append(pt)
        filtered_labels.append(lb)

    return filtered_points, filtered_labels


def filter_background_points(points, labels, back_mask, min_distance):
    """
    筛选背景提示点：
    - 优先保留距离变换值更大的点（更靠近区域中心、远离边缘）
    - 点之间空间距离不低于 min_distance

    参数：
        points: list of (x, y) - 所有提示点坐标
        labels: list of int - 对应的标签
        back_mask: np.ndarray - 背景掩码（二值图，白色为背景）
        min_distance: int - 点之间的最小空间距离阈值

    返回：
        filtered_points: list of (x, y)
        filtered_labels: list of int
    """
    if not points:
        return points, labels

    # 计算背景掩码的距离变换图
    dist_transform = cv2.distanceTransform(back_mask, cv2.DIST_L2, 5)

    # 分离前景点和背景点
    fg_points = [(p, l) for p, l in zip(points, labels) if l == 1]
    bg_points = [(p, l) for p, l in zip(points, labels) if l == 0]

    if not bg_points:
        # 无背景点，直接返回原列表
        return points, labels

    # 计算每个背景点的距离变换值（越大越靠近区域中心）
    bg_with_dist = []
    for pt, lb in bg_points:
        x, y = pt
        if 0 <= x < back_mask.shape[1] and 0 <= y < back_mask.shape[0]:
            dist_val = dist_transform[y, x]
        else:
            dist_val = float('-inf')  # 越界点，给最低优先级
        bg_with_dist.append((pt, lb, dist_val))

    # 按距离变换值从大到小排序（优先保留靠近区域中心的点）
    bg_with_dist.sort(key=lambda item: item[2], reverse=True)

    # 贪心筛选
    selected_bg = []
    for pt, lb, dist_val in bg_with_dist:
        # 检查与已选点的距离
        too_close = False
        for sel_pt, sel_lb, sel_dist in selected_bg:
            d = math.hypot(pt[0] - sel_pt[0], pt[1] - sel_pt[1])
            if d < min_distance:
                too_close = True
                break
        if not too_close:
            selected_bg.append((pt, lb, dist_val))

    # 组合前景点和筛选后的背景点
    filtered_points = [p for p, l in fg_points]
    filtered_labels = [l for p, l in fg_points]

    for pt, lb, dist_val in selected_bg:
        filtered_points.append(pt)
        filtered_labels.append(lb)

    return filtered_points, filtered_labels


def generate_prompts(fore_mask, back_mask, areafilter_fore, areafilter_back, gridspace,
                     foreareaforbox, boxprompt_flag, min_background_point_distance,
                     n_division, backbox_flag):
    """
    生成提示点、标签和提示框
    返回：
        points: list of (x, y)
        labels: list of int (1 前景，0 背景)
        boxes: list of [x1, y1, x2, y2]
        fore_contours: list of contours
        back_contours: list of contours
        small_rectangles: list of [x1, y1, x2, y2] (用于可视化的小矩形)
    """
    points = []
    labels = []
    boxes = []
    small_rectangles = []

    # ---------- 前景处理 ----------
    num_fore, labels_fore, stats_fore, centroids_fore = get_connected_components(fore_mask)
    dist_fore = compute_distance_transform(fore_mask)
    fore_contours, _ = cv2.findContours(fore_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for label_id in range(1, num_fore):
        area = stats_fore[label_id, cv2.CC_STAT_AREA]
        if area < areafilter_fore and area < foreareaforbox:
            continue

        # 生成框提示（如果启用且面积大于 foreareaforbox）
        if boxprompt_flag == 1 and area > foreareaforbox:
            x, y, w, h = stats_fore[label_id, cv2.CC_STAT_LEFT], \
                stats_fore[label_id, cv2.CC_STAT_TOP], \
                stats_fore[label_id, cv2.CC_STAT_WIDTH], \
                stats_fore[label_id, cv2.CC_STAT_HEIGHT]
            box = [x, y, x + w, y + h]
            boxes.append(box)

        # 生成点提示：以距离变换最大值点为中心的均匀格点采样
        if area >= areafilter_fore:
            max_dist_point = get_max_distance_point(dist_fore, labels_fore, label_id)
            grid_pts = get_grid_points_centered(labels_fore, label_id, max_dist_point, gridspace)
            for pt in grid_pts:
                points.append(pt)
                labels.append(1)

    # ---------- 背景处理 ----------
    num_back, labels_back, stats_back, centroids_back = get_connected_components(back_mask)
    dist_back = compute_distance_transform(back_mask)
    back_contours, _ = cv2.findContours(back_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for label_id in range(1, num_back):
        area = stats_back[label_id, cv2.CC_STAT_AREA]
        if area < areafilter_back:
            continue

        # 原始格点采样
        grid_pts = get_grid_points_in_region(labels_back, label_id, gridspace)
        for pt in grid_pts:
            points.append(pt)
            labels.append(0)

    # ---------- 额外：黑色连通区域密集采样 ----------
    # 黑色区域是背景掩码中值为0的连通区域
    black_mask = 255 - back_mask
    _, black_labels, black_stats, _ = get_connected_components(black_mask)
    dense_gridspace = max(1, int(gridspace / ratio))
    for black_label in range(1, black_labels.max() + 1):
        black_area = black_stats[black_label, cv2.CC_STAT_AREA]
        if black_area >= foreregionarea:
            # 获取该黑色连通区域的最小外接矩形
            x, y, w, h = black_stats[black_label, cv2.CC_STAT_LEFT], \
                black_stats[black_label, cv2.CC_STAT_TOP], \
                black_stats[black_label, cv2.CC_STAT_WIDTH], \
                black_stats[black_label, cv2.CC_STAT_HEIGHT]

            # 在矩形内密集采样
            for i in range(x, x + w, dense_gridspace):
                for j in range(y, y + h, dense_gridspace):
                    # 仅当该点在背景掩码中为白色（255）时才添加
                    if back_mask[j, i] == 255:
                        pt = (i, j)
                        points.append(pt)
                        labels.append(0)

            # ---------- 外接矩形等分采样 ----------
            if n_division > 1:
                # 计算每个小矩形的宽度和高度
                small_w = w / n_division
                small_h = h / n_division

                # 遍历每个小矩形
                for row in range(n_division):
                    for col in range(n_division):
                        # 计算小矩形边界
                        x1 = int(x + col * small_w)
                        y1 = int(y + row * small_h)
                        x2 = int(x + (col + 1) * small_w)
                        y2 = int(y + (row + 1) * small_h)

                        # 确保边界不超出图像范围
                        x1 = max(0, min(x1, back_mask.shape[1] - 1))
                        y1 = max(0, min(y1, back_mask.shape[0] - 1))
                        x2 = max(0, min(x2, back_mask.shape[1]))
                        y2 = max(0, min(y2, back_mask.shape[0]))

                        # 保存小矩形用于可视化
                        if backbox_flag == 1:
                            small_rectangles.append([x1, y1, x2, y2])

                        # 提取小矩形区域
                        roi = back_mask[y1:y2, x1:x2]

                        # 查找小矩形内的白色像素连通区域
                        roi_white = roi.copy()
                        roi_white[roi_white != 255] = 0

                        # 使用连通组件分析找到白色区域的连通组件
                        num_roi_labels, roi_labels, roi_stats, roi_centroids = get_connected_components(roi_white)

                        # 对每个连通区域，取距离变换最大值点
                        roi_dist = compute_distance_transform(roi_white)
                        for roi_label in range(1, num_roi_labels):
                            roi_area = roi_stats[roi_label, cv2.CC_STAT_AREA]
                            if roi_area < 10:  # 忽略过小的区域
                                continue

                            # 获取该区域的距离变换最大值点（在小矩形内的局部坐标）
                            roi_region_mask = np.where(roi_labels == roi_label, 255, 0).astype(np.uint8)
                            _, _, _, roi_max_loc = cv2.minMaxLoc(roi_dist, mask=roi_region_mask)

                            # 转换为全局坐标
                            global_x = x1 + roi_max_loc[0]
                            global_y = y1 + roi_max_loc[1]

                            # 确保点在背景掩码中为白色
                            if is_white(back_mask, (global_x, global_y)):
                                points.append((global_x, global_y))
                                labels.append(0)

    # ---------- 去重 ----------
    unique_points = []
    unique_labels = []
    seen = set()
    for pt, lb in zip(points, labels):
        key = (pt[0], pt[1])
        if key not in seen:
            seen.add(key)
            unique_points.append(pt)
            unique_labels.append(lb)

    # ---------- 筛选前景提示点 ----------
    unique_points, unique_labels = filter_foreground_points(
        unique_points, unique_labels, fore_mask, min_background_point_distance
    )

    # ---------- 筛选背景提示点 ----------
    unique_points, unique_labels = filter_background_points(
        unique_points, unique_labels, back_mask, min_background_point_distance
    )

    return unique_points, unique_labels, boxes, fore_contours, back_contours, small_rectangles


def save_prompts_to_txt(points, labels, boxes, output_path):
    """将提示点、标签和框保存到 txt 文件"""
    with open(output_path, 'w') as f:
        f.write("point prompts: {}\n".format(points))
        f.write("labels: {}\n".format(labels))
        f.write("box prompts: {}\n".format(boxes))


def visualize_overlay(image, points, labels, boxes, fore_contours, back_contours,
                      small_rectangles, output_path, backbox_flag):
    """
    在原图上叠加轮廓、提示点、提示框和小矩形并保存
    """
    overlay = image.copy()

    # 绘制背景轮廓（绿色）
    cv2.drawContours(overlay, back_contours, -1, (0, 255, 0), 1)
    # 绘制前景轮廓（红色）
    cv2.drawContours(overlay, fore_contours, -1, (0, 0, 255), 1)

    # 绘制提示点
    for pt, lb in zip(points, labels):
        color = (0, 0, 255) if lb == 1 else (0, 255, 0)  # 前景红，背景绿
        cv2.circle(overlay, pt, 4, color, -1)

    # 绘制提示框（黄色）
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # 绘制小矩形（蓝色）
    if backbox_flag == 1:
        for rect in small_rectangles:
            x1, y1, x2, y2 = rect
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)

    cv2.imwrite(output_path, overlay)


def main():
    # 选择三个文件夹
    print("请选择原始图像文件夹：")
    image_folder = select_directory("选择原图文件夹")
    print("请选择前景掩码文件夹：")
    fore_folder = select_directory("选择前景掩码文件夹")
    print("请选择背景掩码文件夹：")
    back_folder = select_directory("选择背景掩码文件夹")

    # 获取所有图像文件
    image_files = get_image_files(image_folder)
    fore_files = get_image_files(fore_folder)
    back_files = get_image_files(back_folder)

    if not image_files:
        print("原图文件夹为空，程序退出。")
        return

    # 创建输出子文件夹
    output_dir = os.path.join(image_folder, output_subfolder)
    os.makedirs(output_dir, exist_ok=True)

    # 建立前缀索引字典
    def prefix_key(filepath):
        name = os.path.splitext(os.path.basename(filepath))[0]
        return name[:prefix_length] if len(name) >= prefix_length else name

    fore_dict = {}
    for fp in fore_files:
        key = prefix_key(fp)
        fore_dict.setdefault(key, []).append(fp)

    back_dict = {}
    for fp in back_files:
        key = prefix_key(fp)
        back_dict.setdefault(key, []).append(fp)

    processed = 0
    for img_path in image_files:
        key = prefix_key(img_path)
        if key not in fore_dict or key not in back_dict:
            print(f"警告：前缀 '{key}' 没有对应的前景或背景掩码，跳过 {os.path.basename(img_path)}")
            continue

        # 取第一个匹配的文件（假设每个前缀唯一）
        fore_path = fore_dict[key][0]
        back_path = back_dict[key][0]

        print(f"处理：{os.path.basename(img_path)} (前缀: {key})")

        # 读取图像
        image = read_image_bgr(img_path)
        fore_mask = read_binary_mask(fore_path)
        back_mask = read_binary_mask(back_path)

        # 检查尺寸一致性，若不匹配则调整掩码到原图尺寸
        if image.shape[:2] != fore_mask.shape or image.shape[:2] != back_mask.shape:
            fore_mask = cv2.resize(fore_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            back_mask = cv2.resize(back_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 生成提示
        points, labels, boxes, fore_contours, back_contours, small_rectangles = generate_prompts(
            fore_mask, back_mask,
            Areafilter_fore, Areafilter_back, gridspace,
            foreareaforbox, boxprompt, min_background_point_distance,
            n_division, backbox
        )

        # 输出统计信息
        num_fg = sum(1 for lb in labels if lb == 1)
        num_bg = sum(1 for lb in labels if lb == 0)
        print(f"  前景提示点: {num_fg}, 背景提示点: {num_bg}, 提示框: {len(boxes)}, 小矩形: {len(small_rectangles)}")

        # 保存 txt
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_output = os.path.join(output_dir, f"{base_name}_prompts.txt")
        save_prompts_to_txt(points, labels, boxes, txt_output)

        # 保存叠加图像
        overlay_output = os.path.join(output_dir, f"{base_name}_imageoverlap.png")
        visualize_overlay(image, points, labels, boxes, fore_contours, back_contours,
                          small_rectangles, overlay_output, backbox)

        processed += 1

    print(f"批量处理完成，共处理 {processed} 组图像。结果保存在：{output_dir}")


if __name__ == "__main__":
    main()






##前景无格点
# import os
# import cv2
# import numpy as np
# from tkinter import filedialog
# import math
#
# # ---------------------- 用户可调参数 ----------------------
# Areafilter_fore = 2500  # 前景连通区域最小面积阈值（用于生成点提示）
# Areafilter_back = 2500  # 背景连通区域最小面积阈值（用于生成点提示）
# gridspace = 100  # 背景格点采样间距
# foreareaforbox = 1000  # 前景连通区域面积大于该值时生成框提示
# boxprompt = 1  # 1：生成框提示；0：不生成框提示
# prefix_length = 2  # !!!!!文件名前缀匹配位数（例如前5个字符相同视为同一组）
# output_subfolder = "prompt_results"  # 输出子文件夹名称（创建于原图文件夹下）
# distance_centroid_distpoint = 20  # 质心与距离变换最大值点的最小距离阈值（像素）
# foreregionarea = 2500  # 黑色连通区域面积阈值，大于该值时在其外接矩形内密集采样
# ratio = 1  # 密集采样倍数，实际间距为 gridspace / ratio
# min_background_point_distance = 20  # 背景提示点之间的最小空间距离阈值
# n_division = 10  # 外接矩形等分数（横向和纵向各 n 等分）
# backbox = 0  # 1：绘制小矩形；0：不绘制小矩形
#
#
# # ---------------------------------------------------------
#
# def select_directory(title):
#     """使用文件夹选择对话框，返回路径"""
#     folder = filedialog.askdirectory(title=title)
#     if not folder:
#         raise ValueError("未选择文件夹！")
#     return folder
#
#
# def get_image_files(folder):
#     """获取文件夹下所有常见图像文件路径，返回列表"""
#     extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
#     files = []
#     for f in os.listdir(folder):
#         if f.lower().endswith(extensions):
#             files.append(os.path.join(folder, f))
#     return sorted(files)
#
#
# def read_binary_mask(path):
#     """读取图像并转换为二值掩码（0/255）"""
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(f"无法读取图像：{path}")
#     _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#     return binary
#
#
# def read_image_bgr(path):
#     """读取彩色图像（BGR格式）"""
#     img = cv2.imread(path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise FileNotFoundError(f"无法读取图像：{path}")
#     return img
#
#
# def compute_distance_transform(binary_mask):
#     """计算二值掩码的距离变换图（浮点型）"""
#     return cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
#
#
# def get_connected_components(binary_mask):
#     """获取连通组件信息"""
#     return cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
#
#
# def is_white(mask, point):
#     """检查点是否在掩码中为白色"""
#     x, y = point
#     if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
#         return mask[y, x] == 255
#     return False
#
#
# def get_max_distance_point(dist_transform, labels, label):
#     """获取指定连通区域内的距离变换最大值点"""
#     region_mask = np.where(labels == label, 255, 0).astype(np.uint8)
#     _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform, mask=region_mask)
#     return max_loc  # (x, y)
#
#
# def get_grid_points_in_region(labels, label, gridspace):
#     """获取指定连通区域内的格点（标签等于 label 的点）"""
#     points = []
#     x, y, w, h = cv2.boundingRect(np.where(labels == label, 255, 0).astype(np.uint8))
#     for i in range(x, x + w, gridspace):
#         for j in range(y, y + h, gridspace):
#             if labels[j, i] == label:
#                 points.append((i, j))
#     return points
# def filter_background_points(points, labels, back_mask, min_distance):
#     """
#     筛选背景提示点：
#     - 优先保留距离变换值更大的点（更靠近区域中心、远离边缘）
#     - 点之间空间距离不低于 min_distance
#
#     参数：
#         points: list of (x, y) - 所有提示点坐标
#         labels: list of int - 对应的标签
#         back_mask: np.ndarray - 背景掩码（二值图，白色为背景）
#         min_distance: int - 点之间的最小空间距离阈值
#
#     返回：
#         filtered_points: list of (x, y)
#         filtered_labels: list of int
#     """
#     if not points:
#         return points, labels
#
#     # 计算背景掩码的距离变换图
#     dist_transform = cv2.distanceTransform(back_mask, cv2.DIST_L2, 5)
#
#     # 分离前景点和背景点
#     fg_points = [(p, l) for p, l in zip(points, labels) if l == 1]
#     bg_points = [(p, l) for p, l in zip(points, labels) if l == 0]
#
#     if not bg_points:
#         # 无背景点，直接返回原列表
#         return points, labels
#
#     # 计算每个背景点的距离变换值（越大越靠近区域中心）
#     bg_with_dist = []
#     for pt, lb in bg_points:
#         x, y = pt
#         if 0 <= x < back_mask.shape[1] and 0 <= y < back_mask.shape[0]:
#             dist_val = dist_transform[y, x]
#         else:
#             dist_val = float('-inf')  # 越界点，给最低优先级
#         bg_with_dist.append((pt, lb, dist_val))
#
#     # 按距离变换值从大到小排序（优先保留靠近区域中心的点）
#     bg_with_dist.sort(key=lambda item: item[2], reverse=True)
#
#     # 贪心筛选
#     selected_bg = []
#     for pt, lb, dist_val in bg_with_dist:
#         # 检查与已选点的距离
#         too_close = False
#         for sel_pt, sel_lb, sel_dist in selected_bg:
#             d = math.hypot(pt[0] - sel_pt[0], pt[1] - sel_pt[1])
#             if d < min_distance:
#                 too_close = True
#                 break
#         if not too_close:
#             selected_bg.append((pt, lb, dist_val))
#
#     # 组合前景点和筛选后的背景点
#     filtered_points = [p for p, l in fg_points]
#     filtered_labels = [l for p, l in fg_points]
#
#     for pt, lb, dist_val in selected_bg:
#         filtered_points.append(pt)
#         filtered_labels.append(lb)
#
#     return filtered_points, filtered_labels
#
# # def filter_background_points(points, labels, back_mask, min_distance):
# #     """
# #     筛选背景提示点：
# #     - 优先保留距离变换值更小的点（更靠近边缘）
# #     - 点之间空间距离不低于 min_distance
# #
# #     参数：
# #         points: list of (x, y) - 所有提示点坐标
# #         labels: list of int - 对应的标签
# #         back_mask: np.ndarray - 背景掩码（二值图，白色为背景）
# #         min_distance: int - 点之间的最小空间距离阈值
# #
# #     返回：
# #         filtered_points: list of (x, y)
# #         filtered_labels: list of int
# #     """
# #     if not points:
# #         return points, labels
# #
# #     # 计算背景掩码的距离变换图
# #     dist_transform = cv2.distanceTransform(back_mask, cv2.DIST_L2, 5)
# #
# #     # 分离前景点和背景点
# #     fg_points = [(p, l) for p, l in zip(points, labels) if l == 1]
# #     bg_points = [(p, l) for p, l in zip(points, labels) if l == 0]
# #
# #     if not bg_points:
# #         # 无背景点，直接返回原列表
# #         return points, labels
# #
# #     # 计算每个背景点的距离变换值（越小越靠近边缘）
# #     bg_with_dist = []
# #     for pt, lb in bg_points:
# #         x, y = pt
# #         if 0 <= x < back_mask.shape[1] and 0 <= y < back_mask.shape[0]:
# #             dist_val = dist_transform[y, x]
# #         else:
# #             dist_val = float('inf')  # 越界点，给最低优先级
# #         bg_with_dist.append((pt, lb, dist_val))
# #
# #     # 按距离变换值从小到大排序（优先保留靠近边缘的点）
# #     bg_with_dist.sort(key=lambda item: item[2])
# #
# #     # 贪心筛选
# #     selected_bg = []
# #     for pt, lb, dist_val in bg_with_dist:
# #         # 检查与已选点的距离
# #         too_close = False
# #         for sel_pt, sel_lb, sel_dist in selected_bg:
# #             d = math.hypot(pt[0] - sel_pt[0], pt[1] - sel_pt[1])
# #             if d < min_distance:
# #                 too_close = True
# #                 break
# #         if not too_close:
# #             selected_bg.append((pt, lb, dist_val))
# #
# #     # 组合前景点和筛选后的背景点
# #     filtered_points = [p for p, l in fg_points]
# #     filtered_labels = [l for p, l in fg_points]
# #
# #     for pt, lb, dist_val in selected_bg:
# #         filtered_points.append(pt)
# #         filtered_labels.append(lb)
# #
# #     return filtered_points, filtered_labels
#
#
# def generate_prompts(fore_mask, back_mask, areafilter_fore, areafilter_back, gridspace,
#                      foreareaforbox, boxprompt_flag, min_background_point_distance,
#                      n_division, backbox_flag):
#     """
#     生成提示点、标签和提示框
#     返回：
#         points: list of (x, y)
#         labels: list of int (1 前景，0 背景)
#         boxes: list of [x1, y1, x2, y2]
#         fore_contours: list of contours
#         back_contours: list of contours
#         small_rectangles: list of [x1, y1, x2, y2] (用于可视化的小矩形)
#     """
#     points = []
#     labels = []
#     boxes = []
#     small_rectangles = []
#
#     # ---------- 前景处理 ----------
#     num_fore, labels_fore, stats_fore, centroids_fore = get_connected_components(fore_mask)
#     dist_fore = compute_distance_transform(fore_mask)
#     fore_contours, _ = cv2.findContours(fore_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#     for label_id in range(1, num_fore):
#         area = stats_fore[label_id, cv2.CC_STAT_AREA]
#         if area < areafilter_fore and area < foreareaforbox:
#             continue
#
#         # 生成框提示（如果启用且面积大于 foreareaforbox）
#         if boxprompt_flag == 1 and area > foreareaforbox:
#             x, y, w, h = stats_fore[label_id, cv2.CC_STAT_LEFT], \
#                 stats_fore[label_id, cv2.CC_STAT_TOP], \
#                 stats_fore[label_id, cv2.CC_STAT_WIDTH], \
#                 stats_fore[label_id, cv2.CC_STAT_HEIGHT]
#             box = [x, y, x + w, y + h]
#             boxes.append(box)
#
#         # 生成点提示：质心（若满足条件）和距离变换最大值点
#         if area >= areafilter_fore:
#             # 计算质心（取整）
#             centroid = (int(round(centroids_fore[label_id][0])),
#                         int(round(centroids_fore[label_id][1])))
#
#             # 计算距离变换最大值点
#             max_dist_point = get_max_distance_point(dist_fore, labels_fore, label_id)
#
#             # 计算两点之间的欧氏距离
#             dist = math.hypot(centroid[0] - max_dist_point[0],
#                               centroid[1] - max_dist_point[1])
#
#             # 判断是否添加质心：质心在掩码中为白色 且 距离大于阈值
#             add_centroid = is_white(fore_mask, centroid) and dist > distance_centroid_distpoint
#
#             if add_centroid:
#                 points.append(centroid)
#                 labels.append(1)
#
#             # 添加距离变换最大值点（除非它与质心相同且质心已被添加）
#             if max_dist_point != centroid or not add_centroid:
#                 points.append(max_dist_point)
#                 labels.append(1)
#
#     # ---------- 背景处理 ----------
#     num_back, labels_back, stats_back, centroids_back = get_connected_components(back_mask)
#     dist_back = compute_distance_transform(back_mask)
#     back_contours, _ = cv2.findContours(back_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#     for label_id in range(1, num_back):
#         area = stats_back[label_id, cv2.CC_STAT_AREA]
#         if area < areafilter_back:
#             continue
#
#         # 原始格点采样
#         grid_pts = get_grid_points_in_region(labels_back, label_id, gridspace)
#         for pt in grid_pts:
#             points.append(pt)
#             labels.append(0)
#
#     # ---------- 额外：黑色连通区域密集采样 ----------
#     # 黑色区域是背景掩码中值为0的连通区域
#     black_mask = 255 - back_mask
#     _, black_labels, black_stats, _ = get_connected_components(black_mask)
#     dense_gridspace = max(1, int(gridspace / ratio))
#     for black_label in range(1, black_labels.max() + 1):
#         black_area = black_stats[black_label, cv2.CC_STAT_AREA]
#         if black_area >= foreregionarea:
#             # 获取该黑色连通区域的最小外接矩形
#             x, y, w, h = black_stats[black_label, cv2.CC_STAT_LEFT], \
#                 black_stats[black_label, cv2.CC_STAT_TOP], \
#                 black_stats[black_label, cv2.CC_STAT_WIDTH], \
#                 black_stats[black_label, cv2.CC_STAT_HEIGHT]
#
#             # 在矩形内密集采样
#             for i in range(x, x + w, dense_gridspace):
#                 for j in range(y, y + h, dense_gridspace):
#                     # 仅当该点在背景掩码中为白色（255）时才添加
#                     if back_mask[j, i] == 255:
#                         pt = (i, j)
#                         points.append(pt)
#                         labels.append(0)
#
#             # ---------- 新增：外接矩形等分采样 ----------
#             if n_division > 1:
#                 # 计算每个小矩形的宽度和高度
#                 small_w = w / n_division
#                 small_h = h / n_division
#
#                 # 遍历每个小矩形
#                 for row in range(n_division):
#                     for col in range(n_division):
#                         # 计算小矩形边界
#                         x1 = int(x + col * small_w)
#                         y1 = int(y + row * small_h)
#                         x2 = int(x + (col + 1) * small_w)
#                         y2 = int(y + (row + 1) * small_h)
#
#                         # 确保边界不超出图像范围
#                         x1 = max(0, min(x1, back_mask.shape[1] - 1))
#                         y1 = max(0, min(y1, back_mask.shape[0] - 1))
#                         x2 = max(0, min(x2, back_mask.shape[1]))
#                         y2 = max(0, min(y2, back_mask.shape[0]))
#
#                         # 保存小矩形用于可视化
#                         if backbox_flag == 1:
#                             small_rectangles.append([x1, y1, x2, y2])
#
#                         # 提取小矩形区域
#                         roi = back_mask[y1:y2, x1:x2]
#
#                         # 查找小矩形内的白色像素连通区域
#                         roi_white = roi.copy()
#                         roi_white[roi_white != 255] = 0
#
#                         # 使用连通组件分析找到白色区域的连通组件
#                         num_roi_labels, roi_labels, roi_stats, roi_centroids = get_connected_components(roi_white)
#
#                         # 对每个连通区域，取距离变换最大值点
#                         roi_dist = compute_distance_transform(roi_white)
#                         for roi_label in range(1, num_roi_labels):
#                             roi_area = roi_stats[roi_label, cv2.CC_STAT_AREA]
#                             if roi_area < 10:  # 忽略过小的区域
#                                 continue
#
#                             # 获取该区域的距离变换最大值点（在小矩形内的局部坐标）
#                             roi_region_mask = np.where(roi_labels == roi_label, 255, 0).astype(np.uint8)
#                             _, _, _, roi_max_loc = cv2.minMaxLoc(roi_dist, mask=roi_region_mask)
#
#                             # 转换为全局坐标
#                             global_x = x1 + roi_max_loc[0]
#                             global_y = y1 + roi_max_loc[1]
#
#                             # 确保点在背景掩码中为白色
#                             if is_white(back_mask, (global_x, global_y)):
#                                 points.append((global_x, global_y))
#                                 labels.append(0)
#
#     # ---------- 去重 ----------
#     unique_points = []
#     unique_labels = []
#     seen = set()
#     for pt, lb in zip(points, labels):
#         key = (pt[0], pt[1])
#         if key not in seen:
#             seen.add(key)
#             unique_points.append(pt)
#             unique_labels.append(lb)
#
#     # ---------- 筛选背景提示点 ----------
#     unique_points, unique_labels = filter_background_points(
#         unique_points, unique_labels, back_mask, min_background_point_distance
#     )
#
#     return unique_points, unique_labels, boxes, fore_contours, back_contours, small_rectangles
#
#
# def save_prompts_to_txt(points, labels, boxes, output_path):
#     """将提示点、标签和框保存到 txt 文件"""
#     with open(output_path, 'w') as f:
#         f.write("point prompts: {}\n".format(points))
#         f.write("labels: {}\n".format(labels))
#         f.write("box prompts: {}\n".format(boxes))
#
#
# def visualize_overlay(image, points, labels, boxes, fore_contours, back_contours,
#                       small_rectangles, output_path, backbox_flag):
#     """
#     在原图上叠加轮廓、提示点、提示框和小矩形并保存
#     """
#     overlay = image.copy()
#
#     # 绘制背景轮廓（绿色）
#     cv2.drawContours(overlay, back_contours, -1, (0, 255, 0), 1)
#     # 绘制前景轮廓（红色）
#     cv2.drawContours(overlay, fore_contours, -1, (0, 0, 255), 1)
#
#     # 绘制提示点
#     for pt, lb in zip(points, labels):
#         color = (0, 0, 255) if lb == 1 else (0, 255, 0)  # 前景红，背景绿
#         cv2.circle(overlay, pt, 4, color, -1)
#
#     # 绘制提示框（黄色）
#     for box in boxes:
#         x1, y1, x2, y2 = box
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
#
#     # 绘制小矩形（蓝色）
#     if backbox_flag == 1:
#         for rect in small_rectangles:
#             x1, y1, x2, y2 = rect
#             cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)
#
#     cv2.imwrite(output_path, overlay)
#
#
# def main():
#     # 选择三个文件夹
#     print("请选择原始图像文件夹：")
#     image_folder = select_directory("选择原图文件夹")
#     print("请选择前景掩码文件夹：")
#     fore_folder = select_directory("选择前景掩码文件夹")
#     print("请选择背景掩码文件夹：")
#     back_folder = select_directory("选择背景掩码文件夹")
#
#     # 获取所有图像文件
#     image_files = get_image_files(image_folder)
#     fore_files = get_image_files(fore_folder)
#     back_files = get_image_files(back_folder)
#
#     if not image_files:
#         print("原图文件夹为空，程序退出。")
#         return
#
#     # 创建输出子文件夹
#     output_dir = os.path.join(image_folder, output_subfolder)
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 建立前缀索引字典
#     def prefix_key(filepath):
#         name = os.path.splitext(os.path.basename(filepath))[0]
#         return name[:prefix_length] if len(name) >= prefix_length else name
#
#     fore_dict = {}
#     for fp in fore_files:
#         key = prefix_key(fp)
#         fore_dict.setdefault(key, []).append(fp)
#
#     back_dict = {}
#     for fp in back_files:
#         key = prefix_key(fp)
#         back_dict.setdefault(key, []).append(fp)
#
#     processed = 0
#     for img_path in image_files:
#         key = prefix_key(img_path)
#         if key not in fore_dict or key not in back_dict:
#             print(f"警告：前缀 '{key}' 没有对应的前景或背景掩码，跳过 {os.path.basename(img_path)}")
#             continue
#
#         # 取第一个匹配的文件（假设每个前缀唯一）
#         fore_path = fore_dict[key][0]
#         back_path = back_dict[key][0]
#
#         print(f"处理：{os.path.basename(img_path)} (前缀: {key})")
#
#         # 读取图像
#         image = read_image_bgr(img_path)
#         fore_mask = read_binary_mask(fore_path)
#         back_mask = read_binary_mask(back_path)
#
#         # 检查尺寸一致性，若不匹配则调整掩码到原图尺寸
#         if image.shape[:2] != fore_mask.shape or image.shape[:2] != back_mask.shape:
#             fore_mask = cv2.resize(fore_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
#             back_mask = cv2.resize(back_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
#
#         # 生成提示
#         points, labels, boxes, fore_contours, back_contours, small_rectangles = generate_prompts(
#             fore_mask, back_mask,
#             Areafilter_fore, Areafilter_back, gridspace,
#             foreareaforbox, boxprompt, min_background_point_distance,
#             n_division, backbox
#         )
#
#         # 输出统计信息
#         num_fg = sum(1 for lb in labels if lb == 1)
#         num_bg = sum(1 for lb in labels if lb == 0)
#         print(f"  前景提示点: {num_fg}, 背景提示点: {num_bg}, 提示框: {len(boxes)}, 小矩形: {len(small_rectangles)}")
#
#         # 保存 txt
#         base_name = os.path.splitext(os.path.basename(img_path))[0]
#         txt_output = os.path.join(output_dir, f"{base_name}_prompts.txt")
#         save_prompts_to_txt(points, labels, boxes, txt_output)
#
#         # 保存叠加图像
#         overlay_output = os.path.join(output_dir, f"{base_name}_imageoverlap.png")
#         visualize_overlay(image, points, labels, boxes, fore_contours, back_contours,
#                           small_rectangles, overlay_output, backbox)
#
#         processed += 1
#
#     print(f"批量处理完成，共处理 {processed} 组图像。结果保存在：{output_dir}")
#
#
# if __name__ == "__main__":
#     main()
#
