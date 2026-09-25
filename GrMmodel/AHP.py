#https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb.
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import time
import shutil
# 获取当前时间
start_time = time.time()

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def get_mask_contour(mask):
    # 将掩码转换为二值图像
    mask = mask.astype(np.uint8) * 255
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

#https://blog.csdn.net/qq_34451909/article/details/109254204.OpenCV-python 用鼠标在图片上标记位置并返回坐标
def SetPoints(windowname, img):
    """
    输入图片，打开该图片进行标记点，返回的是标记的几个点的字符串
    """
    print('(提示框绘制方法：鼠标左键单击第一个顶点拖动到对角线另一个顶点后松开左键。提示点绘制方法：鼠标左键单击绘制前景点，鼠标右键单击绘制背景点。重复此操作直至绘制完所有提示框和提示点。每按一次Backspace可撤销最新绘制的提示框或者提示点。最后按Enter/ESC结束提示，并开始对图像进行预测。)')

def on_mouse_event(event, x, y, flags, param):
    global drawing_box, drawing_point, rect_start, rect_end, input_boxes, input_points, temp_img, path, imagenm, windowname
    called_predict = False

    if event == cv2.EVENT_LBUTTONDOWN:       #按下鼠标左键
        rect_start = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:       #抬起鼠标左键
        rect_end = (x, y)
        if abs(rect_start[0] - rect_end[0]) * abs(rect_start[1] - rect_end[1]) < 100:      #如果鼠标左键按下和抬起的像素位置上下左右面积差不超过100个像素点的距离，则认为画的是点prompt，标前景部分，标签为1。
            drawing_point = True
            drawing_box = False
            points.append((x, y))
            labels.append(1)
            cv2.circle(temp_img, (x, y), 5, (0, 0, 255), -1)
            # print("全部绘制的点为：", points)
            # print("点的标签为：", labels)
        else:                                                                              #否则认为画的是框prompt。
            drawing_point = False
            drawing_box = True
            insitu_box = [min(rect_start[0], rect_end[0]), min(rect_start[1], rect_end[1]),
                     max(rect_start[0], rect_end[0]), max(rect_start[1], rect_end[1])]
            boxes.append(insitu_box)
            cv2.rectangle(temp_img, rect_start, rect_end, (102, 217, 239), 2)
            # print("全部绘制的框为：", boxes)
        called_predict = True

    elif event == cv2.EVENT_RBUTTONDOWN:     #按下鼠标右键标背景部分，标签label为0。
        drawing_point = True
        drawing_box = False
        points.append((x, y))
        labels.append(0)
        cv2.circle(temp_img, (x, y), 5, (0, 255, 0), -1)
        # print("全部绘制的点为：", points)
        called_predict = True

    elif event == cv2.EVENT_MBUTTONDOWN:     #按下鼠标中键则和右键一样也是标背景部分。
        drawing_point = True
        drawing_box = False
        points.append((x, y))
        labels.append(0)
        cv2.circle(temp_img, (x, y), 5, (0, 255, 0), -1)
        # print("全部绘制的点为：", points)
        called_predict = True

    # if called_predict:
    #     # print("called_predict", called_predict)
    #     maskimage, _ = predict_mask(path, imagenm, temp_img)
    #     cv2.namedWindow('mask in-situ', 0)
    #     # 调整窗口大小为指定宽度和高度
    #     cv2.resizeWindow(windowname, 500, 500)
    #     cv2.imshow("mask in-situ", maskimage)



def show_mask(pathroot, imagename, mask):
    global path_seg
#输入图像名字imageename，如“F1.jpg",调用GrSAM模型预测的mask图像保存在path_seg中，按照mask将原图扣出来的部分保存在path_maskpart文件夹中。
    random_color = False
    pathimage = os.path.join(pathroot, imagename)
    img = cv2.imread(pathimage)
    if random_color:   #random_color默认为False，只输出黑白的mask图像。
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    # 将mask转换为uint8类型，并将True转换为255，False转换为0
    mask = mask.astype(np.uint8) * 255
    h, w = mask.shape[-2:]
    mask_save = np.where(mask != 0, 255, 0).astype(np.uint8)
    mask_save = mask_save.reshape(h, w, 1)
    # path_seg = os.path.join(pathroot, (pathimage.split("\\")[-1]).split(".")[0] + ".png")

    cv2.imwrite(path_seg, mask_save)
    # print("mask图像保存在：", path_seg)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    mask_3channel = np.stack((mask,) * 3, axis=-1)
    # 将mask转换为具有3个通道的图像，以便与原始图像具有相同的通道数
    # 将mask转换为前景掩码
    foreground_mask = mask.astype(np.uint8) * 255
    # 使用前景掩码提取前景部分
    foreground = cv2.bitwise_and(img, img, mask=foreground_mask)
    # 将预测结果覆盖在原图上。
    overlay = img.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.5).astype(np.uint8)
    # 创建一个全新的具有4个通道（RGB + Alpha）的图像
    foreground_with_alpha = np.ones((*img.shape[:2], 4), dtype=np.uint8) * 255
    # 将mask中的前景部分作为alpha通道
    alpha_channel = np.where(mask_3channel == 0, 0, 255)
    # 将alpha通道赋值给前景图像的alpha通道
    foreground_with_alpha[:, :, 3] = alpha_channel[:, :, 0]
    # 将前景图像中的像素值复制到新图像的RGB通道
    foreground_with_alpha[:, :, :3] = foreground
    # 将像素值缩放到[0, 255]的范围，并转换为uint8类型
    foreground_with_alpha = foreground_with_alpha.astype(np.uint8)
    # 创建PIL图像对象
    image_pil = Image.fromarray(foreground_with_alpha, 'RGBA')
    # 保存图像为PNG格式
    path_maskpart = os.path.join(os.path.dirname(pathroot), 'imagemask', (imagename.split(".")[0] + ".png"))
    os.makedirs(os.path.dirname(path_maskpart), exist_ok=True)
    image_pil.save(path_maskpart)
    # print("mask结合image的图像保存在：", path_maskpart)
    return (mask_image, path_seg, overlay)


def predict_mask(pathroot, imagename, image):
    global input_boxes, input_points, predictor
    if not points and not boxes:
        return
    input_points = torch.tensor(points)
    # input_labels = torch.tensor([point[-1] for point in points_labels])
    input_labels = torch.tensor(labels)
    input_boxes = torch.tensor(boxes)

    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=input_boxes,
        multimask_output=True
    )
    mask_input = logits[np.argmax(scores), :, :]  # 选score最高的掩码
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=input_boxes,
        mask_input=mask_input[None, :, :],
        multimask_output=False
    )
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
        # print("label_save_GrSAM:", label_save_GrSAM)
        # result_img = show_mask(SEMImg, mask_resized, input_points.tolist(), input_boxes.tolist(), label_save_GrSAM)
        mask_image, path_seg, overlay = show_mask(pathroot, imagename, mask_resized)

    return(mask_image)


def process_image(pathroot, imagename, predictor):
    global temp_img, path, imagenm, windowname
    path = pathroot
    imagenm = imagename               #imagenm为图像的名称如“F1.jpg”
    pathimage = os.path.join(path, imagenm)
    image = cv2.imread(pathimage)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 通过调用Sam Predictor.set_image对图像进行处理，生成图像嵌入编码。Sam Predictor记住这个嵌入，并将其用于后续的掩码预测。
    # 要选择前景，在其上选择一个点。点以(x,y)格式输入模型，并附有标签1(前景点)或0(背景点)。可以输入多个点；这里我们只使用一个。选取的点将在图像上显示为星点。
    # 创建窗口名称
    windowname = 'Prompts from user'   #显示一个绘图窗口用于用户根据图像用鼠标通过点击和框选提供目标物的prompts。
    # 在之前的代码中添加两个新的列表
    # 创建临时图像副本
    temp_img = image.copy()
    # 创建窗口并设置鼠标回调函数
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    # 调整窗口大小为指定宽度和高度
    cv2.resizeWindow(windowname, 500, 500)
    # 使用 lambda 函数将 image 参数传递给 on_mouse_event 函数
    cv2.setMouseCallback(windowname, lambda event, x, y, flags, param=image: on_mouse_event(event, x, y, flags, param))

    while True:
        # 在图像上显示标注的点和框
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (102, 217, 239), 2)
        for point in points:
            x, y = point
            # 寻找points列表中包含x和y的索引
            # print("point", point)
            try:
                index = points.index(point)
                # print(f"索引为: {index}")
                label = labels[index]
            except ValueError:
                print("Can't find the point.")

            color = (0, 0, 255) if label == 1 else (0, 255, 0)  # BGR格式
            cv2.circle(temp_img, (x, y), 5, color, -1)

        cv2.imshow(windowname, temp_img)
        maskimage = predict_mask(path, imagenm, image)

        if maskimage is None or maskimage.shape[0] <= 0 or maskimage.shape[1] <= 0:
            maskimage = np.zeros((500, 500), dtype=np.uint8)  # 示例纯黑色图像，你可以根据需要调整大小和数据类型


        # 检查 maskimage 的尺寸
        if maskimage.shape[0] > 0 and maskimage.shape[1] > 0:
            cv2.namedWindow('mask in-situ', 0)
            # 调整窗口大小为指定宽度和高度
            cv2.resizeWindow(windowname, 500, 500)
            cv2.imshow("mask in-situ", maskimage)
        else:
            print("图像尺寸无效，无法显示。")

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print('跳过该张图片')
            cv2.destroyAllWindows()
            break
        elif key == 8:  # Backspace键
            if drawing_box:
                print(f"the box {boxes[-1]} was deleted.")
                boxes.pop()  # 删除最后一个绘制的框
            elif drawing_point:
                print(f"the point {points[-1]} was deleted.")
                points.pop()  # 删除最后一个绘制的点
                labels.pop()  # 删除最后一个绘制的点的标签

            maskimage = predict_mask(path, imagenm, image)    #删除点后得重新回到原来的maskimage状态。
            cv2.namedWindow('mask in-situ', 0)
            # 调整窗口大小为指定宽度和高度
            cv2.resizeWindow(windowname, 500, 500)
            cv2.imshow("mask in-situ", maskimage)

        elif key == 13:  # Enter
            cv2.destroyAllWindows()
            break
    # 在图像处理完成后关闭窗口
    cv2.destroyAllWindows()



# sam_checkpoint = 'B:\\code\\GrSAM\\data\\work_dir\\SAM-ViT-B\\sam_vit_b_01ec64.pth'
sam_checkpoint = 'B:\\code\\GrSAM\\data\\work_dir\\SAM-ViT-B\\sam_vit_b_01ec64.pth'
path = 'B:\\code\\GrSAM\\data\\Test\\label\\F1\\Image'
txt_path = 'B:\\code\\GrSAM\\data\\Test\\label\\F1\\GrSAM-APH\\F1-GrSAM-APH-[126, 32].txt'

model_type = "vit_b"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
# 获取文件夹中所有文件名
file_names = os.listdir(path)
# 循环处理每个图像
for file_name in file_names:
    # 初始化一个空列表来存储读取的 box prompts, points, labels
    boxes = []
    points = []
    labels = []

    # 打开文件以读取模式
    with open(txt_path, 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 检查每一行是否包含 "box prompts"，如果是，则提取后面的值
            if "box prompts" in line:
                # 获取 "box prompts" 后面的部分，并用 split() 分割成一个列表
                boxes_str = line.split("box prompts:")[1].strip()
                # 将字符串表示的列表转换为实际的列表
                boxes = eval(boxes_str)
            elif "point prompts" in line:
                points_str = line.split("point prompts:")[1].strip()
                # 将字符串表示的列表转换为实际的列表
                points = eval(points_str)
            elif "labels" in line:
                labels_str = line.split("labels:")[1].strip()
                # 将字符串表示的列表转换为实际的列表
                labels = eval(labels_str)

    # points_labels = [point + [label] for point, label in zip(points, labels)]    #把point，label结合到points_labels里面。

    # # 输出加载的 box prompts
    # print("Loaded points:", points)
    # print("Loaded points:", labels)
    # print("Loaded Box Prompts:", boxes)


    if file_name.endswith(".png") or file_name.endswith(".jpg"):  # 确保只处理图像文件
        # 图像文件的完整路径
        image_path = os.path.join(path, file_name)
        path_seg = os.path.join(os.path.dirname(path), 'SAM-APH', (file_name.split(".")[0] + '-SAM-APH' + '-' + str(points[len(points)-1]) +".png"))
        os.makedirs(os.path.dirname(path_seg), exist_ok=True)

        # 处理单张图像
        process_image(pathroot=path, imagename=file_name, predictor=predictor)
        txt_path_APH = os.path.join(path_seg.split(".")[0] + '.txt')
        # 获取当前时间
        end_time = time.time()
        # 计算并输出代码执行时间
        execution_time = end_time - start_time
        print(f"All points prompts are:{points}, ")
        print(f"with labels{labels}")
        print(f"All boxes prompts are:{boxes}.")
        print(f"总测试时间: {execution_time} seconds.")

        # 打开文件以写入模式
        with open(txt_path_APH, 'w') as file:
            # 写入 point prompts 到文件
            file.write("point prompts: {}\n".format(points))
            # 写入 labels 到文件
            file.write("labels: {}\n".format(labels))
            # 写入 box prompts 到文件
            file.write("box prompts: {}\n".format(boxes))
            file.write("Time cost:{}s\n".format(execution_time))
        print("Time cost and all promts are writen in the txt:", txt_path_APH)
