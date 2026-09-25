#https://cloud.tencent.com/developer/article/1625876
import warnings
import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage
from skimage import feature, exposure
from skimage.morphology import remove_small_objects
import numpy as np

warnings.filterwarnings("ignore")

img = cv2.imread("B:/software-patent/graphenefraction/AFM/AP-10min-15du/try/10min-15du.0_00002.jpg", 1)
# img = cv2.imread("B:/software-patent/graphenefraction/AFM/AP-10min-15du/sinkBPFtransform_10.jpg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

features, output = skimage.feature.hog(
    gray, orientations=9,
    pixels_per_cell=(8,8),
    cells_per_block=(3, 3),
    block_norm='L2-Hys',
    visualize=True,
    transform_sqrt=False,
    feature_vector=True,

 )
print(features.shape)# Rescale histogram for betterdisplayoutput = skimage.exposure.rescale_intensity(output, in_range=(0, 10))f =plt.figure(figsize=(15,15))
# f.add_subplot(2, 1, 1).set_title('Original Image')
# plt.imshow(img[:, :,::-1])
# f.add_subplot(2, 1, 2).set_title('Features')
# 方法2：保留强度关系但映射到白色（可选）
# # 调整强度范围并反转颜色
# hog_normalized = exposure.rescale_intensity(output, in_range=(0, np.max(output)))

# 将所有特征变为白色
hog_morph = np.where(output > 1, 255, 0).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def remove_isolated_pixels_skimage(image, dropsize = 49):
    # 转换为布尔类型（skimage要求）
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = image > 0  # 假设非零为前景
    # 去除小于min_size像素的连通区域
    cleaned = remove_small_objects(binary, min_size=dropsize)
    # 转回图像格式
    cleaned = cleaned.astype(np.uint8) * 255
    return cleaned

# hog_morph = cv2.dilate(hog_morph, kernel, iterations= 1)
# # hog_morph = cv2.erode(hog_morph, kernel, iterations= 1)
# hog_morph = remove_isolated_pixels_skimage( hog_morph)
cv2.imwrite( "B:/software-patent/graphenefraction/AFM/AP-10min-15du/try/10min-15du.0_00002-hog.jpg", hog_morph)
plt.title("HOG")
plt.imshow(hog_morph, cmap='gray')
plt.show()


contours, _ = cv2.findContours(hog_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
HPFmask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
num_labels, label_im, stats, centroids = cv2.connectedComponentsWithStats(hog_morph, connectivity=4)


for contour in contours:
    area = cv2.contourArea(contour)
    if area<100:
        cv2.drawContours(HPFmask, contour, -1, 255, thickness=-1)

plt.title("HPFmask")
plt.imshow(HPFmask, cmap='gray')
plt.show()


