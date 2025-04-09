import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv2.imread('../img/image.png')
# 等比缩小图片
scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# 调整图像大小
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Original Image', img)

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', gray)

# 获取图像的高度和宽度
h, w = gray.shape

# 创建网格
x = np.arange(0, w, 1)
y = np.arange(0, h, 1)
x, y = np.meshgrid(x, y)

# 创建图像的3D表示
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制3D图像
ax.plot_surface(x, y, gray, cmap='gray')

# cv2.GaussianBlur(gray, (5, 5), 0, gray)
# cv2.imshow('Gaussian Blurred Image', gray)

# 计算图像的梯度
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

# 归一化梯度
sobel_x_1 = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX)
sobel_y_1 = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX)
# 转换为8位图像
sobel_x_1 = np.uint8(sobel_x_1)
sobel_y_1 = np.uint8(sobel_y_1)

cv2.imshow('Sobel X', sobel_x_1)
cv2.imshow('Sobel Y', sobel_y_1)

# 计算梯度幅值和方向
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
direction = np.arctan2(sobel_y, sobel_x)

# 归一化幅值
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
# 转换为8位图像
magnitude = np.uint8(magnitude)
# 显示梯度幅值图像
cv2.imshow('Gradient Magnitude', magnitude)

# 显示梯度方向图像
direction_1 = cv2.normalize(direction, None, 0, 255, cv2.NORM_MINMAX)
direction_1 = np.uint8(direction_1)
cv2.imshow('Gradient Direction', direction_1)


# 非最大抑制
def non_max_suppression(magnitude, direction):
    h, w = magnitude.shape
    suppressed = np.zeros((h, w), dtype=np.uint8)
    angle = direction * 180.0 / np.pi  # 转换为角度
    angle[angle < 0] += 180  # 确保角度为正

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = 255
            r = 255

            # 根据梯度方向选择相邻像素
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i - 1, j + 1]
                r = magnitude[i + 1, j - 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i - 1, j]
                r = magnitude[i + 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            # 抑制非最大值
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed

# 应用非最大抑制
nms_result = non_max_suppression(magnitude, direction)
cv2.imshow('Non-Max Suppression', nms_result)

# 双阈值处理
def double_threshold(image, low_threshold, high_threshold):
    strong = 255
    weak = 75
    h, w = image.shape
    result = np.zeros((h, w), dtype=np.uint8)

    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result
# 应用双阈值处理
low_threshold = 50
high_threshold = 150
threshold_result = double_threshold(nms_result, low_threshold, high_threshold)
cv2.imshow('Double Threshold', threshold_result)
# 边缘连接
def edge_tracking(image, weak, strong):
    h, w = image.shape
    result = np.zeros((h, w), dtype=np.uint8)

    strong_i, strong_j = np.where(image == strong)

    for i, j in zip(strong_i, strong_j):
        if image[i, j] == strong:
            result[i, j] = strong
            # 检查8个方向的弱边缘
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if image[i + x, j + y] == weak:
                        result[i + x, j + y] = strong

    return result
# 应用边缘连接
edge_result = edge_tracking(threshold_result, 75, 255)
cv2.imshow('Edge Tracking', edge_result)

# 计算边缘图像
edges = cv2.Canny(gray, 50, 150)
# 显示边缘图像
cv2.imshow('Canny Edges', edges)

# findContours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓
for cnt in contours:
    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
# 显示轮廓图像
cv2.imshow('Contours', img)

# # 显示纯轮廓
# contours_img = np.zeros_like(img)
# for cnt in contours:
#     cv2.drawContours(contours_img, [cnt], 0, (255, 255, 255), -1)
# cv2.imshow('Contours Only', contours_img)

cv2.waitKey(1000)

plt.show()

cv2.waitKey(0)
