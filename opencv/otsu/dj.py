import cv2
import cv2.ximgproc
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('../img/md.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original Image', img)

# 等比缩小图片
scale_percent = 100
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# 调整图像大小
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# 添加滑块控制阈值
def update_threshold(val):
    _, img_thresh = cv2.threshold(img, val, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold Control', img_thresh)

# 创建窗口和滑块
cv2.namedWindow('Threshold Control')
cv2.createTrackbar('Threshold', 'Threshold Control', 127, 255, update_threshold)

# 初始化显示
update_threshold(127)

# 1. OpenCV 大津法
thresh_opencv, binary_opencv = cv2.threshold(
    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
# 2. 手动实现大津法
def manual_otsu(image):
    # 计算直方图
    histogram, _ = np.histogram(image.flatten(), 256, [0, 256])
    total_pixels = np.sum(histogram)

    best_threshold = 0 # 最佳阈值
    max_variance = 0 # 最大类间方差

    current_weight_bg = 0 # 当前背景权重
    current_mean_bg = 0 # 当前背景灰度均值
    current_weight_fg = 0 # 当前前景权重
    current_mean_fg = 0 # 当前前景灰度均值

    mean_bg = 0 # 最终背景均值
    mean_fg = 0 # 最终前景均值

    sum_bg = 0 # 当前背景灰度和
    sum_fg = 0 # 前景灰度和

    for i in range(256):
        sum_fg += i * histogram[i]

    for threshold in range(250): # 遍历阈值
        # 更新当前权重
        current_weight_bg += histogram[threshold]
        current_weight_fg = total_pixels - current_weight_bg
        # 如果当前权重为0，则跳过
        if current_weight_bg == 0 or current_weight_fg == 0:
            continue
        # 更新当前均值
        current_mean_bg = sum_bg / current_weight_bg
        current_mean_fg = (sum_fg - sum_bg) / current_weight_fg
        # 更新类间方差
        # variance = current_weight_bg * current_weight_fg * (current_mean_bg - current_mean_fg) ** 2
        current_weight = current_weight_bg * current_mean_bg + current_weight_fg * current_mean_fg
        variance = (current_weight_bg * (current_weight - current_mean_bg) ** 2 + current_weight_fg * (current_weight - current_mean_fg) ** 2)
        print(f"Threshold: {threshold}, Variance: {variance}, Current Mean BG: {current_mean_bg}, Current Mean FG: {current_mean_fg}")
        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold
            mean_bg = current_mean_bg
            mean_fg = current_mean_fg
        sum_bg += histogram[threshold] * threshold
    return best_threshold, mean_bg, mean_fg

thresh_manual, mean_bg, mean_fg = manual_otsu(img)
print("Otsu Threshold:", thresh_manual)

# 手动二值化
binary_manual = np.where(img < thresh_manual, 0, 255).astype(np.uint8)

# 3. 绘制直方图和阈值
histogram, bins = np.histogram(img.flatten(), 256, [0, 256])
cumulative_sum = np.cumsum(histogram)
total_pixels = np.sum(histogram)
plt.figure(figsize=(12, 6))

# 绘制直方图
plt.subplot(1, 2, 1)
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.axvline(x=thresh_manual, color='b', linestyle='--', label=f'otsu Threshold = {thresh_manual}')
plt.axvline(x=mean_bg, color='purple', linestyle='--', label=f'Background Mean = {mean_bg:.2f}')
plt.axvline(x=mean_fg, color='orange', linestyle='--', label=f'Foreground Mean = {mean_fg:.2f}')
plt.title('Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()

# 绘制图像
plt.subplot(1, 2, 2)
plt.imshow(binary_manual, cmap='gray')
plt.title('Otsu Binarization')
plt.xticks([]), plt.yticks([])

plt.tight_layout()

# 显示图像
cv2.imshow('Manual Otsu Binarization', binary_manual)

# 使用自适应阈值对图像进行二值化
binary_adaptive = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# 显示自适应阈值结果
cv2.imshow('Adaptive Thresholding', binary_adaptive)

def manual_max_diff(img):
    # 计算直方图
    histogram, _ = np.histogram(img.flatten(), 256, [0, 256])
    total_pixels = np.sum(histogram)

    best_thresh = 0  # 最佳阈值
    max_mean_diff = 0  # 最大均值差

    current_weight_bg = 0  # 当前背景权重
    current_mean_bg = 0  # 当前背景均值
    current_weight_fg = 0  # 当前前景权重
    current_mean_fg = 0  # 当前前景均值

    sum_bg = 0  # 当前背景灰度和
    sum_fg = 0  # 当前前景灰度和

    for i in range(250):
        sum_fg += i * histogram[i]

    for threshold in range(256):  # 遍历阈值
        current_weight_bg += histogram[threshold]
        current_weight_fg = total_pixels - current_weight_bg
        if current_weight_bg == 0 or current_weight_fg == 0:
            continue
        current_mean_bg = sum_bg / current_weight_bg
        current_mean_fg = (sum_fg - sum_bg) / current_weight_fg
        mean_diff = abs(current_mean_bg - current_mean_fg)
        if mean_diff > max_mean_diff:
            max_mean_diff = mean_diff
            best_thresh = threshold
            mean_bg = current_mean_bg
            mean_fg = current_mean_fg
        sum_bg += histogram[threshold] * threshold
    
    return best_thresh, mean_bg, mean_fg

thresh_manual, mean_bg, mean_fg = manual_max_diff(img)
print("max_diff Threshold:", thresh_manual)


# 手动二值化
binary_manual = np.where(img < thresh_manual, 0, 255).astype(np.uint8)

# 3. 绘制直方图和阈值
histogram, bins = np.histogram(img.flatten(), 256, [0, 256])
cumulative_sum = np.cumsum(histogram)
total_pixels = np.sum(histogram)
plt.figure(figsize=(12, 6))

# 绘制直方图
plt.subplot(1, 2, 1)
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.axvline(x=thresh_manual, color='b', linestyle='--', label=f'max_diff Threshold = {thresh_manual}')
plt.axvline(x=mean_bg, color='purple', linestyle='--', label=f'Background Mean = {mean_bg:.2f}')
plt.axvline(x=mean_fg, color='orange', linestyle='--', label=f'Foreground Mean = {mean_fg:.2f}')
plt.title('Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()

# 绘制图像
plt.subplot(1, 2, 2)
plt.imshow(binary_manual, cmap='gray')
plt.title('max_diff Binarization')
plt.xticks([]), plt.yticks([])

plt.tight_layout()

# # 设置 niBlackThreshold 的参数
# block_size = 15  # 窗口大小
# k = 0.2  # 调整参数
# max_value = 255  # 最大值
# threshold_type = cv2.THRESH_BINARY  # 二值化类型

# # 应用 niBlackThreshold
# thresholded_image = cv2.ximgproc.niBlackThreshold(
#     img,
#     maxValue=max_value,
#     blockSize=block_size,
#     k=k,
#     type=threshold_type
# )

# # 显示原图和处理后的图像
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Thresholded Image')
# plt.imshow(thresholded_image, cmap='gray')
# plt.axis('off')

cv2.waitKey(100)

plt.show()

cv2.waitKey(0)