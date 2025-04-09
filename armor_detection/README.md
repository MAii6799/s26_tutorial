# 装甲板检测系统 (Armor Detection System)

这是一个基于OpenCV的装甲板检测系统，用于RoboMaster比赛中识别机器人装甲板并估计其位姿。

## 项目概述

本项目实现了对RoboMaster机器人装甲板的检测功能，主要包括：
1. 图像预处理
2. 灯条检测
3. 装甲板匹配
4. 位姿估计

## 目录结构

```
armor_detection/
├── CMakeLists.txt      # 项目构建文件
├── armors/             # 测试用的装甲板图像
├── build/              # 构建输出目录
├── include/            # 头文件
│   ├── armor.hpp       # 装甲板结构定义
│   └── armor_detector.hpp # 装甲板检测器类定义
└── src/                # 源代码
    ├── armor_detector.cpp      # 装甲板检测器实现
    └── armor_detector_demo.cpp # 演示程序
```

## 核心组件

### 1. 装甲板结构 (armor.hpp)

定义了装甲板的数据结构：
- `Light` 结构：表示灯条，包含顶部和底部坐标
- `Armor` 结构：表示装甲板，包含左右灯条、类型、数字图像和识别结果

### 2. 装甲板检测器 (armor_detector.hpp)

实现了装甲板的检测算法：
- 图像预处理：亮度对比度调整、颜色分离和二值化
- 灯条检测：轮廓查找和筛选
- 装甲板匹配：基于灯条对匹配装甲板
- 位姿估计：使用PnP算法计算装甲板的3D位置和姿态

## 使用方法

### 构建项目

```bash
# 创建并进入构建目录
mkdir -p build && cd build

# 使用CMake构建项目
cmake ..
make

# 运行演示程序
./armor_detector_demo
```

### 代码示例

```cpp
// 创建装甲板检测器
ArmorDetector detector;

// 处理图像
cv::Mat frame = cv::imread("path/to/image.jpg");
cv::Mat debug_img;
std::vector<armor::Armor> armors = detector.detectArmors(frame, debug_img);

// 计算位姿
std::vector<cv::Mat> tvecs, rvecs;
detector.calculatePose(armors, tvecs, rvecs, cameraMatrix, distCoeffs);
```

## 参数说明

装甲板检测器包含多个重要的参数：
- 灯条筛选：长宽比、角度容差
- 装甲板匹配：长宽比、尺寸容差
- 位姿估计：装甲板尺寸标定（小装甲板：135mm×55mm）

## 测试数据

在`armors/`目录下提供了多张测试图片，可用于测试装甲板检测算法。

## 依赖项

- OpenCV 4.x
- CMake 3.10+
- C++14 兼容的编译器

## 性能特点

1. 支持小装甲板和大装甲板的识别
2. 实现了装甲板的位姿估计，可用于后续的瞄准和跟踪
3. 包含调试可视化，便于算法优化

## 未来改进

- 增加深度学习模型进行数字识别
- 改进灯条匹配算法，提高识别鲁棒性
- 添加装甲板跟踪功能，处理装甲板的遮挡问题