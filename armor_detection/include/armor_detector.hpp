#ifndef ARMOR_DETECTOR_HPP
#define ARMOR_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "armor.hpp"

class ArmorDetector {
public:
    ArmorDetector();
    
    /**
     * @brief 处理图像并检测装甲板
     * 
     * @param src 输入图像
     * @param debug_img 调试输出图像（显示检测结果）
     * @return std::vector<armor::Armor> 检测到的装甲板列表
     */
    std::vector<armor::Armor> detectArmors(const cv::Mat &src, cv::Mat &debug_img);
    
    /**
     * @brief 计算装甲板的位姿
     * 
     * @param armors 检测到的装甲板
     * @param tvecs 平移向量输出
     * @param rvecs 旋转向量输出
     * @param cameraMatrix 相机内参矩阵
     * @param distCoeffs 相机畸变系数
     */
    void calculatePose(const std::vector<armor::Armor> &armors, 
                      std::vector<cv::Mat> &tvecs, 
                      std::vector<cv::Mat> &rvecs,
                      const cv::Mat &cameraMatrix,
                      const cv::Mat &distCoeffs);

private:
    // 装甲板检测参数
    const float SMALL_ARMOR_WIDTH = 0.135;
    const float SMALL_ARMOR_HEIGHT = 0.055;
    const std::vector<cv::Point3f> SMALL_ARMOR_POINTS = {
        {0, +SMALL_ARMOR_WIDTH / 2, -SMALL_ARMOR_HEIGHT / 2},
        {0, +SMALL_ARMOR_WIDTH / 2, +SMALL_ARMOR_HEIGHT / 2},
        {0, -SMALL_ARMOR_WIDTH / 2, +SMALL_ARMOR_HEIGHT / 2},
        {0, -SMALL_ARMOR_WIDTH / 2, -SMALL_ARMOR_HEIGHT / 2}
    };
    
    // 灯条筛选参数
    const float target_ratio = 3.0f;  // 灯条长宽比
    const float tolerance = 6.0f;     // 灯条长宽比容差
    const float angle_tolerance = 3.0f; // 灯条角度容差
    
    // 装甲板筛选参数
    const float armor_ratio = 1.7f;   // 装甲板长宽比
    const float armor_tolerance = 1.0f; // 装甲板长宽比容差
    
    /**
     * @brief 预处理图像
     * 
     * @param src 输入图像
     * @param binary 输出二值图
     */
    void preprocess(const cv::Mat &src, cv::Mat &binary);
    
    /**
     * @brief 检测灯条
     * 
     * @param binary 二值图
     * @param light_rects 输出灯条信息
     * @param debug_img 调试图像
     */
    void detectLights(const cv::Mat &binary, 
                     std::vector<std::pair<cv::RotatedRect, std::vector<cv::Point2f>>> &light_rects,
                     cv::Mat &debug_img);
    
    /**
     * @brief 匹配装甲板
     * 
     * @param light_rects 灯条信息
     * @param armors 输出装甲板
     * @param debug_img 调试图像
     */
    void matchArmors(const std::vector<std::pair<cv::RotatedRect, std::vector<cv::Point2f>>> &light_rects,
                    std::vector<armor::Armor> &armors,
                    cv::Mat &debug_img);
                    
    /**
     * @brief 欧拉角转换辅助函数
     * 
     * @param R 旋转矩阵
     * @return cv::Mat 欧拉角
     */
    cv::Mat rotationMatrixToEulerAngles(const cv::Mat &R);

    /**
     * @brief 亮度和对比度调整辅助函数
     * 
     * @param src 输入图像
     * 
     * @param dst 输出图像
     * 
     */
    void adjustBrightnessContrast(const cv::Mat &src, cv::Mat &dst, double alpha, int beta);
};

#endif // ARMOR_DETECTOR_HPP