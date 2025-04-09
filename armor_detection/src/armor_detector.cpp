#include "../include/armor_detector.hpp"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <omp.h>

// 构造函数
ArmorDetector::ArmorDetector() {}

// 辅助类：用于并行处理图像亮度对比度调整
class BrightnessContrastAdjuster : public cv::ParallelLoopBody {
public:
    BrightnessContrastAdjuster(const cv::Mat &src, cv::Mat &dst, double alpha, int beta)
        : src_(src), dst_(dst), alpha_(alpha), beta_(beta) {}

    virtual void operator()(const cv::Range &range) const override {
        for (int y = range.start; y < range.end; y++) {
            for (int x = 0; x < src_.cols; x++) {
                for (int c = 0; c < 3; c++) {
                    dst_.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(alpha_ * src_.at<cv::Vec3b>(y, x)[c] + beta_);
                }
            }
        }
    }

private:
    const cv::Mat &src_;
    cv::Mat &dst_;
    double alpha_;
    int beta_;
};

// 旋转矩阵转欧拉角
cv::Mat ArmorDetector::rotationMatrixToEulerAngles(const cv::Mat &R) {
    float sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    
    bool singular = sy < 1e-6; // 如果 sy 接近于零，则矩阵接近奇异
    
    float x, y, z;
    if (!singular) {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }

    cv::Mat eulerAngles = (cv::Mat_<double>(3, 1) << x, y, z);
    return eulerAngles;
}

// 图像预处理
void ArmorDetector::preprocess(const cv::Mat &src, cv::Mat &binary) {
    // 调整亮度和对比度
    cv::Mat adjusted_img = src.clone();
    double alpha = 0.5; // 对比度
    int beta = -50;      // 亮度
    cv::Mat new_image = cv::Mat::zeros(src.size(), src.type());
    BrightnessContrastAdjuster body(src, new_image, alpha, beta);
    cv::parallel_for_(cv::Range(0, src.rows), body);
    cv::imshow("adjusted_img", new_image);
    cv::Mat gray;
    std::vector<cv::Mat> channels;
    cv::split(new_image, channels);
    gray = channels[0]; // 使用B通道
    cv::threshold(gray, binary, 180, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(binary, binary, element, cv::Point(-1, -1), 2);
    cv::dilate(binary, binary, element, cv::Point(-1, -1), 2);
}

// 灯条检测
void ArmorDetector::detectLights(const cv::Mat &binary, 
                              std::vector<std::pair<cv::RotatedRect, std::vector<cv::Point2f>>> &light_rects,
                              cv::Mat &debug_img) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    debug_img = cv::Mat::zeros(binary.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++) {
        cv::Scalar color = cv::Scalar(0, 0, 255);
        cv::drawContours(debug_img, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
        // cv::putText(debug_img, std::to_string(i), contours[i][0], 
        //           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);        
    }
    size_t i =0;
    for (const auto &contour : contours) {
        // 过滤太小的轮廓
        if (contour.size() < 5)
            continue;

        cv::RotatedRect rotatedRect = cv::minAreaRect(contour);
        cv::Point2f vertices[4];
        rotatedRect.points(vertices);

        cv::Point2f midPoint1, midPoint2;
        if (cv::norm(vertices[0] - vertices[1]) < cv::norm(vertices[1] - vertices[2])) {
            midPoint1 = (vertices[0] + vertices[1]) * 0.5f;
            midPoint2 = (vertices[2] + vertices[3]) * 0.5f;
        } else {
            midPoint1 = (vertices[1] + vertices[2]) * 0.5f;
            midPoint2 = (vertices[3] + vertices[0]) * 0.5f;
        }

        // 计算长宽比
        float width = rotatedRect.size.width;
        float height = rotatedRect.size.height;
        if (width == 0 || height == 0)
            continue;
        float aspect_ratio = std::max(width, height) / std::min(width, height);

        // 检查长宽比是否在目标范围内
        if (std::abs(aspect_ratio - target_ratio) < tolerance) {
            light_rects.emplace_back(rotatedRect, std::vector<cv::Point2f>{midPoint1, midPoint2});
        }
    }
    
    // 按x坐标排序
    std::sort(light_rects.begin(), light_rects.end(), [](const auto &a, const auto &b) {
        return a.first.center.x < b.first.center.x;
    });
    std::cout<<"count: "<<light_rects.size()<<std::endl;
    std::cout<<"light_rects: "<<std::endl;
    for (const auto &light_rect : light_rects) {
        std::cout << "center: " << light_rect.first.center << std::endl;
        std::cout << "angle: " << light_rect.first.angle << std::endl;
        std::cout << "size: " << light_rect.first.size << std::endl;
        std::cout << "points: ";
        for (const auto &point : light_rect.second) {
            std::cout << point << " ";
        }
    }
    std::cout<<std::endl;
}

// 装甲板匹配
void ArmorDetector::matchArmors(const std::vector<std::pair<cv::RotatedRect, std::vector<cv::Point2f>>> &light_rects,
                             std::vector<armor::Armor> &armors,
                             cv::Mat &debug_img) {
    for (size_t i = 0; i < light_rects.size(); ++i) {
        for (size_t j = i + 1; j < light_rects.size(); ++j) {
            std::cout<<"i: "<<i<<std::endl;
            std::cout<<"j: "<<j<<std::endl;
            cv::RotatedRect rect1 = light_rects[i].first;
            cv::RotatedRect rect2 = light_rects[j].first;

            float angle1 = rect1.angle;
            float angle2 = rect2.angle;
            std::cout<<"angle1: "<<angle1<<std::endl;
            std::cout<<"angle2: "<<angle2<<std::endl;
            // 检查灯条是否平行
            if (std::abs(angle1 - angle2) < angle_tolerance || std::abs((angle1 + angle2) - 90) < angle_tolerance) {
                std::vector<cv::Point2f> points;
                
                // 按顺时针顺序获取装甲板四个点：左下、左上、右上、右下
                if (light_rects[i].second[0].x < light_rects[j].second[0].x) {
                    points.push_back(light_rects[i].second[1]);  // 左下
                    points.push_back(light_rects[i].second[0]);  // 左上
                    std::cout<<"use i"<<std::endl;
                    std::cout<<"left down: "<<light_rects[i].second[1]<<std::endl;
                    std::cout<<"left up: "<<light_rects[i].second[0]<<std::endl;
                } else {
                    points.push_back(light_rects[j].second[1]);  // 左下
                    points.push_back(light_rects[j].second[0]);  // 左上
                    std::cout<<"use j"<<std::endl;
                    std::cout<<"left down: "<<light_rects[j].second[1]<<std::endl;
                    std::cout<<"left up: "<<light_rects[j].second[0]<<std::endl;
                }
                
                if (light_rects[i].second[1].x > light_rects[j].second[1].x) {
                    points.push_back(light_rects[i].second[0]);  // 右上
                    points.push_back(light_rects[i].second[1]);  // 右下
                    std::cout<<"use i"<<std::endl;
                    std::cout<<"right up: "<<light_rects[j].second[0]<<std::endl;
                    std::cout<<"right down: "<<light_rects[j].second[1]<<std::endl;
                } else {
                    points.push_back(light_rects[j].second[0]);  // 右上
                    points.push_back(light_rects[j].second[1]);  // 右下
                    std::cout<<"use j"<<std::endl;
                    std::cout<<"right up: "<<light_rects[i].second[0]<<std::endl;
                    std::cout<<"right down: "<<light_rects[i].second[1]<<std::endl;
                }
                
                // 计算装甲板长宽比
                float length = std::abs(std::max(cv::norm(light_rects[i].second[1] - light_rects[j].second[1]), 
                                        cv::norm(light_rects[i].second[0] - light_rects[j].second[0])));
                float width = std::abs(std::min(cv::norm((light_rects[i].second[1] - light_rects[i].second[0])), 
                                      cv::norm((light_rects[j].second[1] - light_rects[j].second[0]))));
                
                if (width == 0 || length == 0)
                    continue;
                    
                float aspect_ratio = std::max(width, length) / std::min(width, length);
                
                // 检查装甲板长宽比
                if (std::abs(aspect_ratio - armor_ratio) > armor_tolerance)
                    continue;

                // 可视化
                cv::Point2f vertices1[4], vertices2[4];
                rect1.points(vertices1);
                rect2.points(vertices2);
                
                for (const auto &point : points) {
                    std::cout<<"point: "<<point<<std::endl;
                    cv::circle(debug_img, point, 5, cv::Scalar(255, 0, 0), -1);
                }
                
                // 创建装甲板对象
                armor::Armor armor;
                armor.left_light.top = points[1];     // 左上
                armor.left_light.bottom = points[0];  // 左下
                armor.right_light.top = points[2];    // 右上
                armor.right_light.bottom = points[3]; // 右下
                armor.type = armor::ArmorType::SMALL; // 默认为小装甲板
                
                armors.push_back(armor);
                
                // 绘制连线
                cv::line(debug_img, points[0], points[2], cv::Scalar(0, 0, 255), 2);
                cv::line(debug_img, points[1], points[3], cv::Scalar(0, 0, 255), 2);
                
                // 显示点序号
                for (int k = 0; k < 4; ++k) {
                    cv::putText(debug_img, std::to_string(k), points[k], 
                              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                }
            }
        }
    }
}

// 检测装甲板
std::vector<armor::Armor> ArmorDetector::detectArmors(const cv::Mat &src, cv::Mat &debug_img) {
    std::vector<armor::Armor> armors;
    
    // 复制源图像
    cv::Mat processing_img = src.clone();
    cv::Mat binary;
    
    // 1. 预处理图像
    preprocess(processing_img, binary);
    
    // 2. 检测灯条
    std::vector<std::pair<cv::RotatedRect, std::vector<cv::Point2f>>> light_rects;
    debug_img = cv::Mat::zeros(binary.size(), CV_8UC3);
    detectLights(binary, light_rects, debug_img);
    
    // 3. 匹配装甲板
    matchArmors(light_rects, armors, debug_img);
    
    // 4. 后处理 - 合并调试图像与原图
    cv::addWeighted(src, 0.5, debug_img, 0.5, 0, debug_img);
    
    return armors;
}

// 位姿解算
void ArmorDetector::calculatePose(const std::vector<armor::Armor> &armors, 
                               std::vector<cv::Mat> &tvecs, 
                               std::vector<cv::Mat> &rvecs,
                               const cv::Mat &cameraMatrix,
                               const cv::Mat &distCoeffs) {
    for (const auto &armor : armors) {
        std::vector<cv::Point2f> imagePoints = {
            armor.left_light.bottom,   // 左下
            armor.left_light.top,      // 左上
            armor.right_light.top,     // 右上
            armor.right_light.bottom   // 右下
        };
        std::cout<< "imagePoints: "<<std::endl;
        for (const auto &point : imagePoints) {
            std::cout<<point<<std::endl;
        }
        
        cv::Mat rvec, tvec;
        cv::solvePnP(SMALL_ARMOR_POINTS, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
        
        // 旋转向量转欧拉角
        cv::Mat rotMatrix;
        cv::Rodrigues(rvec, rotMatrix);
        cv::Mat eulerAngles = rotationMatrixToEulerAngles(rotMatrix);
        
        tvecs.push_back(tvec);
        rvecs.push_back(eulerAngles);
    }
}

void ArmorDetector::adjustBrightnessContrast(const cv::Mat &src, cv::Mat &dst, double alpha, int beta) {
    dst = cv::Mat::zeros(src.size(), src.type());
    BrightnessContrastAdjuster body(src, dst, alpha, beta);
    cv::parallel_for_(cv::Range(0, src.rows), body);
}