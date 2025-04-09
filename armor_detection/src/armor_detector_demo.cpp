#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/armor_detector.hpp"

int main(int argc, char** argv) {
    // 创建装甲板检测器实例
    ArmorDetector detector;

    // 相机参数
    // clang-format off
    cv::Mat cameraMatrix =
        (cv::Mat_<double>(3, 3) << 2065.0580175762857,  0.0,               658.9098266395495,
                                   0.0,                 2086.886458338243, 531.5333174739342,
                                   0.0,                 0.0,               1.0);
    cv::Mat distCoeffs =
        (cv::Mat_<double>(5, 1) << -0.051836613762195866,
         0.29341513924119095,
         0.001501183796729562,
         0.0009386915104617738,
         0.0);
    // clang-format on
    cv::Mat frame;
    frame = cv::imread("../armors/9.jpg");
    if (frame.empty()) {
        std::cerr << "无法读取图像！" << std::endl;
        return -1;
    }

    // 创建调试图像
    cv::Mat debug_img;

    // 检测装甲板
    std::vector<armor::Armor> armors = detector.detectArmors(frame, debug_img);

    // 计算位姿
    std::vector<cv::Mat> tvecs, rvecs;
    if (!armors.empty()) {
        detector.calculatePose(armors, tvecs, rvecs, cameraMatrix, distCoeffs);

        // 显示位姿信息
        for (size_t i = 0; i < tvecs.size(); ++i) {
            std::cout << "装甲板 " << i << " 位置："
                      << "X: " << tvecs[i].at<double>(0) << ", Y: " << tvecs[i].at<double>(1)
                      << ", Z: " << tvecs[i].at<double>(2) << "m" << std::endl;

            std::cout << "装甲板 " << i << " 旋转(欧拉角)："
                      << "X: " << rvecs[i].at<double>(0) * 180 / CV_PI
                      << ", Y: " << rvecs[i].at<double>(1) * 180 / CV_PI
                      << ", Z: " << rvecs[i].at<double>(2) * 180 / CV_PI << "度" << std::endl;

            // 在图像上显示距离信息
            std::string dist_text = "Dist: " + std::to_string(sqrt(pow(tvecs[i].at<double>(0),2) + pow(tvecs[i].at<double>(1),2)+pow(tvecs[i].at<double>(2),2)) ) + "m";
            cv::putText(
                debug_img,
                dist_text,
                armors[i].left_light.top - cv::Point2f(0, 20),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 255, 255),
                1
            );
        }
    }

    // 显示结果
    cv::imshow("装甲板检测", debug_img);

    // 按ESC键退出
    if (cv::waitKey(0) == 27)
        cv::destroyAllWindows();

    return 0;
}