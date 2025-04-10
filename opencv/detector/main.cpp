#include <fstream>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat& read(const std::string& filename) {
    static cv::Mat img;
    img = cv::imread(filename);
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
    }
    return img;
}
/**
 * @brief 创建一个窗口和滑块
 * 
 * @param window_name 窗口名
 * @param bar_name 滑块名
 * @param value 值，int类型
 * @param max_value 最大值
 */
void createBarwithWindow(const std::string& window_name, const std::string& bar_name, int* value, int max_value) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::createTrackbar(bar_name, window_name, value, max_value);
    cv::setTrackbarPos(bar_name, window_name, *value);
}

/**
 * @brief 创建一个滑块
 * 
 * @param window_name 窗口名
 * @param bar_name 滑块名
 * @param value 值，int类型
 * @param max_value 最大值
 */
void createBaronWindow(const std::string& window_name, const std::string& bar_name, int* value, int max_value) {
    cv::createTrackbar(bar_name, window_name, value, max_value);
    cv::setTrackbarPos(bar_name, window_name, *value);
}

double getDistance(cv::Point p1, cv::Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

cv::Mat preprocess(
    cv::Mat image,
    int image_scale_100 = 100,
    int lower_blue_h = 100,
    int lower_blue_s = 50,
    int lower_blue_v = 50,
    int upper_blue_h = 140,
    int upper_blue_s = 255,
    int upper_blue_v = 255,
    int op_kernel_size_h = 5, // 横
    int op_kernel_size_v = 5, // 竖
    int cl_kernel_size_h = 5,
    int cl_kernel_size_v = 5,
    int fliter = 0,
    int actibvate_binary = 0,
    int binary_threshold = 0
) {
    if (image_scale_100 < 1) {
        image_scale_100 = 1;
    }
    if (op_kernel_size_h < 1) {
        op_kernel_size_h = 1;
    }
    if (op_kernel_size_v < 1) {
        op_kernel_size_v = 1;
    }
    if (cl_kernel_size_h < 1) {
        cl_kernel_size_h = 1;
    }
    if (cl_kernel_size_v < 1) {
        cl_kernel_size_v = 1;
    }

    double image_scale_ = image_scale_100 / 100.0;

    // 去畸变(相机没标)
    // cv::undistort(image, image, cameraMatrix_, distCoeffs_);
    // 获取图像长宽
    double image_width_ = image.cols;
    double image_height_ = image.rows;

    // 缩放图像
    cv::resize(image, image, cv::Size(image_width_ * image_scale_, image_height_ * image_scale_));

    cv::Mat mask;

    if (actibvate_binary == 1) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        mask = channels[0];
        cv::imshow("gray", mask);
        cv::threshold(mask, mask, binary_threshold, 255, cv::THRESH_BINARY);

    } else if (actibvate_binary == 2) {
        cv::cvtColor(image, mask, cv::COLOR_BGR2GRAY);
        cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::imshow("Otsu Threshold", mask);
    }

    else
    {
        // 转换到HSV色彩空间
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        // 定义蓝HSV范围

        cv::Scalar lower_blue(lower_blue_h, lower_blue_s, lower_blue_v);
        cv::Scalar upper_blue(upper_blue_h, upper_blue_s, upper_blue_v);
        cv::inRange(hsv, lower_blue, upper_blue, mask);
    }

    if (fliter == 1) {
        // 中值滤波
        cv::medianBlur(mask, mask, op_kernel_size_h);
        cv::imshow("medianBlur", mask);
    } else if (fliter == 2) {
        // 高斯滤波
        cv::GaussianBlur(mask, mask, cv::Size(op_kernel_size_h, op_kernel_size_v), 0);
        cv::imshow("GaussianBlur", mask);
    } else if (fliter == 3) {
        // 双边滤波
        cv::bilateralFilter(mask, mask, 9, 75, 75);
        cv::imshow("bilateralFilter", mask);
    }

    // 开闭运算
    cv::Mat op_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(op_kernel_size_h, op_kernel_size_v));
    cv::Mat cl_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(cl_kernel_size_h, cl_kernel_size_v));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, op_kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cl_kernel);

    cv::imshow("mask", mask);

    image = mask;

    return image;
}

// 轮廓检测与筛选
std::vector<std::vector<cv::Point>> detect_contours(
    cv::Mat& preprocessed_image,
    int canny_threshold1,
    int canny_threshold2,
    int canny_apertureSize,
    bool canny_L2gradient,
    int findContours_mode,
    int findContours_method,
    double approxPolyDP_epsilon,
    double min_area,
    double max_area,
    double ratio,
    double min_edges,
    double max_edges,
    int min_distance
) {
    if (preprocessed_image.empty()) {
        return std::vector<std::vector<cv::Point>>();
    }

    if (canny_apertureSize % 2 == 0) {
        canny_apertureSize += 1; // 确保为奇数
    }
    if (canny_apertureSize < 3) {
        canny_apertureSize = 3; // 最小值
    }
    if (canny_apertureSize > 7) {
        canny_apertureSize = 7; // 最大值
    }

    // std::cout << canny_threshold1 << " " << canny_threshold2 << " " << canny_apertureSize << " " << canny_L2gradient << std::endl;

    // 输入格式化
    if (ratio < 1) {
        ratio = 1 / ratio;
    }

    if (min_area == -1) {
        min_area = 0;
    }
    if (min_area == -1) {
        min_area = 0;
    }

    if (max_area == -1) {
        max_area = preprocessed_image.cols * preprocessed_image.rows;
    }

    if (max_edges == -1) {
        max_edges = 1024;
    }

    // 边缘检测
    cv::Mat edges;
    cv::Canny(preprocessed_image, edges, canny_threshold1, canny_threshold2, canny_apertureSize, canny_L2gradient);

    cv::Mat debug_image = edges.clone();
    cv::imshow("edges", debug_image);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, findContours_mode, findContours_method);

    // 过滤掉不符合要求的轮廓
    contours.erase(
        std::remove_if(
            contours.begin(),
            contours.end(),
            [approxPolyDP_epsilon, min_area, max_area, ratio, min_edges, max_edges](std::vector<cv::Point>& contour) {
                double contour_area = cv::contourArea(contour);

                cv::RotatedRect rect = cv::minAreaRect(contour);
                double contour_ratio = rect.size.width / rect.size.height;

                std::vector<cv::Point> approxCurve;
                cv::approxPolyDP(contour, approxCurve, approxPolyDP_epsilon, true);
                double contour_edges = approxCurve.size();
                return contour_area < min_area || contour_area > max_area      // 轮廓面积过小或过大
                    || contour_ratio < 1 / ratio || contour_ratio > ratio      // 长宽比过大
                    || contour_edges < min_edges || contour_edges > max_edges; // 轮廓边数过少或过多
            }
        ),
        contours.end()
    );

    // 删除邻近轮廓
    for (size_t i = 0; i < contours.size(); i++) {
        for (size_t j = i + 1; j < contours.size(); j++) {
            if (getDistance(cv::minAreaRect(contours[i]).center, cv::minAreaRect(contours[j]).center) < min_distance) {
                if (cv::contourArea(contours[i]) > cv::contourArea(contours[j])) {
                    contours.erase(contours.begin() + j);
                    j--;
                } else {
                    contours.erase(contours.begin() + i);
                    i--;
                    break;
                }
            }
        }
    }
    return contours;
}

void sortPoints(std::vector<cv::Point>& points) {
    // 计算凸包
    std::vector<int> hullIndices;
    cv::convexHull(points, hullIndices, false, true); // 返回索引，按顺序排列

    // 检查凸包方向
    std::vector<cv::Point> hullPoints;
    for (int idx: hullIndices) {
        hullPoints.push_back(points[idx]);
    }

    bool isConvex = cv::isContourConvex(hullPoints);

    // 如果凸包不是逆时针方向，则反转
    if (isConvex) {
        std::reverse(hullPoints.begin(), hullPoints.end());
    }

    // 重新排序原始点集
    points = hullPoints;
}

// L型轮廓检测
std::vector<cv::Point> L_detect(
    std::vector<cv::Point>& corner_points,
    bool* detected,
    cv::Mat* image = nullptr,
    double eps = 0.01,
    int* corner_points_detected = nullptr
) {
    *detected = false;
    std::vector<cv::Point> detected_points;

    if (image->type() == CV_8UC1) {
        cv::cvtColor(*image, *image, cv::COLOR_GRAY2BGR);
    }

    // 使用Contour Approximation， 尝试收缩至6个点
    std::vector<cv::Point> approxCurve;
    cv::approxPolyDP(corner_points, approxCurve, cv::arcLength(corner_points, true) * eps, true);

    // 如果未检测到6个点
    if (approxCurve.size() != 6) {
        *corner_points_detected = approxCurve.size();
        return approxCurve;
    }

    std::vector<cv::Point> approxCurve_1;
    approxCurve_1 = approxCurve;
    sortPoints(approxCurve);
    detected_points = approxCurve;

    // 找到最小方差的点，将其作为起始点
    double min_variance = std::numeric_limits<double>::max();
    size_t min_variance_index = 0;

    for (size_t i = 0; i < detected_points.size(); i++) {
        std::vector<double> distances;
        // 计算每个点到其他点的距离
        for (size_t j = 0; j < detected_points.size(); j++) {
            if (i != j) {
                distances.push_back(getDistance(detected_points[i], detected_points[j]));
            }
        }

        // 计算方差
        double sum = 0.0;
        for (double dist: distances) {
            sum += dist;
        }
        double mean = sum / distances.size();

        double variance = 0.0;
        for (double dist: distances) {
            variance += (dist - mean) * (dist - mean);
        }
        variance /= distances.size();
        variance /= mean; // 归一化

        if (variance < min_variance) {
            min_variance = variance;
            min_variance_index = i;
        }
    }

    // 旋转数组，使得最小方差的点作为起始点
    if (min_variance_index > 0) {
        std::rotate(detected_points.begin(), detected_points.begin() + min_variance_index, detected_points.end());
    }

    // 加入剩下的点在[3]的位置
    for (cv::Point point: approxCurve_1) {
        if (std::find(detected_points.begin(), detected_points.end(), point) != detected_points.end()) {
            continue;
        } else {
            detected_points.insert(detected_points.begin() + 3, point);
            break;
        }
    }

    for (size_t i = 0; i < detected_points.size(); i++) {
        cv::circle(*image, detected_points[i], 5, cv::Scalar(0, 0, 255), -1);
        cv::line(
            *image,
            detected_points[i],
            detected_points[(i + 1) % detected_points.size()],
            cv::Scalar(255, 0, 0),
            2
        );
        cv::putText(
            *image,
            std::to_string(i),
            detected_points[i],
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(255, 0, 0),
            2
        );
    }

    cv::imshow("L_contour_image", *image);

    *detected = true;

    return detected_points;
}

std::vector<cv::Point> modified_L_detect(
    std::vector<cv::Point>& corner_points,
    bool* detected,
    cv::Mat* image = nullptr,
    double eps = 0.013,
    int* corner_points_detected = nullptr,
    double min_variance = 0.001,
    double max_variance = 0.150,
    int max_iterations = 10,
    int iter = 0
) {
    *detected = false;

    // 尝试一次检测
    std::vector<cv::Point> detected_points;
    detected_points = L_detect(corner_points, detected, image, eps, corner_points_detected);

    // 超过最大迭代次数
    if (++iter > max_iterations) {
        for (size_t i = 0; i < detected_points.size(); i++) {
            cv::circle(*image, detected_points[i], 5, cv::Scalar(0, 0, 255), -1);
            cv::line(
                *image,
                detected_points[i],
                detected_points[(i + 1) % detected_points.size()],
                cv::Scalar(255, 0, 0),
                2
            );
            cv::putText(
                *image,
                std::to_string(i),
                detected_points[i],
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                cv::Scalar(255, 0, 0),
                2
            );
        }
        return std::vector<cv::Point>();
    }

    if (*detected) {
        return detected_points;
    }
    // 检测失败，使用二分法调整eps
    else
    {
        if (*corner_points_detected > 6) {
            eps = (max_variance + eps) / 2;
        } else {
            eps = (min_variance + eps) / 2;
        }

        return modified_L_detect(
            corner_points,
            detected,
            image,
            eps,
            corner_points_detected,
            min_variance,
            max_variance,
            max_iterations,
            iter
        );
    }
}

std::vector<std::vector<cv::Point>> predetect(
    cv::Mat& preprocessed_image,
    double eps,
    int canny_threshold1,
    int canny_threshold2,
    int canny_apertureSize,
    int canny_L2gradient ,
    int findContours_mode,
    int findContours_method,
    double approxPolyDP_epsilon,
    int min_area ,
    int max_area ,
    int ratio_100,
    int min_edges,
    int max_edges,
    int min_distance
) {
    if (preprocessed_image.empty()) {
        return std::vector<std::vector<cv::Point>>();
    }

    std::vector<cv::Point> detected_points;

    std::vector<std::vector<cv::Point>> contours = detect_contours(
        preprocessed_image,
        canny_threshold1,
        canny_threshold2,
        canny_apertureSize,
        canny_L2gradient,
        findContours_mode,
        findContours_method,
        approxPolyDP_epsilon,
        min_area,
        max_area,
        ratio_100,
        min_edges,
        max_edges,
        min_distance
    );

    cv::Mat debug_image = preprocessed_image.clone();
    cv::cvtColor(debug_image, debug_image, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(debug_image, contours, i, cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("contour_image", debug_image);

    if (contours.size() == 0) {
        return contours;
    }

    // 按照轮廓面积大小排序
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
        return cv::contourArea(c1) > cv::contourArea(c2);
    });

    std::vector<std::vector<cv::Point>> detected_Ls;
    std::vector<cv::Point> L;
    bool is_L = false;
    for (size_t i = 0; i < contours.size(); i++) {
        is_L = false;
        int corner_points_detected = 0;
        L = modified_L_detect(contours[i], &is_L, &debug_image, eps, &corner_points_detected);
        if (is_L) {
            detected_Ls.push_back(L);
        }
    }

    if (detected_Ls.size() == 0) {
        return detected_Ls;
    }

    return detected_Ls;
}

int main() {
    // 参数
    int image_scale_100 = 100;
    int lower_blue_h = 100;
    int lower_blue_s = 50;
    int lower_blue_v = 50;
    int upper_blue_h = 140;
    int upper_blue_s = 255;
    int upper_blue_v = 255;
    int op_kernel_size_h = 5; // 横
    int op_kernel_size_v = 5; // 竖
    int cl_kernel_size_h = 5;
    int cl_kernel_size_v = 5;

    int canny_threshold1 = 100;
    int canny_threshold2 = 200;
    int canny_apertureSize = 3;
    int canny_L2gradient = 0;
    int findContours_mode = cv::RETR_EXTERNAL;
    int findContours_method = cv::CHAIN_APPROX_SIMPLE;
    int approxPolyDP_epsilon_100 = 170;
    int min_area = 200;
    int max_area = 1000000;
    int ratio_100 = 600;
    int min_edges = 5;
    int max_edges = 100;
    int min_distance = 100;

    int eps_1000 = 13;

    int fliter = 0;           // 0:无滤波 1:中值滤波 2:高斯滤波 3:双边滤波
    int actibvate_binary = 0; // 0:无二值化 1:二值化
    int binary_threshold = 0; // 二值化阈值

    // 创建滑块
    createBarwithWindow("preprocess", "image_scale", &image_scale_100, 1000);
    createBaronWindow("preprocess", "lower_blue_h", &lower_blue_h, 255);
    createBaronWindow("preprocess", "lower_blue_s", &lower_blue_s, 255);
    createBaronWindow("preprocess", "lower_blue_v", &lower_blue_v, 255);
    createBaronWindow("preprocess", "upper_blue_h", &upper_blue_h, 255);
    createBaronWindow("preprocess", "upper_blue_s", &upper_blue_s, 255);
    createBaronWindow("preprocess", "upper_blue_v", &upper_blue_v, 255);
    createBaronWindow("preprocess", "op_kernel_size_h", &op_kernel_size_h, 20);
    createBaronWindow("preprocess", "op_kernel_size_v", &op_kernel_size_v, 20);
    createBaronWindow("preprocess", "cl_kernel_size_h", &cl_kernel_size_h, 20);
    createBaronWindow("preprocess", "cl_kernel_size_v", &cl_kernel_size_v, 20);
    createBaronWindow("preprocess", "fliter", &fliter, 3);
    createBaronWindow("preprocess", "actibvate_binary", &actibvate_binary, 2);
    createBaronWindow("preprocess", "binary_threshold", &binary_threshold, 255);

    createBarwithWindow("detect_contours", "canny_threshold1", &canny_threshold1, 255);
    createBaronWindow("detect_contours", "canny_threshold2", &canny_threshold2, 255);
    createBaronWindow("detect_contours", "canny_apertureSize", &canny_apertureSize, 7);
    createBaronWindow("detect_contours", "canny_L2gradient", &canny_L2gradient, 1);
    createBaronWindow("detect_contours", "approxPolyDP_epsilon_100", &approxPolyDP_epsilon_100, 1000);
    createBaronWindow("detect_contours", "min_area", &min_area, 10000);
    createBaronWindow("detect_contours", "max_area", &max_area, 1000000);
    createBaronWindow("detect_contours", "ratio_100", &ratio_100, 1000);
    createBaronWindow("detect_contours", "min_edges", &min_edges, 100);
    createBaronWindow("detect_contours", "max_edges", &max_edges, 100);
    createBaronWindow("detect_contours", "min_distance", &min_distance, 1000);

    // 从文件读取图像名称
    std::string filename = "../img_name.txt";
    std::ifstream fin;
    fin.open(filename);

    if (!fin.is_open()) {
        std::cerr << "Error: Could not open the file!" << std::endl;
        return -1;
    }

    std::string img_name;
    fin >> img_name;
    fin.close();

    while (1) {
        cv::Mat src_img = read(img_name);
        cv::imshow("src_img", src_img);

        // 预处理
        cv::Mat preprocessed_image = preprocess(
            src_img.clone(),
            image_scale_100,
            lower_blue_h,
            lower_blue_s,
            lower_blue_v,
            upper_blue_h,
            upper_blue_s,
            upper_blue_v,
            op_kernel_size_h,
            op_kernel_size_v,
            cl_kernel_size_h,
            cl_kernel_size_v,
            fliter = 0,
            actibvate_binary,
            binary_threshold
        );
        std::vector<std::vector<cv::Point>> Ls = predetect(
            preprocessed_image,
            eps_1000 / 1000.0,
            canny_threshold1,
            canny_threshold2,
            canny_apertureSize,
            canny_L2gradient,
            findContours_mode,
            findContours_method,
            approxPolyDP_epsilon_100 / 100.0,
            min_area,
            max_area,
            ratio_100 / 100.0,
            min_edges,
            max_edges,
            min_distance
        );

        cv::waitKey(1);
    }

    return 0;
}