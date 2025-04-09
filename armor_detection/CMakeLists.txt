cmake_minimum_required(VERSION 3.10)
project(armor_detection)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找OpenCV包
find_package(OpenCV REQUIRED)
find_package(OpenMP)

# 添加包含目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 设置源文件
set(SOURCES
    src/armor_detector.cpp
)

# 添加库和可执行文件
add_library(armor_detector ${SOURCES})
add_executable(armor_detector_demo src/armor_detector_demo.cpp)

# 链接库
target_link_libraries(armor_detector ${OpenCV_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(armor_detector OpenMP::OpenMP_CXX)
endif()

target_link_libraries(armor_detector_demo armor_detector ${OpenCV_LIBS})

# 设置安装规则
install(TARGETS armor_detector armor_detector_demo
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

# 将头文件安装到include目录
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp")

# 复制图像文件到安装目录
install(DIRECTORY armors/
    DESTINATION share/armor_detection/armors
    FILES_MATCHING PATTERN "*.jpg")