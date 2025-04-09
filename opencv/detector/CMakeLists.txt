cmake_minimum_required(VERSION 3.16)

project(cv)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# 默认启用编译警告
add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_options(-Og)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_DIRS})

add_executable(cv main.cpp)

target_link_libraries(cv ${OpenCV_LIBS})
