cmake_minimum_required(VERSION 3.15)
project(Ch04)

set(CMAKE_CXX_STANDARD 11)
set(Opencv_DIR "/home/zeng/ide/opencv-4.2.0/build")
set(Eigen_DIR "/home/zeng/ide/eigen/build")
set(Pangolin_DIR "/home/zeng/ide/Pangolin-master/build")
set(Ceres_DIR "/home/zeng/ide/ceres-solver-master/build")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_DIR})
include_directories(${Pangolin_DIR})
include_directories(${CERES_INCLUDE_DIRS})

add_executable(Ch04 curvefitting.cpp)
target_link_libraries(Ch04 ${OpenCV_LIBS})
target_link_libraries(Ch04 ${Pangolin_LIBRARIES})
target_link_libraries (Ch04 Eigen3::Eigen)
target_link_libraries(Ch04 ${CERES_LIBRARIES})