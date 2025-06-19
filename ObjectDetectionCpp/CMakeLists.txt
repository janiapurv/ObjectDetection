cmake_minimum_required(VERSION 3.10)
project(ObjectDetectionCpp)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(ObjectDetectionCpp
    src/main.cpp
    src/object_detector.cpp
)

target_include_directories(ObjectDetectionCpp PRIVATE ${OpenCV_INCLUDE_DIRS} src)
target_link_libraries(ObjectDetectionCpp ${OpenCV_LIBS}) 