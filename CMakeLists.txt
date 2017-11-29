cmake_minimum_required(VERSION 2.8)
project(main)

set(QMAKE_CXXFLAGS "-std=c++11 -O3  -Wno-invalid-partial-specialization -Wno-sign-compare -w  -D_THREAD_SAFE -I/usr/local/include -L/usr/local/lib -lprotobuf -D_THREAD_SAFE")
SET( CMAKE_CXX_FLAGS "-std=c++11 -O3  -Wno-invalid-partial-specialization -Wno-sign-compare -w  -D_THREAD_SAFE -I/usr/local/include -L/usr/local/lib -lprotobuf -D_THREAD_SAFE")
include_directories(/usr/local/opt/openblas/include 
	/usr/local/Cellar/opencv/3.3.1_1/include
/usr/local/include/tf
/usr/local/include/eigen3)
link_directories(/usr/local/opt/openblas/lib /usr/local/Cellar/opencv/3.3.1_1/lib)
#find_package( OpenCV REQUIRED)
#message(STATUS "${OpenCV_INCLUDE_DIRS}") 
AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable( main ${DIR_SRCS})
target_link_libraries( main opencv_videoio opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs )
target_link_libraries( main openblas)
target_link_libraries( main tensorflow_cc) 
