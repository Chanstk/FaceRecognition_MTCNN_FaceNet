#face recognition system
##Introduction
This repository is a face recognition system. The face detection part is the implementation of MTCNN algorithm and the recognition part using tensorflow c++ api is the implementation of facenet algorithm.
## Dependencies
opencv3
openblas
tensorflow(C++ lib)
## usage
1. cd root_directory
2.vim CMakeLists.txt
3.change the path of tensorflow/opencv/openblas according to your machine
4.Download the ["facenet model"]() and put it in the model/
5.cmake . ( or cmake -DCMAKE_BUILD_TYPE = RELEASE . if you care about the speed)
6. make
7. ./main

