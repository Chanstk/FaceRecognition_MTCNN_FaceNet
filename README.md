# Face recognition system
## Introduction
This repository is a face recognition system. The face detection part is the implementation of MTCNN algorithm and the recognition part using tensorflow c++ api is the implementation of facenet algorithm.
## Dependencies
1. opencv3
2. openblas
3. tensorflow(C++ shared library) 
## Usage(Demo)
1. cd root_directory
2. vim CMakeLists.txt
3. change the path of tensorflow/opencv/openblas according to your machine
4. Download the ["Facenet model"](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) and put it in the model/
5. cmake . ( or cmake -DCMAKE_BUILD_TYPE = RELEASE . if you care about the speed performance)
6. make
7. ./main xxxx.jpg "yourname" (xxx.jpg must be a photo of your face) 
## Inspiration
1. Facenet part : the model file is taken from [here](https://github.com/davidsandberg/facenet)
2. MTCNN part : the code is taken from [here](https://github.com/AlphaQi/MTCNN-light) 

