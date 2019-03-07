#include<opencv2\opencv.hpp>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>  

using namespace dlib;
using namespace std;

extern "C" __declspec(dllexport) void init();
extern "C" __declspec(dllexport) int FaceID(char* src_name, char*  dst_name);
