#include<opencv2/opencv.hpp>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>  

using namespace dlib;
using namespace std;

void init();
int FaceID(char* src_name, char*  dst_name);
std::vector<matrix<float, 0, 1>> get_src_vector(std::string path);
std::vector<matrix<float, 0, 1>> get_src_vector(std::string path);
int compare(std::string path, matrix<rgb_pixel> img, std::vector<rectangle> face);