// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#include"face.h"
//using namespace cv;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------



// ----------------------------------------------------------------------------------------
frontal_face_detector detector = get_frontal_face_detector();
shape_predictor sp;
anet_type net;


void init()
{
	deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
	// And finally we load the DNN responsible for face recognition.
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

}


std::vector<matrix<float, 0, 1>> comput_vector(matrix<rgb_pixel> img, std::vector<rectangle> face)
{
	std::vector<matrix<rgb_pixel>> faces;

	auto shape = sp(img, face[0]);
	matrix<rgb_pixel> face_chip;
	extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
	faces.push_back(move(face_chip));


	if (faces.size() == 0)
	{
		cout << "No faces found in image!" << endl;

	}

	std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);

	return face_descriptors;

}

std::vector<matrix<float, 0, 1>> get_src_vector(std::string path)
{
	matrix<rgb_pixel> img;
	std::vector<matrix<float, 0, 1>> src_vector;
	//cv::Mat img = cv::imread("bald_guys.jpg",0);
	//load_image(img, path);
	cv::Mat src_img = cv::imread(path);
	assign_image(img, cv_image<rgb_pixel>(src_img));

	std::vector<rectangle> face = detector(img);

	src_vector = comput_vector(img, face);

	return src_vector;
}

int compare(std::string path, matrix<rgb_pixel> img, std::vector<rectangle> face)
{
	int flag = 0;
//	double t1 = (double)cvGetTickCount();
	std::vector<matrix<float, 0, 1>> src_vector = get_src_vector(path);
//	t1 = (double)cvGetTickCount() - t1;
//	printf("get_src所用时间 = %gms\n", t1 / ((double)cvGetTickFrequency()*1000.));

	double t2 = (double)cvGetTickCount();
	std::vector<matrix<float, 0, 1>> face_vector = comput_vector(img, face);
//	t2 = (double)cvGetTickCount() - t2;
//	printf("comput_vector所用时间 = %gms\n", t2 / ((double)cvGetTickFrequency()*1000.));
	if (length(src_vector[0] - face_vector[0]) < 0.6)
	{
		flag = 1;
	}
	t2 = (double)cvGetTickCount() - t2;
	printf("comput_vector所用时间 = %gms\n", t2 / ((double)cvGetTickFrequency()*1000.));

	return flag;
}

int align(matrix<rgb_pixel> img, matrix<rgb_pixel>& aface)
{
	std::vector<rectangle> face = detector(img);
	auto shape = sp(img, face[0]);
	matrix<rgb_pixel> face_chip;
	extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);

	aface = move(face_chip);
	if (face.size() > 0)
	{
		return 1;
	}
	else {
		return 0;
	}
}

std::vector<matrix<float, 0, 1>> get_src_vectors(std::string path, std::string path2)
{
	matrix<rgb_pixel> img, aimg;
	matrix<rgb_pixel> img2, aimg2;
	std::vector<matrix<float, 0, 1>> src_vector;
	//cv::Mat img = cv::imread("bald_guys.jpg",0);
	load_image(img, path);
	load_image(img2, path2);


	std::vector<matrix<rgb_pixel>> faces;

	int f1 = align(img, aimg);int f2 = align(img2, aimg2);
	faces.push_back(aimg);faces.push_back(aimg2);

	if ((f1 == 1) && (f2 == 1))
	{
		src_vector = net(faces);
	}
	else {
		cout << "no face." << endl;

	}

	return src_vector;
}


int SaveDat(std::string path1, std::string path2) //path1为img路径，path2为dat路径
{
	//serialize("metric_network_renset.dat") << net;
	matrix<rgb_pixel> img;
	std::vector<matrix<float, 0, 1>> src_vector;
	//cv::Mat img = cv::imread("bald_guys.jpg",0);
	load_image(img, path1);
	std::vector<rectangle> face = detector(img);
	if (face.size() > 0)
	{
		src_vector = comput_vector(img, face);
		serialize(path2) << src_vector[0];
		return 1;
	}
	else {
		cout << "no face." << endl;
		return 0;

	}


}
int CompareDat(std::string path1, std::string path2)//path1-img ;path2-dat
{
	int flag = 0;

	std::vector<matrix<float, 0, 1>> src_vector1 = get_src_vector(path1);
	std::vector<matrix<float, 0, 1>> src_vector2;// = get_src_vector(path2);
	matrix<float, 0, 1> dat1;
	deserialize(path2) >> dat1;
	src_vector2.push_back(dat1);

	if ((src_vector1.size() > 0) && (src_vector2.size() > 0))
	{
		if (length(src_vector1[0] - src_vector2[0]) < 0.6)
		{
			flag = 1;
		}
	}
	else { flag = -1; }

	return flag;
}

int CompareImg(std::string path1, std::string path2)
{
	int flag = 0;

	std::vector<matrix<float, 0, 1>> src_vector1 = get_src_vector(path1);
	std::vector<matrix<float, 0, 1>> src_vector2 = get_src_vector(path2);


	if ((src_vector1.size() > 0) && (src_vector2.size() > 0))
	{
		if (length(src_vector1[0] - src_vector2[0]) < 0.6)
		{
			flag = 1;
		}
	}
	else { flag = -1; }

	return flag;
}


extern "C" __declspec(dllexport) int GetDat(char* src_path, char*  dst_path)
{
	std::string path1, path2;
	path1.assign(src_path);
	path2.assign(dst_path);

	int f = SaveDat(path1, path2);

	return f;
}

extern "C" __declspec(dllexport) int FaceIDD(char* img_path, char*  dat_path)
{
	std::string path1, path2;
	path1.assign(img_path);
	path2.assign(dat_path);
	int f = CompareDat(path1, path2);

	return f;
}




extern "C" __declspec(dllexport) int FaceID(char* src_name, char*  dst_name)
{
	string src, dst;
	int f = 0;
	src.assign(src_name);
	dst.assign(dst_name);
	

	matrix<rgb_pixel> img;
	matrix<rgb_pixel> img2;

	load_image(img, dst);
	cv::Mat src_img = cv::imread(dst);

	std::vector<rectangle> face = detector(img);

	if (face.size()>0)
	{
		f = compare(src, img, face);
		cv::rectangle(src_img, cv::Rect(face[0].left(), face[0].top(), face[0].width(), face[0].height()), cv::Scalar(0, 0, 255), 2, 8, 0);
		
	}
	else
	{
		f = -1;
	}
	 
	cv::imwrite("dst.jpg", src_img);
	return f;

}

extern "C" __declspec(dllexport) int FaceID1(char* src_name, uchar* Ddata, int rows, int cols)
{
	string src;
	int f = 0;
	src.assign(src_name);
	
	cv::Mat img_cv(rows, cols, 16);
	img_cv.data = Ddata;
	

	matrix<rgb_pixel> img;
	matrix<rgb_pixel> img2;

	assign_image(img, cv_image<rgb_pixel>(img_cv));
	
	
	std::vector<rectangle> face = detector(img);

	if (face.size() > 0)
	{
		f = compare(src, img, face);
		cv::rectangle(img_cv, cv::Rect(face[0].left(), face[0].top(), face[0].width(), face[0].height()), cv::Scalar(0, 0, 255), 2, 8, 0);
		
	}
	else
	{
		f = -1;
	}

	cv::imwrite("dst.jpg", img_cv);
	return f;

}




int saveimg(uchar* datap,int rows,int cols)
{
	cv::Mat src(rows, cols, 16);
	src.data = datap;

	cv::imwrite("dst22.jpg", src);
	return 0;
}


