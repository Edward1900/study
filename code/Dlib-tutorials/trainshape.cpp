// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt

#include<opencv2\opencv.hpp>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv.h>  
#include "iostream"

#include <dlib/image_processing.h>
#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/data_io.h>

using namespace dlib;
using namespace std;

frontal_face_detector face_detector = get_frontal_face_detector();

void eye_train(
	const std::string _eyes_directory,
	const std::string _file_name,
	int _scanner_wide,
	int _scanner_high,
	int _machine_core_num,
	bool _verbose,
	int _SVM_C,
	double _loss_eps
)
{
	try
	{
		//装载测试集和训练集
		dlib::array<array2d<unsigned char> > images_train, images_test;
		std::vector<std::vector<rectangle> > eye_boxes_train, eye_boxes_test;
		load_image_dataset(images_train, eye_boxes_train, _eyes_directory + "/train1.xml");
		load_image_dataset(images_test, eye_boxes_test, _eyes_directory + "/train2.xml");

		//// 图片上采样
		// upsample_image_dataset<pyramid_down<2> >(images_train, eye_boxes_train);
		// upsample_image_dataset<pyramid_down<2> >(images_test,  eye_boxes_test);

		//镜像处理图像，来数据增强
		add_image_left_right_flips(images_train, eye_boxes_train);

		//金字塔处理级数
		typedef scan_fhog_pyramid<pyramid_down < 6 > > image_scanner_type;

		// 定义一个扫面器
		image_scanner_type scanner;

		// 设置扫描器窗口大小
		scanner.set_detection_window_size(_scanner_wide, _scanner_high);

		// 定义一个训练器
		structural_object_detection_trainer<image_scanner_type> trainer(scanner);

		// 设置处理器核心
		trainer.set_num_threads(_machine_core_num);

		// 打印训练详情
		if (_verbose) trainer.be_verbose();

		// 设置训练集拟合率，增大这个数字可能会过拟合
		trainer.set_c(_SVM_C);

		// 停止迭代时的loss值，越小训练时间越长
		trainer.set_epsilon(_loss_eps);

		// 目标框不规范时要加上，能不加尽量不加，会有数据损失。
		remove_unobtainable_rectangles(trainer, images_train, eye_boxes_train);

		// 正则化
		scanner.set_nuclear_norm_regularization_strength(0.15);

		cout << "开始训练" << endl;
		object_detector<image_scanner_type> detector = trainer.train(images_train, eye_boxes_train);

		cout << "存储模型中..." << endl;
		//存储训练好的模型
		serialize(_file_name) << detector;
		cout << "存储完成！" << endl;
	}
	catch (exception & e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}


shape_predictor sp;



void eye_test(
	const std::string _eyes_directory,
	const std::string _modles_directory,
	string *models_name,
	const int _model_num
)
{
	//装在测试集和训练集
	dlib::array<array2d<unsigned char> > images_train, images_test;
	std::vector<std::vector<rectangle> > eye_boxes_train, eye_boxes_test;

	//若要在训练集上做测试请加入下面的语句，并指定正确的xml文件，并将下面在训练集上测试结果输出的语句取消注释
	//load_image_dataset(images_train, eye_boxes_train, _eyes_directory + "/training.xml");
	load_image_dataset(images_test, eye_boxes_test, _eyes_directory + "/train2.xml");

	deserialize("sp.dat") >> sp;
	//金字塔处理级数
	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;

	//装载训练好的模型
	std::vector<object_detector<image_scanner_type>> detectors;
	//detectors.push_back(face_detector);
	for (int i = 0; i < _model_num; i++)
	{
		object_detector<image_scanner_type> detector_temp;
		deserialize(_modles_directory + "/" + models_name[i]) >> detector_temp;

		//detector_temp = threshold_filter_singular_values(detector_temp, 0.1);
		detectors.push_back(detector_temp);
	}
	
	//打印精度和可视化特征图只能在单个svm模型上使用

	//打印在训练集上的测试结果，分别为precision, recall, and then average precision.
	//cout << "training results: " << test_object_detection_function(detector, images_train, eye_boxes_train) << endl;

	//打印在测试集上的测试结果，分别为precision, recall, and then average precision.
	//cout << "testing results:  " << test_object_detection_function(detectors, images_test, eye_boxes_test) << endl;

	/// You can see how many separable filters are inside your detector like so:
	//cout << "num filters: " << num_separable_filters(detector) << endl;
	
	/// You can also control how many filters there are by explicitly thresholding the
	///singular values of the filters like this:

	//detector = threshold_filter_singular_values(detector, 0.1);
	
	/// That removes filter components with singular values less than 0.1.  The bigger
	/// this number the fewer separable filters you will have and the faster the
	/// detector will run.  However, a large enough threshold will hurt detection
	/// accuracy.  
	
	
	//可视化特征图
	//image_window hogwin(draw_fhog(detector), "Learned fHOG detector");

	//// 在训练集上做预测
	//image_window win1;
	//for (unsigned long i = 0; i < images_train.size(); i++)
	//{
	//	// Run the detector and get the eye detections.
	//	std::vector<rectangle> dets = detector(images_train[i]);
	//	win1.clear_overlay();
	//	win1.set_image(images_train[i]);
	//	win1.add_overlay(dets, rgb_pixel(255, 0, 0));
	//	cout << "Hit enter to process the next image..." << endl;
	//	cin.get();
	//}

	//object_detector<image_scanner_type> detector_all(detectors);

	// 在测试集上做预测
	image_window win2;
	for (unsigned long i = 0; i < images_test.size(); ++i)
	{
		// Run the detector and get the eye detections.
		
		double t = (double)cvGetTickCount();

		std::vector<rect_detection> dets;
		//resize_image(0.75, images_test[i]);
		evaluate_detectors(detectors, images_test[i], dets, 0.1);
		
		//std::vector<rectangle> dets = detector_all(images_test[i]);
		//std::vector<rectangle> dets = evaluate_detectors(detectors, images_test[i], 0.1);
		//std::vector<rectangle> fdets = face_detector(images_test[i]);
		std::vector<rectangle> edets = detectors[0](images_test[i]);
		std::vector<full_object_detection> shapes;

		win2.clear_overlay();
		win2.set_image(images_test[i]);
		for (unsigned long j = 0; j < dets.size(); ++j)
		{
				rectangle r(edets[j].left()-5, edets[j].top()-5, edets[j].right()+5, edets[j].bottom()+5);
				auto shape = sp(images_test[i], r);
				cout << "number of parts: " << shape.num_parts() << endl;
				cout << "pixel position of first part:  " << shape.part(0) << endl;
				cout << "pixel position of second part: " << shape.part(1) << endl;
				shapes.push_back(shape);
				//win2.add_overlay(shape.part(0));
				//win2.add_overlay(shape.part(4));
				//image_window::overlay_circle c;
		}
		t = (double)cvGetTickCount() - t;
		printf("检测 = %gms\n", t / ((double)cvGetTickFrequency()*1000.));

		win2.add_overlay(shapes);
		
		std::cout << "D1:" << detectors[0].num_detectors() << std::endl;
		std::cout << "D2:" << detectors[1].num_detectors() << std::endl;
		//std::cout << "face:" << fdets.size() << std::endl;
		//if (fdets.size() > 0)
		//{
		//	win2.add_overlay(fdets[0], rgb_pixel(0, 0, 255));
		//}
		
		if (dets.size()>0)
		{
			
			for (int k = 0;k < dets.size();k++)
			{
				//win2.add_overlay(dets[k], rgb_pixel(255, 0, 0));
				
				if (dets[k].weight_index == 0)
				{
					win2.add_overlay(dets[k].rect, rgb_pixel(0, 0, 255));
				}
				else {
					win2.add_overlay(dets[k].rect, rgb_pixel(0, 255, 0));
				}
			}		
		}
		
		// win2.add_overlay(dets, rgb_pixel(255, 0, 0));
		cout << "Hit enter to process the next image..." << endl;
		cin.get();
	}
}

void eye_test_img(
	const std::string _eyes_name,
	const std::string _modles_name
)
{
	//装在测试集和训练集
	matrix<rgb_pixel> img_test,img_test2;
	
	cv::Mat src0 = cv::imread(_eyes_name);
	cv::Mat src;
	cv::resize(src0,src,cv::Size(int(src0.cols*0.75),int(src0.rows*0.75)));
	//dlib::array<array2d<unsigned char> > images_test;
	load_image(img_test, _eyes_name);
	//resize_image(0.75, img_test);

	//array2d<unsigned char> img_t(img_test);
	//若要在训练集上做测试请加入下面的语句，并指定正确的xml文件，并将下面在训练集上测试结果输出的语句取消注释

	//金字塔处理级数
	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;

	//装载训练好的模型
	std::vector<object_detector<image_scanner_type>> detectors;
	detectors.push_back(face_detector);

	object_detector<image_scanner_type> detector_temp;
	deserialize(_modles_name) >> detector_temp;

	//detector_temp = threshold_filter_singular_values(detector_temp, 0.1);
	detectors.push_back(detector_temp);


	double t = (double)cvGetTickCount();

	std::vector<rect_detection> dets;
	evaluate_detectors(detectors, img_test, dets, 0.1);

	//std::vector<rectangle> dets = detector_all(images_test[i]);
	//std::vector<rectangle> dets = evaluate_detectors(detectors, images_test[i], 0.1);
	//std::vector<rectangle> fdets = face_detector(images_test[i]);
	//std::vector<rectangle> dets = detectors[0](images_test[i]);
	t = (double)cvGetTickCount() - t;
	printf("检测 = %gms\n", t / ((double)cvGetTickFrequency()*1000.));

	std::cout << "D1:" << detectors[0].num_detectors() << std::endl;
	std::cout << "D2:" << detectors[1].num_detectors() << std::endl;

	if (dets.size()>0)
	{

		for (int k = 0;k < dets.size();k++)
		{
			//win2.add_overlay(dets[k], rgb_pixel(255, 0, 0));
			if (dets[k].weight_index == 0)
			{

				cv::rectangle(src,cv::Rect(dets[k].rect.left(), dets[k].rect.top(), dets[k].rect.width(), dets[k].rect.height()), cv::Scalar(0, 0, 255), 1, 8, 0);
			}
			else {
			
				cv::rectangle(src, cv::Rect(dets[k].rect.left(), dets[k].rect.top(), dets[k].rect.width(), dets[k].rect.height()), cv::Scalar(0, 255, 0), 1, 8, 0);
			}
		}
	}

	cv::imwrite("res.jpg",src);
	
}

void eye_test_img2(
	cv::Mat src0,
	const std::string _modles_name
)
{
	//装在测试集和训练集
	matrix<rgb_pixel> img_test, img_test2;

	//cv::Mat src0 = cv::imread(_eyes_name);

	cv::Mat src,src1;
	cv::resize(src0, src, cv::Size(int(src0.cols*0.5), int(src0.rows*0.5)));
	cv::cvtColor(src, src1, cv::COLOR_BGR2RGB);
	//dlib::array<array2d<unsigned char> > images_test;
	//load_image(img_test, _eyes_name);
	
	assign_image(img_test, cv_image<rgb_pixel>(src1));
	
	//resize_image(0.75, img_test);

	//array2d<unsigned char> img_t(img_test);
	//若要在训练集上做测试请加入下面的语句，并指定正确的xml文件，并将下面在训练集上测试结果输出的语句取消注释

	//金字塔处理级数
	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;

	//装载训练好的模型
	std::vector<object_detector<image_scanner_type>> detectors;
	detectors.push_back(face_detector);

	object_detector<image_scanner_type> detector_temp;
	deserialize(_modles_name) >> detector_temp;

	//detector_temp = threshold_filter_singular_values(detector_temp, 0.1);
	detectors.push_back(detector_temp);

	double t = (double)cvGetTickCount();

	std::vector<rect_detection> dets;
	evaluate_detectors(detectors, img_test, dets, 0.1);

	//std::vector<rectangle> dets = detector_all(images_test[i]);
	//std::vector<rectangle> dets = evaluate_detectors(detectors, images_test[i], 0.1);
	//std::vector<rectangle> fdets = face_detector(images_test[i]);
	//std::vector<rectangle> dets = detectors[0](images_test[i]);
	t = (double)cvGetTickCount() - t;
	printf("检测 = %gms\n", t / ((double)cvGetTickFrequency()*1000.));

	std::cout << "D1:" << detectors[0].num_detectors() << std::endl;
	std::cout << "D2:" << detectors[1].num_detectors() << std::endl;

	if (dets.size()>0)
	{

		for (int k = 0;k < dets.size();k++)
		{
			//win2.add_overlay(dets[k], rgb_pixel(255, 0, 0));
			if (dets[k].weight_index == 0)
			{

				cv::rectangle(src, cv::Rect(dets[k].rect.left(), dets[k].rect.top(), dets[k].rect.width(), dets[k].rect.height()), cv::Scalar(0, 0, 255), 1, 8, 0);
			}
			else {

				cv::rectangle(src, cv::Rect(dets[k].rect.left(), dets[k].rect.top(), dets[k].rect.width(), dets[k].rect.height()), cv::Scalar(0, 255, 0), 1, 8, 0);
			}
		}
	}

	cv::imshow("res", src);

}




int main_y(int argc, char** argv)
{
	cv::String rtsp_addr = "rtsp://192.168.1.12:554/stream1";
	std::string model = "eye_test3.svm";
	cv::VideoCapture cap(rtsp_addr);
	// cap.open(rtsp_addr);

	cv::Mat frame;

	for (;;) {
		cap >> frame;
		if (frame.empty())
			break;
		eye_test_img2(
			frame,
			model
		);
		//imshow("Video Stream", frame);

		if (cv::waitKey(5) == 'q')
			break;
	}
}




int main__(int argc, char** argv)
{
	try
	{
	
		const std::string faces_directory = "J:/Project/Eyes/eyes2.0/DetEyes_train/opencvproject/eye/";
		// The faces directory contains a training dataset and a separate
		// testing dataset.  The training data consists of 4 images, each
		// annotated with rectangles that bound each human face along with 68
		// face landmarks on each face.  The idea is to use this training data
		// to learn to identify the position of landmarks on human faces in new
		// images. 
		// 
		// Once you have trained a shape_predictor it is always important to
		// test it on data it wasn't trained on.  Therefore, we will also load
		// a separate testing set of 5 images.  Once we have a shape_predictor 
		// created from the training data we will see how well it works by
		// running it on the testing images. 
		// 
		// So here we create the variables that will hold our dataset.
		// images_train will hold the 4 training images and faces_train holds
		// the locations and poses of each face in the training images.  So for
		// example, the image images_train[0] has the faces given by the
		// full_object_detections in faces_train[0].
		dlib::array<array2d<unsigned char> > images_train, images_test;
		std::vector<std::vector<full_object_detection> > faces_train, faces_test;

		// Now we load the data.  These XML files list the images in each
		// dataset and also contain the positions of the face boxes and
		// landmarks (called parts in the XML file).  Obviously you can use any
		// kind of input format you like so long as you store the data into
		// images_train and faces_train.  But for convenience dlib comes with
		// tools for creating and loading XML image dataset files.  Here you see
		// how to load the data.  To create the XML files you can use the imglab
		// tool which can be found in the tools/imglab folder.  It is a simple
		// graphical tool for labeling objects in images.  To see how to use it
		// read the tools/imglab/README.txt file.
		load_image_dataset(images_train, faces_train, faces_directory + "/train2_test.xml");
		load_image_dataset(images_test, faces_test, faces_directory + "/train2_test.xml");

		// Now make the object responsible for training the model.  
		shape_predictor_trainer trainer;
		// This algorithm has a bunch of parameters you can mess with.  The
		// documentation for the shape_predictor_trainer explains all of them.
		// You should also read Kazemi's paper which explains all the parameters
		// in great detail.  However, here I'm just setting three of them
		// differently than their default values.  I'm doing this because we
		// have a very small dataset.  In particular, setting the oversampling
		// to a high amount (300) effectively boosts the training set size, so
		// that helps this example.
		trainer.set_oversampling_amount(300);
		// I'm also reducing the capacity of the model by explicitly increasing
		// the regularization (making nu smaller) and by using trees with
		// smaller depths.  
		trainer.set_nu(0.02);
		trainer.set_tree_depth(10);

		// some parts of training process can be parallelized.
		// Trainer will use this count of threads when possible
		trainer.set_num_threads(6);

		// Tell the trainer to print status messages to the console so we can
		// see how long the training will take.
		
		trainer.be_verbose();
		cout << "start" << endl;
		// Now finally generate the shape model
		shape_predictor sp = trainer.train(images_train, faces_train);
		cout << "end" << endl;

		serialize("sp.dat") << sp;
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}





int main(int argc, char** argv) try
{
	std::string model = "eye_test3.svm";
	//std::string model[2];
	//model[0] = "eye_test2.svm";
	//model[1]= "eye_test.svm";

	//eye_train(
	//	"J:/Project/Eyes/eyes2.0/DetEyes_train/opencvproject/eye/",
	//	model,
	//	150,
	//	70,
	//	6,
	//	true,
	//	1200,
	//	0.01
	//);

	
	eye_test(
		"J:/Project/Eyes/eyes2.0/DetEyes_train/opencvproject/eye/",
		"J:/Project/Eyes/eyes2.0/DetEyes_train/opencvproject/",
		&model,
		1
	);

	//eye_test_img(
	//	"1.jpg",
	//	model
	//);

	cin.get();
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}
