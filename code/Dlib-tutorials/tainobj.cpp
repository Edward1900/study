// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt

#include<opencv2\opencv.hpp>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "iostream"

#include <dlib/image_processing.h>
#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/data_io.h>

using namespace dlib;
using namespace std;

frontal_face_detector face_detector = get_frontal_face_detector();

void obj_train(
	const std::string _objs_directory,
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
		std::vector<std::vector<rectangle> > obj_boxes_train, obj_boxes_test;
		load_image_dataset(images_train, obj_boxes_train, _objs_directory + "/training.xml");
		load_image_dataset(images_test, obj_boxes_test, _objs_directory + "/testing.xml");

		// 图片上采样
		// upsample_image_dataset<pyramid_down<2> >(images_train, obj_boxes_train);
		// upsample_image_dataset<pyramid_down<2> >(images_test,  obj_boxes_test);

		//镜像处理图像，来数据增强
		// add_image_left_right_flips(images_train, obj_boxes_train);

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
		remove_unobtainable_rectangles(trainer, images_train, obj_boxes_train);

		// 正则化
		// scanner.set_nuclear_norm_regularization_strength(1.0);

		cout << "开始训练" << endl;
		object_detector<image_scanner_type> detector = trainer.train(images_train, obj_boxes_train);

		
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


void obj_test(
	const std::string _objs_directory,
	const std::string _modles_directory,
	string *models_name,
	const int _model_num
)
{
	//装在测试集和训练集
	dlib::array<array2d<unsigned char> > images_train, images_test;
	std::vector<std::vector<rectangle> > obj_boxes_train, obj_boxes_test;

	//若要在训练集上做测试请加入下面的语句，并指定正确的xml文件，并将下面在训练集上测试结果输出的语句取消注释
	//load_image_dataset(images_train, obj_boxes_train, _objs_directory + "/training.xml");
	load_image_dataset(images_test, obj_boxes_test, _objs_directory + "/testing.xml");

	//金字塔处理级数
	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;

	//装载训练好的模型
	std::vector<object_detector<image_scanner_type>> detectors;
	detectors.push_back(face_detector);
	for (int i = 0; i < _model_num; i++)
	{
		object_detector<image_scanner_type> detector_temp;
		deserialize(_modles_directory + "/" + models_name[i]) >> detector_temp;

		//detector_temp = threshold_filter_singular_values(detector_temp, 0.1);
		detectors.push_back(detector_temp);
	}
	
	//打印精度和可视化特征图只能在单个svm模型上使用

	//打印在训练集上的测试结果，分别为precision, recall, and then average precision.
	//cout << "training results: " << test_object_detection_function(detector, images_train, obj_boxes_train) << endl;

	//打印在测试集上的测试结果，分别为precision, recall, and then average precision.
	//cout << "testing results:  " << test_object_detection_function(detectors, images_test, obj_boxes_test) << endl;

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
	//	// Run the detector and get the obj detections.
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
		// Run the detector and get the obj detections.
		
		double t = (double)cvGetTickCount();

		std::vector<rect_detection> dets;
		evaluate_detectors(detectors, images_test[i], dets, 0.1);

		//std::vector<rectangle> dets = detector_all(images_test[i]);
		//std::vector<rectangle> dets = evaluate_detectors(detectors, images_test[i], 0.1);
		//std::vector<rectangle> fdets = face_detector(images_test[i]);
		//std::vector<rectangle> dets = detectors[0](images_test[i]);
		t = (double)cvGetTickCount() - t;
		printf("检测 = %gms\n", t / ((double)cvGetTickFrequency()*1000.));
		win2.clear_overlay();
		win2.set_image(images_test[i]);
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

void encode_dat(std::string models_name)
{

	dlib::base64 base64_coder;
	dlib::compress_stream::kernel_1ea compressor;
	std::ostringstream sout;
	std::istringstream sin;
	
	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
	object_detector<image_scanner_type> detector_temp;
	deserialize(models_name) >> detector_temp;
	serialize(detector_temp, sout);
	
	//std::cout <<"1:"<< sout.str() << endl;
	//// Decode the base64 text into its compressed binary form
	////base64_coder.decode(sin, sout);
	//sin.str(sout.str());
	//sout.str("");
	//base64_coder.encode(sin, sout);
	//std::cout << "2:" << sout.str() << endl;
	//sin.clear();
	//sin.str(sout.str());
	//sout.str("");

	//// Decompress the data into its original form
	////compressor.decompress(sin, sout);
	//compressor.compress(sin, sout);
	//std::cout << "2:" << sout.str() << endl;

}

int main(int argc, char** argv) try
{
	
	std::string model = "test2.svm";
	//std::string model[2];
	//model[0] = "obj_test2.svm";
	//model[1]= "obj_test.svm";
	//obj_train(
	//	"/obj/",
	//	model,
	//	110,
	//	110,
	//	6,
	//	true,
	//	700,
	//	0.01
	//);

	
	obj_test(
		"/obj/",
		"/",
		&model,
		1
	);

	cin.get();
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}
