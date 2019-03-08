#define COMPILER_MSVC
#define NOMINMAX
#define PLATFORM_WINDOWS   // 指定使用tensorflow/core/platform/windows/cpu_info.h

#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/ops/standard_ops.h"  // 为了使用ops
#include "opencv2/opencv.hpp"

using namespace tensorflow;
using namespace tensorflow::ops;
using std::cout;
using std::endl;

class TF_algorithm {

public:

	GraphDef graph_def;
	Session* session;

	int TF_algorithm::alg_create(const std::string& model_path);
	int TF_algorithm::alg_run(cv::Mat img1);


};

int TF_algorithm::alg_create(const std::string& model_path)
{
	TF_CHECK_OK(ReadBinaryProto(Env::Default(), model_path, &graph_def));
	TF_CHECK_OK(NewSession(tensorflow::SessionOptions(), &session));
	TF_CHECK_OK(session->Create(graph_def));

	return 0;

}

int TF_algorithm::alg_run(cv::Mat img1)
{

	const std::string model_input_name = "image_tensor";    // 模型文件中的输入名

	std::vector<std::string> model_output_names;
	model_output_names.emplace_back("detection_boxes");
	model_output_names.emplace_back("detection_scores");
	model_output_names.emplace_back("detection_classes");
	model_output_names.emplace_back("num_detections");


	cv::Mat img;
	cv::cvtColor(img1, img, cv::COLOR_BGR2RGB);
	int input_height = img.rows;
	int input_width = img.cols;
	int channels = img.channels();

	// 取图像数据，赋给tensorflow支持的Tensor变量中
	uint8* source_data = (uint8*)img.data;
	tensorflow::Tensor input_tensor(tensorflow::DT_UINT8, TensorShape({ 1, input_height, input_width, channels })); //这里只输入一张图片，参考tensorflow的数据格式NCHW
	auto input_tensor_mapped = input_tensor.tensor<uint8, 4>(); // input_tensor_mapped相当于input_tensor的数据接口，“4”表示数据是4维的

																// 把数据复制到input_tensor_mapped中，实际上就是遍历opencv的Mat数据
	for (int i = 0; i < input_height; i++) {
		uint8* source_row = source_data + (i * input_width * channels);
		for (int j = 0; j < input_width; j++) {
			uint8* source_pixel = source_row + (j * channels);
			for (int c = 0; c < channels; c++) {
				uint8* source_value = source_pixel + c;
				input_tensor_mapped(0, i, j, c) = *source_value;
			}
		}
	}



	// 输入inputs
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
		{ model_input_name, input_tensor },
	};

	// 输出outputs
	std::vector<tensorflow::Tensor> outputs;

	TF_CHECK_OK(session->Run(inputs, { model_output_names }, {}, &outputs));
	// 对检测到的结果做一些限定
	auto out_boxes = outputs[0].tensor<float, 3>();
	auto out_scores = outputs[1].tensor<float, 2>();
	auto out_classes = outputs[2].tensor<float, 2>();

	//	cout << out_scores << endl;
	//  cout << out_classes << endl;
	// 只取得分最高的那一组结果（可以根据需要改成多组结果）

	if (std::abs(out_classes(0, 0) - 1) > -1e-4) {
		if (out_scores(0, 0) > 0.6) {
			// 扩大人脸区域1.5倍，得到扩大后的结果
			cv::Rect face_region(cv::Point(int(out_boxes(0, 0, 1)*input_width), int(out_boxes(0, 0, 0)*input_height)),
				cv::Point(int(out_boxes(0, 0, 3)*input_width), int(out_boxes(0, 0, 2)*input_height)));

			cv::rectangle(img, face_region, cv::Scalar(0, 255, 0), 2);
			cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
			cv::String scor = cv::format("%.2f:%.2f", out_classes(0, 0), out_scores(0, 0));

			//cout << "s:"<< scor << endl;
			cv::putText(img, scor, cv::Point(int(out_boxes(0, 0, 1)*input_width), int(out_boxes(0, 0, 0)*input_height)), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1, 8);
			//cv::imshow("resault", img);
			cv::imwrite("resault1.jpg", img);

		}
		else {
			return -2; //检测到的人脸可信度不高
		}
	}
	else {
		return -1;  //未检测到人脸
	}

	return 1;

}


// 人脸检测，返回检测出的结果
int FaceDetection(
	const std::string& model_path,
	const std::string& image_path,
	const std::string& model_input_name,
	std::vector<std::string> model_output_names,
	cv::Mat& out_image) {

	GraphDef graph_def;
	TF_CHECK_OK(ReadBinaryProto(Env::Default(), model_path, &graph_def));

	//std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
	Session* session;
	TF_CHECK_OK(NewSession(tensorflow::SessionOptions(), &session));

	TF_CHECK_OK(session->Create(graph_def));

	// 将图片读入Tensor
	cv::Mat img1 = cv::imread(image_path);
	cv::Mat img;
	cv::cvtColor(img1, img, cv::COLOR_BGR2RGB);
	int input_height = img.rows;
	int input_width = img.cols;
	int channels = img.channels();

	// 取图像数据，赋给tensorflow支持的Tensor变量中
	uint8* source_data = (uint8*)img.data;
	tensorflow::Tensor input_tensor(tensorflow::DT_UINT8, TensorShape({ 1, input_height, input_width, channels })); //这里只输入一张图片，参考tensorflow的数据格式NCHW
	auto input_tensor_mapped = input_tensor.tensor<uint8, 4>(); // input_tensor_mapped相当于input_tensor的数据接口，“4”表示数据是4维的

																// 把数据复制到input_tensor_mapped中，实际上就是遍历opencv的Mat数据
	for (int i = 0; i < input_height; i++) {
		uint8* source_row = source_data + (i * input_width * channels);
		for (int j = 0; j < input_width; j++) {
			uint8* source_pixel = source_row + (j * channels);
			for (int c = 0; c < channels; c++) {
				uint8* source_value = source_pixel + c;
				input_tensor_mapped(0, i, j, c) = *source_value;
			}
		}
	}


	// 输入inputs
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
		{ model_input_name, input_tensor },
	};

	// 输出outputs
	std::vector<tensorflow::Tensor> outputs;

	TF_CHECK_OK(session->Run(inputs, { model_output_names }, {}, &outputs));

	for (int i = 0;i < 10;i++)
	{
		// 运行会话，计算输出model_output_names，即模型中定义的输出数据名称，最终结果保存在outputs中
		double t = (double)cvGetTickCount();//精确测量函数
		TF_CHECK_OK(session->Run(inputs, { model_output_names }, {}, &outputs));
		t = (double)cvGetTickCount() - t; //计算检测到人脸所需时间
		printf("检测时间 = %gms\n", t / ((double)cvGetTickFrequency()*1000.));//打印到屏幕
	}
	cv::waitKey(1000);
	for (int i = 0;i < 10;i++)
	{
		// 运行会话，计算输出model_output_names，即模型中定义的输出数据名称，最终结果保存在outputs中
		double t = (double)cvGetTickCount();//精确测量函数
		TF_CHECK_OK(session->Run(inputs, { model_output_names }, {}, &outputs));
		t = (double)cvGetTickCount() - t; //计算检测到人脸所需时间
		printf("检测时间 = %gms\n", t / ((double)cvGetTickFrequency()*1000.));//打印到屏幕
	}

	// 对检测到的结果做一些限定
	auto out_boxes = outputs[0].tensor<float, 3>();
	auto out_scores = outputs[1].tensor<float, 2>();
	auto out_classes = outputs[2].tensor<float, 2>();

	cout << out_scores << endl;
	cout << out_classes << endl;
	// 只取得分最高的那一组结果（可以根据需要改成多组结果）

	if (std::abs(out_classes(0, 0) - 1) > -1e-4) {
		if (out_scores(0, 0) > 0.6) {
			// 扩大人脸区域1.5倍，得到扩大后的结果
			cv::Rect face_region(cv::Point(int(out_boxes(0, 0, 1)*input_width), int(out_boxes(0, 0, 0)*input_height)),
				cv::Point(int(out_boxes(0, 0, 3)*input_width), int(out_boxes(0, 0, 2)*input_height)));

			cv::rectangle(img, face_region, cv::Scalar(0, 255, 0), 2);
			cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
			cv::String scor = cv::format("%.2f:%.2f", out_classes(0, 0), out_scores(0, 0));

			//cout << "s:"<< scor << endl;
			cv::putText(img, scor, cv::Point(int(out_boxes(0, 0, 1)*input_width), int(out_boxes(0, 0, 0)*input_height)), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1, 8);
			//cv::imshow("resault", img);
			cv::imwrite("resault1.jpg", img);

		}
		else {
			return -2; //检测到的人脸可信度不高
		}
	}
	else {
		return -1;  //未检测到人脸
	}

	return 1;
}

int main()
{
	// 模型相关参数
	const std::string face_model_path = "frozen_inference_graph.pb";
	const std::string face_model_inputs = "image_tensor";    // 模型文件中的输入名
	const std::string face_boxes = "detection_boxes";  // 模型文件中的输出名，注意这个顺序，后面统一会放入vector中
	const std::string face_scores = "detection_scores";
	const std::string face_classes = "detection_classes";
	const std::string face_num_detections = "num_detections";

	std::vector<std::string> face_model_outputs;
	face_model_outputs.emplace_back(face_boxes);
	face_model_outputs.emplace_back(face_scores);
	face_model_outputs.emplace_back(face_classes);
	face_model_outputs.emplace_back(face_num_detections);

	TF_algorithm tf1;

	tf1.alg_create(face_model_path);

	cv::Mat img1 = cv::imread("5.jpg");
	tf1.alg_run(img1);

	cv::Mat src_image;
	//int ret = FaceDetection(face_model_path, "2.jpg", face_model_inputs, face_model_outputs, src_image);

	double t = (double)cvGetTickCount();//精确测量函数
										// 人脸提取
										//int ret = FaceDetection( face_model_path, "2.jpg", face_model_inputs, face_model_outputs, src_image);

	tf1.alg_run(img1);

	t = (double)cvGetTickCount() - t; //计算检测到人脸所需时间
	printf("总检测所用时间 = %gms\n", t / ((double)cvGetTickFrequency()*1000.));//打印到屏幕


																		 //cv::imshow("dst", src_image);
	cv::waitKey(0);
	/*if (1 != ret) {
	return ret;
	}*/

	return 1;
}
