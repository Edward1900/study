#define COMPILER_MSVC
#define NOMINMAX
#define PLATFORM_WINDOWS   // ָ��ʹ��tensorflow/core/platform/windows/cpu_info.h

#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/ops/standard_ops.h"  // Ϊ��ʹ��ops
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

	const std::string model_input_name = "image_tensor";    // ģ���ļ��е�������

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

	// ȡͼ�����ݣ�����tensorflow֧�ֵ�Tensor������
	uint8* source_data = (uint8*)img.data;
	tensorflow::Tensor input_tensor(tensorflow::DT_UINT8, TensorShape({ 1, input_height, input_width, channels })); //����ֻ����һ��ͼƬ���ο�tensorflow�����ݸ�ʽNCHW
	auto input_tensor_mapped = input_tensor.tensor<uint8, 4>(); // input_tensor_mapped�൱��input_tensor�����ݽӿڣ���4����ʾ������4ά��

																// �����ݸ��Ƶ�input_tensor_mapped�У�ʵ���Ͼ��Ǳ���opencv��Mat����
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



	// ����inputs
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
		{ model_input_name, input_tensor },
	};

	// ���outputs
	std::vector<tensorflow::Tensor> outputs;

	TF_CHECK_OK(session->Run(inputs, { model_output_names }, {}, &outputs));
	// �Լ�⵽�Ľ����һЩ�޶�
	auto out_boxes = outputs[0].tensor<float, 3>();
	auto out_scores = outputs[1].tensor<float, 2>();
	auto out_classes = outputs[2].tensor<float, 2>();

	//	cout << out_scores << endl;
	//  cout << out_classes << endl;
	// ֻȡ�÷���ߵ���һ���������Ը�����Ҫ�ĳɶ�������

	if (std::abs(out_classes(0, 0) - 1) > -1e-4) {
		if (out_scores(0, 0) > 0.6) {
			// ������������1.5�����õ������Ľ��
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
			return -2; //��⵽���������ŶȲ���
		}
	}
	else {
		return -1;  //δ��⵽����
	}

	return 1;

}


// ������⣬���ؼ����Ľ��
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

	// ��ͼƬ����Tensor
	cv::Mat img1 = cv::imread(image_path);
	cv::Mat img;
	cv::cvtColor(img1, img, cv::COLOR_BGR2RGB);
	int input_height = img.rows;
	int input_width = img.cols;
	int channels = img.channels();

	// ȡͼ�����ݣ�����tensorflow֧�ֵ�Tensor������
	uint8* source_data = (uint8*)img.data;
	tensorflow::Tensor input_tensor(tensorflow::DT_UINT8, TensorShape({ 1, input_height, input_width, channels })); //����ֻ����һ��ͼƬ���ο�tensorflow�����ݸ�ʽNCHW
	auto input_tensor_mapped = input_tensor.tensor<uint8, 4>(); // input_tensor_mapped�൱��input_tensor�����ݽӿڣ���4����ʾ������4ά��

																// �����ݸ��Ƶ�input_tensor_mapped�У�ʵ���Ͼ��Ǳ���opencv��Mat����
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


	// ����inputs
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
		{ model_input_name, input_tensor },
	};

	// ���outputs
	std::vector<tensorflow::Tensor> outputs;

	TF_CHECK_OK(session->Run(inputs, { model_output_names }, {}, &outputs));

	for (int i = 0;i < 10;i++)
	{
		// ���лỰ���������model_output_names����ģ���ж��������������ƣ����ս��������outputs��
		double t = (double)cvGetTickCount();//��ȷ��������
		TF_CHECK_OK(session->Run(inputs, { model_output_names }, {}, &outputs));
		t = (double)cvGetTickCount() - t; //�����⵽��������ʱ��
		printf("���ʱ�� = %gms\n", t / ((double)cvGetTickFrequency()*1000.));//��ӡ����Ļ
	}
	cv::waitKey(1000);
	for (int i = 0;i < 10;i++)
	{
		// ���лỰ���������model_output_names����ģ���ж��������������ƣ����ս��������outputs��
		double t = (double)cvGetTickCount();//��ȷ��������
		TF_CHECK_OK(session->Run(inputs, { model_output_names }, {}, &outputs));
		t = (double)cvGetTickCount() - t; //�����⵽��������ʱ��
		printf("���ʱ�� = %gms\n", t / ((double)cvGetTickFrequency()*1000.));//��ӡ����Ļ
	}

	// �Լ�⵽�Ľ����һЩ�޶�
	auto out_boxes = outputs[0].tensor<float, 3>();
	auto out_scores = outputs[1].tensor<float, 2>();
	auto out_classes = outputs[2].tensor<float, 2>();

	cout << out_scores << endl;
	cout << out_classes << endl;
	// ֻȡ�÷���ߵ���һ���������Ը�����Ҫ�ĳɶ�������

	if (std::abs(out_classes(0, 0) - 1) > -1e-4) {
		if (out_scores(0, 0) > 0.6) {
			// ������������1.5�����õ������Ľ��
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
			return -2; //��⵽���������ŶȲ���
		}
	}
	else {
		return -1;  //δ��⵽����
	}

	return 1;
}

int main()
{
	// ģ����ز���
	const std::string face_model_path = "frozen_inference_graph.pb";
	const std::string face_model_inputs = "image_tensor";    // ģ���ļ��е�������
	const std::string face_boxes = "detection_boxes";  // ģ���ļ��е��������ע�����˳�򣬺���ͳһ�����vector��
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

	double t = (double)cvGetTickCount();//��ȷ��������
										// ������ȡ
										//int ret = FaceDetection( face_model_path, "2.jpg", face_model_inputs, face_model_outputs, src_image);

	tf1.alg_run(img1);

	t = (double)cvGetTickCount() - t; //�����⵽��������ʱ��
	printf("�ܼ������ʱ�� = %gms\n", t / ((double)cvGetTickFrequency()*1000.));//��ӡ����Ļ


																		 //cv::imshow("dst", src_image);
	cv::waitKey(0);
	/*if (1 != ret) {
	return ret;
	}*/

	return 1;
}
