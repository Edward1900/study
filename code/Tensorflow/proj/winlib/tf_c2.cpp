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

	const std::string model_input_name = "input";    // 模型文件中的输入名

	std::vector<std::string> model_output_names;
	model_output_names.emplace_back("decoded_out");
	//model_output_names.emplace_back("detection_scores");
	//model_output_names.emplace_back("detection_classes");
	//model_output_names.emplace_back("num_detections");
	
	
	cv::Mat img, img2;
	cv::resize(img1, img2, cv::Size(300, 300));
	cv::cvtColor(img2, img, cv::COLOR_BGR2RGB);
	
	int input_height = 300;
	int input_width = 300;
	int channels = 3;

	// 取图像数据，赋给tensorflow支持的Tensor变量中
	uint8* source_data = (uint8*)img.data;
	tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, TensorShape({ 1, input_height, input_width, channels })); //这里只输入一张图片，参考tensorflow的数据格式NCHW
	auto input_tensor_mapped = input_tensor.tensor<float, 4>(); // input_tensor_mapped相当于input_tensor的数据接口，“4”表示数据是4维的

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
	
	cout << outputs.size() << endl;
	for (int i = 0;i < outputs.size();i++)
	{
		cout << outputs[i].dims() << endl;
		for (int j = 0;j < outputs.size();j++)
		{
			auto shap = outputs[i].shape();
			auto d = shap.dim_sizes();
			cout << d[1] << endl;
			
			auto out_boxes = outputs[0].tensor<float, 3>();
			for (int k = 0; k < d[1]; k++)
			{
				//cout << out_boxes(0, k, m) << endl;
				if ((out_boxes(0, k, 0) == 7) || (out_boxes(0, k, 0) == 6))
				{
					if (out_boxes(0, k, 1) >= 0.3)
					{
						
						cv::Rect face_region(cv::Point(int(img1.cols*out_boxes(0, k, 2)/300), int(img1.rows*out_boxes(0, k, 3)/ 300)),
											cv::Point(int(img1.cols*out_boxes(0, k, 4)/ 300), int(img1.rows*out_boxes(0, k, 5)/ 300)));
						cv::String score = cv::format("%.2f", out_boxes(0, k, 1));
						cv::putText(img1, score, cv::Point(int(img1.cols*out_boxes(0, k, 2) / 300), int(img1.rows*out_boxes(0, k, 3) / 300)
										),1,1, cv::Scalar(0, 255, 0), 1);
						cv::rectangle(img1, face_region, cv::Scalar(0, 255, 0), 2);
						cv::imwrite("resault1.jpg", img1);
					}
				}
				
			}
			
		}
	
	}



	return 1;

}



int main()
{
	// 模型相关参数
	const std::string face_model_path = "xxx.pb";


	TF_algorithm tf1;

	tf1.alg_create(face_model_path);

	cv::Mat img1 = cv::imread("1.jpg");
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
