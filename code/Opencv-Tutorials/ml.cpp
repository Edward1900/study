
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "ml.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;
/*
Adboost����������
1��Adboost������������������ѵ��һ��Ŀ�������������������������ǿ��������ÿ��ǿ�������ְ����������������
ͨ������Ŀ������һ�����ų�����֤��Ŀ�����׼ȷ�ԡ�
2���������Adboostѵ��Ŀ��������
opencv�ṩ��ѵ���Ĺ���opencv_traincascade.exe��ͨ���ռ��ͱ�ע������ʹ��
opencv_createsamples.exe����ѵ����������ݸ�ʽ.vec�ļ���Ȼ��ʹ��opencv_traincascade.exe������Ӧ�Ĳ���
����ѵ����
ָ�����£�
��1��opencv_createsamples.exe  -info  pos.txt  -bg  neg.txt -vec  pos.vec  -num  2018 -w 30  -h  20
����˵����
-info ���������������ļ���Ĭ��NULL
-img ����ͼ���ļ�����Ĭ��NULL
-bg �����������ļ����ļ��а���һϵ�еı����ѡ�����屳����ͼ���ļ�����Ĭ��NULL
-num ��������������Ŀ��Ĭ��1000
-bgcolor ������ɫ����ʾ͸����ɫ��Ĭ��0
-bgthresh ��ɫ�ݲ���д���bgcolor-bgthresh��bgcolor+bgthresh֮������ر���Ϊ͸�����أ�Ҳ���ǽ��������ӵ�ǰ��ͼ���ϣ�Ĭ��80
-inv ǰ��ͼ����ɫ��ת��־�����ָ����ɫ��ת��Ĭ��0(����ת)
-randinv ���ָ����ɫ�������ת��Ĭ��0
-maxidev ǰ��ͼ�������ص������ݶ����ֵ��Ĭ��40
-maxxangle X�������ת�Ƕȣ��Ի���Ϊ��λ��Ĭ��1.1
-maxyangle Y�������ת�Ƕȣ��Ի���Ϊ��λ��Ĭ��1.1
-maxzangle Z�������ת�Ƕȣ��Ի���Ϊ��λ��Ĭ��0.5
����ͼ�����������������ת����ת�Ƕ�������3��ֵ�޶���
-show ���ָ����ÿ��������������ʾ������Esc�������򽫼�������������������ʾ��Ĭ��Ϊ0(����ʾ)
-scale ��ʾͼ������ű�����Ĭ��4.0
-w ���������ȣ�Ĭ��24
-h ��������߶ȣ�Ĭ��24
-vec �������ѵ����.vec�ļ���Ĭ��NULL

��2��opencv_traincascade.exe  -data "J:\train_model\data" -vec  pos.vec  -bg  neg.txt -numPos  2000  -numNeg  2500
-numStages  15  -precalcValBufSize  5000  -precalcIdxBufSize  5000  -w 30  -h  20
-maxWeakCount  150   -minHitRate  0.9995  -maxFalseAlarmRate 0.20
����˵����
-data Ŀ¼��xml�����ѵ���õķ����������������ѵ���������д���
-vec pos.vec�ļ�����opencv_createsamples����
-bg �����������ļ�, neg\neg.txt
-numPos ÿ��������ѵ��ʱ���õ�����������Ŀ
-numNeg ÿ��������ѵ��ʱ���õ��ĸ�������Ŀ�����Դ���-bgָ����ͼƬ��Ŀ
-numStages ѵ���������ļ�����Ĭ��20����һ����14-25��֮����ɡ�
����������࣬��������fals alarm�͸�С�����ǲ���������������ʱ���������������hitrate�͸�С������ٶȾ�������������������٣�����û��Ҫ���úܶࡣ
-precalcValBufSize �����С�����ڴ洢Ԥ�ȼ��������ֵ����λMB
-precalcIdxBufSize �����С�����ڴ洢Ԥ�ȼ����������������λM��
-baseFormatSave ����ʹ��Haar����ʱ��Ч�����ָ�������������������ϸ�ʽ�洢
-stageType �������ͣ�staticconst char* stageTypes[] = { CC_BOOST };
-featureType �������ͣ�staticconst char* featureTypes[] = { CC_HAAR, CC_LBP, CC_HOG };
-w
-h ѵ�������ĳߴ磬�����ʹ��opencv_createsamples������ѵ�������ߴ籣��һ��
-bt Boosted����������
DAB-discrete Adaboost, RAB-RealAdaboost, LB-LogiBoost, GAB-Gentle Adaboost
-minHitRate ��������ÿһ��ϣ���õ�����С����ʣ��ܵ�������ʴ�ԼΪmin_hit_rate^number_of_stages
-maxFalseAlarmRate ��������ÿһ��ϣ���õ����������ʣ��ܵ�����ʴ�ԼΪmax_false_rate^number_of_stages
-weightTrimRate Specifies whether trimming should beused and its weight. һ�����������ֵ��0.95
-maxDepth ���������������ȣ�һ��������ֵ��1��������
-maxWeightCount ÿһ�������������������Ŀ
-mode ѵ������ʹ�õ�Haar�������ͣ�CORE-Allupright ALL-All Features BASIC-Viola

3���������������ԣ�



*/

/*
svm ������ѵ��
1�����hog������svm������ѵ�����ԺܺõĶ�������м�⡣HOG��һ���ݶ�ֱ��ͼ������ͨ����
ͼ��ָ��С����ͨ�������ٶ���ͨ�������ݶȷ����������ɴ�����������ֱ��ͼ������һ��ͼ���������
svm��������������֮��ѵ���γ�һ�����ŵĳ�ƽ�棬���ô˳�ƽ����ԺܺõĶ����ݽ��з��ࡣ�Է����Կɷֵ�����
����ͨ���˺�������ӳ�䵽�ɷֿռ���з��ࡣ
2��hog������ȡ��
Mat src; vector<float> descriptors;
HOGDescriptor myHog = HOGDescriptor(Size(64, 64), Size(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
myHog.compute(src.clone(), descriptors, Size(1, 1), Size(0, 0));
Mat desMat = Mat(descriptors);

3,svm������ѵ����
Mat labelsMat(num, 1, CV_32SC1, labels);
Mat train_images1;
train_images.convertTo(train_images1, CV_32FC1);

����������ѵ��
CvSVMParams params;
params.svm_type    = CvSVM::C_SVC;
params.kernel_type = CvSVM::LINEAR;
params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
CvSVM SVM;
SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
SVM����. ��������ѡ���� CvSVM::C_SVC ���ͣ������Ϳ�������n-��������� (n  2)��
������������� CvSVMParams.svm_type ������.
Note CvSVM::C_SVC ���͵���Ҫ�����������Դ����������������� (��ѵ�����ݲ�������ȫ�����Էָ�)��

���ߣ�

Ptr<ml::SVM> svm = ml::SVM::create();
svm->setType(ml::SVM::C_SVC);
svm->setKernel(ml::SVM::LINEAR);
svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5000, 1e-6));
svm->train(train_images1, ml::ROW_SAMPLE, labelsMat);
svm->save("svm.xml");

4�����������ط���
CvSVM SVM;
SVM.load("svm.xml");
float response = SVM.predict(sampleMat);

����
Ptr<ml::SVM> svm = Algorithm::load<ml::SVM>("svm.xml");
float response = svm->predict(img);

5�����ԣ�
*/
int main_()
{
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	float labels[4] = { 1.0, -1.0, -1.0, -1.0 };
	Mat labelsMat(3, 1, CV_32FC1, labels);

	float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
	Mat trainingDataMat(3, 2, CV_32FC1, trainingData);

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	Vec3b green(0, 255, 0), blue(255, 0, 0);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << i, j);
			float response = SVM.predict(sampleMat);

			if (response == 1)
				image.at<Vec3b>(j, i) = green;
			else if (response == -1)
				image.at<Vec3b>(j, i) = blue;
		}

	// Show the training data
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

	// Show support vectors
	thickness = 2;
	lineType = 8;
	int c = SVM.get_support_vector_count();

	for (int i = 0; i < c; ++i)
	{
		const float* v = SVM.get_support_vector(i);
		circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
	}

	imwrite("result.png", image);        // save the image

	imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);

}




/*
https://docs.opencv.org/3.3.1/d8/d89/classcv_1_1ml_1_1DTrees.html
https://blog.csdn.net/qq_34106574/article/details/82020244
���ɭ��
���ɭ����һ�ּ���ѧϰ������ɭ�����ɺܶ��������ɣ�
ÿһ����������Ϊһ������ģ�;�������ѧϰ���㷨��ϳ�Ϊһ��ɭ�ַ���ģ�͡�
����������ÿһ�������Ľ����Ȩ��ͶƱ�ó����ս��ۡ�
������⼯��ѧϰ���ǽ��������ģ���ںϵ�һ��ѧϰ��ʽ����ôʲô�Ǿ������������ָ����ʲô��
�������һ����ָ��ѵ��ÿ����ʱѡ��ѵ������ʱ������зŻص�ѡȡ����ѵ���ﵽ�趨����������ľ������
������׼ȷ��ʱ��ѵ����ֹ��
*/


int main(int argc, char** argv)
{
	vector<Mat> images;
	vector<int> respones;

	for (int i = 0;i < 10000;i++)
	{
		string path = format(".//train//%.jpg", i);
		Mat img = imread(path, 0);
		images.push_back(img);
		respones.push_back(i % 100);

	}


	Ptr<cv::ml::RTrees> RT = cv::ml::RTrees::create();
	RT->setActiveVarCount = 100;//ÿ��������Ѳ��ʱ���ѡȡѵ�������ĸ�����
	RT->setMaxDepth = 5;  //ÿ�������������
	RT->setTermCriteria = TermCriteria(1, 50, 0.9);//��ֹ���������ͣ��������ĸ�����������׼ȷ��

	RT->train(images, 1, respones);

	waitKey(0);

}

