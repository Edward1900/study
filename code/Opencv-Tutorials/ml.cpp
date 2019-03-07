
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "ml.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;
/*
Adboost级联分类器
1，Adboost级联分类器可以用来训练一个目标检测器，级联分类器包括多个强分类器，每个强分类器又包含多个弱分类器，
通过将非目标样本一层层的排除，保证了目标检测的准确性。
2，如何利用Adboost训练目标检测器？
opencv提供了训练的工具opencv_traincascade.exe。通过收集和标注样本，使用
opencv_createsamples.exe生成训练所需的数据格式.vec文件，然后使用opencv_traincascade.exe设置相应的参数
运行训练。
指令如下：
（1）opencv_createsamples.exe  -info  pos.txt  -bg  neg.txt -vec  pos.vec  -num  2018 -w 30  -h  20
参数说明：
-info 输入正样本描述文件，默认NULL
-img 输入图像文件名，默认NULL
-bg 负样本描述文件，文件中包含一系列的被随机选作物体背景的图像文件名，默认NULL
-num 生成正样本的数目，默认1000
-bgcolor 背景颜色，表示透明颜色，默认0
-bgthresh 颜色容差，所有处于bgcolor-bgthresh和bgcolor+bgthresh之间的像素被置为透明像素，也就是将白噪声加到前景图像上，默认80
-inv 前景图像颜色翻转标志，如果指定颜色翻转，默认0(不翻转)
-randinv 如果指定颜色将随机翻转，默认0
-maxidev 前景图像中像素的亮度梯度最大值，默认40
-maxxangle X轴最大旋转角度，以弧度为单位，默认1.1
-maxyangle Y轴最大旋转角度，以弧度为单位，默认1.1
-maxzangle Z轴最大旋转角度，以弧度为单位，默认0.5
输入图像沿着三个轴进行旋转，旋转角度由上述3个值限定。
-show 如果指定，每个样本都将被显示，按下Esc键，程序将继续创建样本而不在显示，默认为0(不显示)
-scale 显示图像的缩放比例，默认4.0
-w 输出样本宽度，默认24
-h 输出样本高度，默认24
-vec 输出用于训练的.vec文件，默认NULL

（2）opencv_traincascade.exe  -data "J:\train_model\data" -vec  pos.vec  -bg  neg.txt -numPos  2000  -numNeg  2500
-numStages  15  -precalcValBufSize  5000  -precalcIdxBufSize  5000  -w 30  -h  20
-maxWeakCount  150   -minHitRate  0.9995  -maxFalseAlarmRate 0.20
参数说明：
-data 目录名xml，存放训练好的分类器，如果不存在训练程序自行创建
-vec pos.vec文件，由opencv_createsamples生成
-bg 负样本描述文件, neg\neg.txt
-numPos 每级分类器训练时所用到的正样本数目
-numNeg 每级分类器训练时所用到的负样本数目，可以大于-bg指定的图片数目
-numStages 训练分类器的级数，默认20级，一般在14-25层之间均可。
如果层数过多，分类器的fals alarm就更小，但是产生级联分类器的时间更长，分类器的hitrate就更小，检测速度就慢。如果正负样本较少，层数没必要设置很多。
-precalcValBufSize 缓存大小，用于存储预先计算的特征值，单位MB
-precalcIdxBufSize 缓存大小，用于存储预先计算的特征索引，单位M币
-baseFormatSave 仅在使用Haar特征时有效，如果指定，级联分类器将以老格式存储
-stageType 级联类型，staticconst char* stageTypes[] = { CC_BOOST };
-featureType 特征类型，staticconst char* featureTypes[] = { CC_HAAR, CC_LBP, CC_HOG };
-w
-h 训练样本的尺寸，必须跟使用opencv_createsamples创建的训练样本尺寸保持一致
-bt Boosted分类器类型
DAB-discrete Adaboost, RAB-RealAdaboost, LB-LogiBoost, GAB-Gentle Adaboost
-minHitRate 分类器的每一级希望得到的最小检测率，总的最大检测率大约为min_hit_rate^number_of_stages
-maxFalseAlarmRate 分类器的每一级希望得到的最大误检率，总的误检率大约为max_false_rate^number_of_stages
-weightTrimRate Specifies whether trimming should beused and its weight. 一个还不错的数值是0.95
-maxDepth 弱分类器的最大深度，一个不错数值是1，二叉树
-maxWeightCount 每一级中弱分类器的最大数目
-mode 训练过程使用的Haar特征类型，CORE-Allupright ALL-All Features BASIC-Viola

3，级联分类器测试：



*/

/*
svm 分类器训练
1，结合hog特征的svm分类器训练可以很好的对人体进行检测。HOG是一种梯度直方图特征，通过对
图像分割成小的连通域区域，再对连通域区求梯度方向特征，由此特征构建的直方图来描述一副图像的特征。
svm可以在两类数据之间训练形成一个最优的超平面，利用此超平面可以很好的对数据进行分类。对非线性可分的数据
可以通过核函数设置映射到可分空间进行分类。
2，hog特征提取：
Mat src; vector<float> descriptors;
HOGDescriptor myHog = HOGDescriptor(Size(64, 64), Size(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
myHog.compute(src.clone(), descriptors, Size(1, 1), Size(0, 0));
Mat desMat = Mat(descriptors);

3,svm分类器训练：
Mat labelsMat(num, 1, CV_32SC1, labels);
Mat train_images1;
train_images.convertTo(train_images1, CV_32FC1);

参数设置与训练
CvSVMParams params;
params.svm_type    = CvSVM::C_SVC;
params.kernel_type = CvSVM::LINEAR;
params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
CvSVM SVM;
SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
SVM类型. 这里我们选择了 CvSVM::C_SVC 类型，该类型可以用于n-类分类问题 (n  2)。
这个参数定义在 CvSVMParams.svm_type 属性中.
Note CvSVM::C_SVC 类型的重要特征是它可以处理非完美分类的问题 (及训练数据不可以完全的线性分割)。

或者：

Ptr<ml::SVM> svm = ml::SVM::create();
svm->setType(ml::SVM::C_SVC);
svm->setKernel(ml::SVM::LINEAR);
svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5000, 1e-6));
svm->train(train_images1, ml::ROW_SAMPLE, labelsMat);
svm->save("svm.xml");

4，分类器加载分类
CvSVM SVM;
SVM.load("svm.xml");
float response = SVM.predict(sampleMat);

或者
Ptr<ml::SVM> svm = Algorithm::load<ml::SVM>("svm.xml");
float response = svm->predict(img);

5，测试：
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
随机森林
随机森林是一种集成学习方法。森林是由很多决策树组成，
每一个决策树作为一个分类模型经过集成学习的算法组合成为一个森林分类模型。
分类结果根据每一决策树的结果及权重投票得出最终结论。
不难理解集成学习就是将多个分类模型融合的一种学习方式，那么什么是决策树？随机又指的是什么？
随机其中一点是指在训练每个树时选择训练样本时是随机有放回的选取。当训练达到设定决策树的树木，或者
期望的准确率时，训练终止。
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
	RT->setActiveVarCount = 100;//每个决策树巡礼时随机选取训练样本的个数；
	RT->setMaxDepth = 5;  //每个决策树的深度
	RT->setTermCriteria = TermCriteria(1, 50, 0.9);//终止条件：类型，决策树的个数，期望的准确率

	RT->train(images, 1, respones);

	waitKey(0);

}

