/*
直方图及均衡化
1，直方图是求像素在各个数值上的统计数值，如0-255的图像，其灰度直方图即它各个数值出现次数的统计结果，
可以用直方图的形式显示。
函数：calcHist( &rgb[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
参数：
&rgb[0]:	输入数组(或数组集)
1:			输入数组的个数 (这里我们使用了一个单通道图像，我们也可以输入数组集 )
0:			需要统计的通道 (dim)索引 ，这里我们只是统计了灰度 (且每个数组都是单通道)所以只要写 0 就行了。
Mat():		掩码( 0 表示忽略该像素)， 如果未定义，则不使用掩码
r_hist:		储存直方图的矩阵
1:			直方图维数
histSize:	每个维度的bin数目
histRange:	每个维度的取值范围
uniform 和 accumulate:		bin大小相同，清除直方图痕迹

2，直方图均衡化是将各个数值出现的次数均匀化，如图像各个像素出现的都集中在一段区域，
可以用直方图均衡化将其拉伸到各个区域。
函数：equalizeHist( InputArray src, OutputArray dst );

3,直方图统计测试：*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;


void main()
{
	Mat src, dst;

	/// 装载图像
	src = imread("2.jpg");
	imshow("src", src);
	/// 分割成3个单通道图像 ( R, G 和 B )
	vector<Mat> rgb_planes;
	split(src, rgb_planes);
	/// 设定bin数目
	int histSize = 255;
	/// 设定取值范围 ( R,G,B) )
	float range[] = { 0, 255 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat r_hist, g_hist, b_hist;
	/// 计算直方图:
	calcHist(&rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	// 创建直方图画布
	int hist_w = 400; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
	/// 将直方图归一化到范围 [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// 在直方图画布上画出直方图
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	// 显示直方图
	imshow("calcHist Demo", histImage);

	waitKey(0);

	/// 装载图像
	src = imread("2.jpg", 0);
	imshow("src", src);
	equalizeHist(src, dst);
	imshow("dst", dst);

	waitKey(0);


}




