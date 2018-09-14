/*
求积分图及任意区域像素和
1，积分图就是图像上一点（x,y）的左上面积之和，利用积分图可以对图像中的任意矩形区域，
利用其四个点的积分值的加减运算即可求出此区域的像素之和，对于图像需要求和的特征可以加快运算速度，
只需计算一次积分图，后续即可使用查询（结合加减运算）的方式获得特征值。
2，函数：integral(image, imageIntegral, CV_32F);
计算任意区域的像素和：
float integra_rect(Mat& image,int x1,int y1,int x2,int y2)
{
float dst_val = image.at<float>(x2,y2)+ image.at<float>(x1, y1) - image.at<float>(x1, y2) - image.at<float>(x2, y1);
return dst_val;
}

*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;

float integra_rect(Mat& image, int x1, int y1, int x2, int y2)
{
	float dst_val = image.at<float>(x2, y2) + image.at<float>(x1, y1) - image.at<float>(x1, y2) - image.at<float>(x2, y1);
	return dst_val;
}

void main()
{
	Mat src = imread("2.jpg");

	Mat image;
	cvtColor(src, image, CV_RGB2GRAY); //原图像是三通道，积分图也是三通道
	Mat imageIntegral;
	integral(image, imageIntegral, CV_32F); //计算积分图
	normalize(imageIntegral, imageIntegral, 0, 255, CV_MINMAX);  //归一化，方便显示
	Mat imageIntegralNorm;
	convertScaleAbs(imageIntegral, imageIntegralNorm); //精度转换为8位int整型

	imshow("src", image);
	imshow("dst", imageIntegralNorm);
	cvWaitKey(0);


}