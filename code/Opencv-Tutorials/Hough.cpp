/*
霍夫线变换，圆变换
1，霍夫线变换是检测直线的一种方法，主要方法是判断多个点通过了同一条直线，首先使用极坐标表述一个点可能通过的所有直线，
则极坐标方程就是一条正弦曲线，通过比较多个点的正弦曲线知否交汇在同一点，即可判断它们是否通过同一条直线。
2，霍夫圆变换同霍夫线变换类似，在极坐标中使用圆心和坐标来表述一个圆，得到描述过一点所有圆的曲线，同样如果多个点交汇，
则说明这几个点在同一个圆上。
3，opencv函数：
（1）标准霍夫线变换
在OpenCV 中通过函数 HoughLines 来实现，得到一些极坐标参数
（2）统计概率霍夫线变换
在OpenCV 中它通过函数 HoughLinesP 来实现，得到直线的端点
（3）霍夫圆变换 HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );
函数带有以下自变量:
src_gray:		 输入图像 (灰度图)
circles:		存储下面三个参数:  集合的容器来表示每个检测到的圆.
CV_HOUGH_GRADIENT:		指定检测方法. 现在OpenCV中只有霍夫梯度法
dp = 1:		累加器图像的反比分辨率
min_dist = src_gray.rows/8:	检测到圆心之间的最小距离
param_1 = 200:			Canny边缘函数的高阈值
param_2 = 100:			圆心检测阈值.
min_radius = 0:		能检测到的最小圆半径, 默认为0.
max_radius = 0:		能检测到的最大圆半径, 默认为0 
*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;

void main()
{
	Mat src, dst, gray;

	/// 装载图像
	src = imread("5.jpg");


	cvtColor(src, gray, CV_BGR2GRAY);
	imshow("src", gray);
	GaussianBlur(gray, gray, Size(3, 3), 1, 2);
	vector<Vec4i> lines;
	vector<Vec3f> circles;

	//HoughLinesP(gray, lines, 1, CV_PI / 20, 200, 20, 10);
	HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, 1, 200, 60, 0, 2000);
	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	Vec4i l = lines[i];
	//	line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	//}
	for (size_t i = 0; i < circles.size(); i++)
	{

		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}



	imshow("dst", src);

	waitKey(0);
}