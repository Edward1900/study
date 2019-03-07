/*
�����߱任��Բ�任
1�������߱任�Ǽ��ֱ�ߵ�һ�ַ�������Ҫ�������ж϶����ͨ����ͬһ��ֱ�ߣ�����ʹ�ü��������һ�������ͨ��������ֱ�ߣ�
�����귽�̾���һ���������ߣ�ͨ���Ƚ϶�������������֪�񽻻���ͬһ�㣬�����ж������Ƿ�ͨ��ͬһ��ֱ�ߡ�
2������Բ�任ͬ�����߱任���ƣ��ڼ�������ʹ��Բ�ĺ�����������һ��Բ���õ�������һ������Բ�����ߣ�ͬ���������㽻�㣬
��˵���⼸������ͬһ��Բ�ϡ�
3��opencv������
��1����׼�����߱任
��OpenCV ��ͨ������ HoughLines ��ʵ�֣��õ�һЩ���������
��2��ͳ�Ƹ��ʻ����߱任
��OpenCV ����ͨ������ HoughLinesP ��ʵ�֣��õ�ֱ�ߵĶ˵�
��3������Բ�任 HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );
�������������Ա���:
src_gray:		 ����ͼ�� (�Ҷ�ͼ)
circles:		�洢������������:  ���ϵ���������ʾÿ����⵽��Բ.
CV_HOUGH_GRADIENT:		ָ����ⷽ��. ����OpenCV��ֻ�л����ݶȷ�
dp = 1:		�ۼ���ͼ��ķ��ȷֱ���
min_dist = src_gray.rows/8:	��⵽Բ��֮�����С����
param_1 = 200:			Canny��Ե�����ĸ���ֵ
param_2 = 100:			Բ�ļ����ֵ.
min_radius = 0:		�ܼ�⵽����СԲ�뾶, Ĭ��Ϊ0.
max_radius = 0:		�ܼ�⵽�����Բ�뾶, Ĭ��Ϊ0 
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

	/// װ��ͼ��
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