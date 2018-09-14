/*
�����ͼ�������������غ�
1������ͼ����ͼ����һ�㣨x,y�����������֮�ͣ����û���ͼ���Զ�ͼ���е������������
�������ĸ���Ļ���ֵ�ļӼ����㼴����������������֮�ͣ�����ͼ����Ҫ��͵��������Լӿ������ٶȣ�
ֻ�����һ�λ���ͼ����������ʹ�ò�ѯ����ϼӼ����㣩�ķ�ʽ�������ֵ��
2��������integral(image, imageIntegral, CV_32F);
����������������غͣ�
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
	cvtColor(src, image, CV_RGB2GRAY); //ԭͼ������ͨ��������ͼҲ����ͨ��
	Mat imageIntegral;
	integral(image, imageIntegral, CV_32F); //�������ͼ
	normalize(imageIntegral, imageIntegral, 0, 255, CV_MINMAX);  //��һ����������ʾ
	Mat imageIntegralNorm;
	convertScaleAbs(imageIntegral, imageIntegralNorm); //����ת��Ϊ8λint����

	imshow("src", image);
	imshow("dst", imageIntegralNorm);
	cvWaitKey(0);


}