/*
ֱ��ͼ�����⻯
1��ֱ��ͼ���������ڸ�����ֵ�ϵ�ͳ����ֵ����0-255��ͼ����Ҷ�ֱ��ͼ����������ֵ���ִ�����ͳ�ƽ����
������ֱ��ͼ����ʽ��ʾ��
������calcHist( &rgb[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
������
&rgb[0]:	��������(�����鼯)
1:			��������ĸ��� (��������ʹ����һ����ͨ��ͼ������Ҳ�����������鼯 )
0:			��Ҫͳ�Ƶ�ͨ�� (dim)���� ����������ֻ��ͳ���˻Ҷ� (��ÿ�����鶼�ǵ�ͨ��)����ֻҪд 0 �����ˡ�
Mat():		����( 0 ��ʾ���Ը�����)�� ���δ���壬��ʹ������
r_hist:		����ֱ��ͼ�ľ���
1:			ֱ��ͼά��
histSize:	ÿ��ά�ȵ�bin��Ŀ
histRange:	ÿ��ά�ȵ�ȡֵ��Χ
uniform �� accumulate:		bin��С��ͬ�����ֱ��ͼ�ۼ�

2��ֱ��ͼ���⻯�ǽ�������ֵ���ֵĴ������Ȼ�����ͼ��������س��ֵĶ�������һ������
������ֱ��ͼ���⻯�������쵽��������
������equalizeHist( InputArray src, OutputArray dst );

3,ֱ��ͼͳ�Ʋ��ԣ�*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;


void main()
{
	Mat src, dst;

	/// װ��ͼ��
	src = imread("2.jpg");
	imshow("src", src);
	/// �ָ��3����ͨ��ͼ�� ( R, G �� B )
	vector<Mat> rgb_planes;
	split(src, rgb_planes);
	/// �趨bin��Ŀ
	int histSize = 255;
	/// �趨ȡֵ��Χ ( R,G,B) )
	float range[] = { 0, 255 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat r_hist, g_hist, b_hist;
	/// ����ֱ��ͼ:
	calcHist(&rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	// ����ֱ��ͼ����
	int hist_w = 400; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
	/// ��ֱ��ͼ��һ������Χ [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// ��ֱ��ͼ�����ϻ���ֱ��ͼ
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

	// ��ʾֱ��ͼ
	imshow("calcHist Demo", histImage);

	waitKey(0);

	/// װ��ͼ��
	src = imread("2.jpg", 0);
	imshow("src", src);
	equalizeHist(src, dst);
	imshow("dst", dst);

	waitKey(0);


}




