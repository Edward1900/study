#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;

/*
����任��͸�ӱ任
1������任��ʵ����������ĳ˷�������ټ���һ��������������һ����������ӳ�䵽��һ�������ռ䣬�����ڶ�ά�ռ��е���ֱ�ߣ�
ͨ������������Խ�����ת�����Ա任����ƽ�ƣ�Ҳ���Կ�����������ĸı䣨�ռ�任����
��opencv��������Ҫ�õ�һ��ӳ���������ӳ�����ԭͼ���б任����ʹ�õĺ����У�
��1���� getAffineTransform( srcTri, dstTri ); //��������������õ�ӳ�����
��2���� getRotationMatrix2D( center, angle, scale ); //������ת�����ģ��Ƕ�������õ�ӳ�����
��3���� warpAffine( src, warp_dst, warp_mat, warp_dst.size() );/// ��Դͼ���ĳ��ӳ�������÷���任
2��͸�ӱ任�ڷ���任�Ļ����Ϻ������һ��ά�ȣ�����任ֻ�Ƕ�ͼ���һ����ת�����ţ�ƽ�ƣ�͸�ӱ任ò�ƽ�ͼ����z���Ͻ�����Ť��������
ͬ����ʹ��ʱ��Ҫ�õ�һ��ӳ���������ӳ������ԭͼ���б任��ʹ�ú�����
��1��, getPerspectiveTransform( InputArray src, InputArray dst );//���������ĸ���õ�ӳ�����
��2����perspectiveTransform(InputArray src, OutputArray dst, InputArray m ); //����͸�ӱ任

3��͸�ӱ任���ԣ�*/

void main()
{

	Point2f srcTri[4];
	Point2f dstTri[4];

	Mat rot_mat(2, 3, CV_32FC1);
	Mat pers_mat(2, 3, CV_32FC1);
	Mat src, pers_dst;
	src = imread("2.jpg");
	imshow("src", src);
	int img_height = src.rows;
	int img_width = src.cols;
	/// ����Ŀ��ͼ��Ĵ�С��������Դͼ��һ��
	//pers_dst = Mat::zeros(src.rows, src.cols, src.type());

	/// ����Դͼ���Ŀ��ͼ���ϵ�������Լ������任
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src.cols - 1, 0);
	srcTri[2] = Point2f(0, src.rows - 1);
	srcTri[3] = Point2f(src.cols - 1, src.rows - 1);

	dstTri[0] = Point2f(src.cols*0.0, src.rows*0.33);
	dstTri[1] = Point2f(src.cols*0.85, src.rows*0.25);
	dstTri[2] = Point2f(src.cols*0.15, src.rows*0.7);
	dstTri[3] = Point2f(src.cols*0.85, src.rows*0.7);
	/// ��÷���任
	pers_mat = getPerspectiveTransform(srcTri, dstTri);

	vector<Point2f> ponits, points_trans;
	for (int i = 0;i < img_height;i++) {
		for (int j = 0;j < img_width;j++) {
			ponits.push_back(Point2f(j, i));
		}
	}

	perspectiveTransform(ponits, points_trans, pers_mat);
	Mat img_trans = Mat::zeros(img_height, img_width, CV_8UC3);
	int count = 0;
	for (int i = 0;i < img_height;i++) {
		uchar* p = src.ptr<uchar>(i);
		for (int j = 0;j < img_width;j++) {
			int y = points_trans[count].y;
			int x = points_trans[count].x;
			uchar* t = img_trans.ptr<uchar>(y);
			t[x * 3] = p[j * 3];
			t[x * 3 + 1] = p[j * 3 + 1];
			t[x * 3 + 2] = p[j * 3 + 2];
			count++;
		}
	}

	/// ��ʾ���
	imshow("img_trans", img_trans);

	/// �ȴ��û������ⰴ���˳�����
	waitKey(0);

}

