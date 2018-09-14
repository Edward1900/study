#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;

/*
仿射变换与透视变换
1，仿射变换其实是两个矩阵的乘法运算后再加上一个向量矩阵，它将一个矩阵向量映射到另一个向量空间，正如在二维空间中的条直线，
通过线性运算可以将其旋转（线性变换）和平移，也可以看作是坐标轴的改变（空间变换）。
在opencv中首先需要得到一个映射矩阵，利用映射矩阵将原图进行变换。常使用的函数有：
（1）， getAffineTransform( srcTri, dstTri ); //根据两组三个点得到映射矩阵
（2）， getRotationMatrix2D( center, angle, scale ); //根据旋转的中心，角度与比例得到映射矩阵
（3）， warpAffine( src, warp_dst, warp_mat, warp_dst.size() );/// 对源图像对某个映射矩阵求得仿射变换
2，透视变换在仿射变换的基础上好像多了一个维度，仿射变换只是对图像的一个旋转，缩放，平移，透视变换貌似将图像在z轴上进行了扭曲操作。
同样在使用时需要得到一个映射矩阵，利用映射矩阵对原图进行变换。使用函数：
（1）, getPerspectiveTransform( InputArray src, InputArray dst );//根据两组四个点得到映射矩阵
（2），perspectiveTransform(InputArray src, OutputArray dst, InputArray m ); //进行透视变换

3，透视变换测试：*/

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
	/// 设置目标图像的大小和类型与源图像一致
	//pers_dst = Mat::zeros(src.rows, src.cols, src.type());

	/// 设置源图像和目标图像上的三组点以计算仿射变换
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src.cols - 1, 0);
	srcTri[2] = Point2f(0, src.rows - 1);
	srcTri[3] = Point2f(src.cols - 1, src.rows - 1);

	dstTri[0] = Point2f(src.cols*0.0, src.rows*0.33);
	dstTri[1] = Point2f(src.cols*0.85, src.rows*0.25);
	dstTri[2] = Point2f(src.cols*0.15, src.rows*0.7);
	dstTri[3] = Point2f(src.cols*0.85, src.rows*0.7);
	/// 求得仿射变换
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

	/// 显示结果
	imshow("img_trans", img_trans);

	/// 等待用户按任意按键退出程序
	waitKey(0);

}

