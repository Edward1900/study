/*
边缘检测：
1，Sobel 算子是一个离散微分算子 (discrete differentiation operator)。 它用来计算图像灰度函数的近似梯度。
Sobel 算子结合了高斯平滑和微分求导。图像在边缘变化明显，所以用一阶微分求极值就可以得到边缘。
2，Laplacian算子使用二阶导数可以用来 检测边缘 。图像在边缘时，一阶导数是极值，而二阶导数过零点。
因为图像是 “2维”, 我们需要在两个方向求导。使用Laplacian算子将会使求导过程变得简单。
3,canny算子 是 John F. Canny 于 1986年开发出来的一个多级边缘检测算法，也被很多人认为是边缘检测的 最优算法,
最优边缘检测的三个主要评价标准是:
低错误率: 标识出尽可能多的实际边缘，同时尽可能的减少噪声产生的误报。
高定位性: 标识出的边缘要与图像中的实际边缘尽可能接近。
最小响应: 图像中的边缘只能标识一次。
*滞后阈值: 最后一步，Canny 使用了滞后阈值，滞后阈值需要两个阈值(高阈值和低阈值):
如果某一像素位置的幅值超过 高 阈值, 该像素被保留为边缘像素。
如果某一像素位置的幅值小于 低 阈值, 该像素被排除。
如果某一像素位置的幅值在两个阈值之间,该像素仅仅在连接到一个高于 高 阈值的像素时被保留。
*Canny 推荐的 高:低 阈值比在 2:1 到3:1之间。
*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;



Mat edge_sobel(Mat src)
{
	Mat dst, src_gray;
	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	/// 转换为灰度图
	if (src.channels() > 1)
	{
		cvtColor(src, src_gray, CV_RGB2GRAY);
	}
	else
	{
		src_gray = src.clone();
	}

	Mat grad;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	/// 创建 grad_x 和 grad_y 矩阵
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// 求 X方向梯度
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// 求Y方向梯度
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// 合并梯度(近似)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	//imshow("x", abs_grad_x);
	//imshow("y", abs_grad_y);
	imshow("Sobel", grad);


	return grad;

}

Mat edge_laplacian(Mat src)
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat src_gray;
	if (src.channels() > 1)
	{
		cvtColor(src, src_gray, CV_RGB2GRAY);
	}
	else
	{
		src_gray = src.clone();
	}
	Mat edge;
	GaussianBlur(src_gray, src_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Laplacian(src_gray, edge, ddepth, 3, scale, delta, BORDER_DEFAULT);

	convertScaleAbs(edge, edge);

	imshow("Laplacian", edge);
	return edge;
}


Mat edge_canny(Mat src, double th1, double th2)
{
	Mat edge;
	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Canny(src, edge, th1, th2);

	imshow("Canny", edge);
	return edge;
}


int main(int argc, char** argv)
{

	Mat img = imread("2.jpg", IMREAD_COLOR);
	imshow("img", img);
	edge_sobel(img);
	edge_laplacian(img);
	edge_canny(img, 20, 60);

	waitKey(0);

}

