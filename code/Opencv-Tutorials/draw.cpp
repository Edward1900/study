/*
给图像附加额外边框或掩膜，画图形及文字
1，附加额外边框（常在卷积中使用）
函数：copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );
参数:
src: 原图像
dst: 目标图像
top, bottom, left, right: 各边界的宽度，此处定义为原图像尺寸的5%。
borderType: 边界类型，此处可以选择常数边界或者复制边界。
value: 如果 borderType 类型是 BORDER_CONSTANT, 该值用来填充边界像素。
2，与加框对应的是附加0掩膜
Result.row(0).setTo(Scalar(0));             // 上边界
Result.row(Result.rows-1).setTo(Scalar(0)); // 下边界
Result.col(0).setTo(Scalar(0));             // 左边界
Result.col(Result.cols-1).setTo(Scalar(0)); // 右边界

3，画图形及文字
图形函数：
Point 在图像中定义 2D 点
如何以及为何使用 Scalar()
函数 line() 绘 直线
函数 ellipse() 绘 椭圆
函数 rectangle() 绘 矩形
函数 circle() 绘 圆
函数 fillPoly() 绘 填充的多边形
文字函数：
putText( image, "Testing text ", org, rng.uniform(0,8),
rng.uniform(0,100)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), lineType);
例子中：
在 image 上绘制文字 “Testing text ” 。
文字的左下角将用点 org 指定。
字体参数是用一个在  之间的整数来定义。
字体的缩放比例是用表达式 rng.uniform(0, 100)x0.05 + 0.1 指定(表示它的范围是 )。
字体的颜色是随机的 (记为 randomColor(rng))。
字体的粗细范围是从 1 到 10, 表示为 rng.uniform(1,10) 。*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;
Scalar value;
RNG rng(12345);

void main()
{
	Mat src, dst;

	/// 装载图像
	src = imread("2.jpg");
	imshow("src", src);

	putText(src, "opencv goes from 0 to N", Point(80, 80), 1, 1.5, Scalar(0, 255, 0), 1, 8);
	rectangle(src, Rect(10, 10, 50, 50), Scalar(0, 255, 0), 1, 8);

	int top = (int)(0.05*src.rows), bottom = (int)(0.05*src.rows);
	int left = (int)(0.05*src.cols), right = (int)(0.05*src.cols);
	dst = src;
	value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	copyMakeBorder(src, dst, top, bottom, left, right, BORDER_CONSTANT, value);

	imshow("dst", dst);

	waitKey(0);
}