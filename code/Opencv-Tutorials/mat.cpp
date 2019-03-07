#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;
/*
Mat则是一个基础的图像容器，作为一个
c++的对象类，我们在使用是不需要担心的内存管理情况，它由由两个数据部分组成：矩阵头（包含矩阵尺寸，
存储方法，存储地址等信息）
和一个指向存储所有像素值的矩阵（根据所选存储方法的不同矩阵可以是不同的维数）的指针。
使用Mat读取图像十分便捷：Mat srcImg = imread("1.jpg");

1，Mat对象创建
1.1 使用Mat() 构造函数
Mat M(2,2, CV_8UC3, Scalar(0,0,255));//对二维多通道图像，要定义行数和列数。
指定存储元素的数据类型以及每个矩阵点的通道数。第三个参数格式如下
CV_[The number of bits per item][Signed or Unsigned][Type Prefix][The channel number]
Scalar 是short型vector。指定这个能用定制值初始化矩阵。
1.2 使用数组初始化Mat对象
int sz[3] = {2,2,2};
Mat L(3,sz, CV_8UC(1), Scalar::all(0));
1.3 调用函数创建
M.create(4,4, CV_8UC(2));
cout << "M = "<< endl << " "  << M << endl << endl;
1.4 初始为基础矩阵（单位矩阵，零矩阵等）
Mat E = Mat::eye(4, 4, CV_64F);

Mat n = Mat::ones(2, 2, CV_32F);

Mat Z = Mat::zeros(3,3, CV_8UC1);

1.5 小矩阵初始化
Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);


2, Mat对象的拷贝
2.1浅拷贝
Mat A, C;                                 // 只创建信息头部分
A = imread("src1.BMP", CV_LOAD_IMAGE_COLOR); // 这里为矩阵开辟内存
Mat B(A);                                 // 使用拷贝构造函数
C = A;                                    // 赋值运算符
2.2 ROI设置
Mat D (A, Rect(10, 10, 100, 100) ); // using a rectangle
Mat E = A(Range:all(), Range(1,3)); // using row and column boundaries
Mat RowClone = C.row(1).clone();
2.3 深拷贝
Mat F = A.clone();
Mat G;
A.copyTo(G);

3. ROI 测试*/

void main()
{
Mat srcImg0 = imread("src.jpg");
imshow("src0", srcImg0);

Mat D(srcImg0, Rect(10, 10, 100, 100));
D = D.mul(2);
imshow("DST", srcImg0);
}


