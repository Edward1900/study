#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;
/*
Mat����һ��������ͼ����������Ϊһ��
c++�Ķ����࣬������ʹ���ǲ���Ҫ���ĵ��ڴ����������������������ݲ�����ɣ�����ͷ����������ߴ磬
�洢�������洢��ַ����Ϣ��
��һ��ָ��洢��������ֵ�ľ��󣨸�����ѡ�洢�����Ĳ�ͬ��������ǲ�ͬ��ά������ָ�롣
ʹ��Mat��ȡͼ��ʮ�ֱ�ݣ�Mat srcImg = imread("1.jpg");

1��Mat���󴴽�
1.1 ʹ��Mat() ���캯��
Mat M(2,2, CV_8UC3, Scalar(0,0,255));//�Զ�ά��ͨ��ͼ��Ҫ����������������
ָ���洢Ԫ�ص����������Լ�ÿ��������ͨ������������������ʽ����
CV_[The number of bits per item][Signed or Unsigned][Type Prefix][The channel number]
Scalar ��short��vector��ָ��������ö���ֵ��ʼ������
1.2 ʹ�������ʼ��Mat����
int sz[3] = {2,2,2};
Mat L(3,sz, CV_8UC(1), Scalar::all(0));
1.3 ���ú�������
M.create(4,4, CV_8UC(2));
cout << "M = "<< endl << " "  << M << endl << endl;
1.4 ��ʼΪ�������󣨵�λ���������ȣ�
Mat E = Mat::eye(4, 4, CV_64F);

Mat n = Mat::ones(2, 2, CV_32F);

Mat Z = Mat::zeros(3,3, CV_8UC1);

1.5 С�����ʼ��
Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);


2, Mat����Ŀ���
2.1ǳ����
Mat A, C;                                 // ֻ������Ϣͷ����
A = imread("src1.BMP", CV_LOAD_IMAGE_COLOR); // ����Ϊ���󿪱��ڴ�
Mat B(A);                                 // ʹ�ÿ������캯��
C = A;                                    // ��ֵ�����
2.2 ROI����
Mat D (A, Rect(10, 10, 100, 100) ); // using a rectangle
Mat E = A(Range:all(), Range(1,3)); // using row and column boundaries
Mat RowClone = C.row(1).clone();
2.3 ���
Mat F = A.clone();
Mat G;
A.copyTo(G);

3. ROI ����*/

void main()
{
Mat srcImg0 = imread("src.jpg");
imshow("src0", srcImg0);

Mat D(srcImg0, Rect(10, 10, 100, 100));
D = D.mul(2);
imshow("DST", srcImg0);
}


