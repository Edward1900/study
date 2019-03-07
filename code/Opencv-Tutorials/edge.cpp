/*
��Ե��⣺
1��Sobel ������һ����ɢ΢������ (discrete differentiation operator)�� ����������ͼ��ҶȺ����Ľ����ݶȡ�
Sobel ���ӽ���˸�˹ƽ����΢���󵼡�ͼ���ڱ�Ե�仯���ԣ�������һ��΢����ֵ�Ϳ��Եõ���Ե��
2��Laplacian����ʹ�ö��׵����������� ����Ե ��ͼ���ڱ�Եʱ��һ�׵����Ǽ�ֵ�������׵�������㡣
��Ϊͼ���� ��2ά��, ������Ҫ�����������󵼡�ʹ��Laplacian���ӽ���ʹ�󵼹��̱�ü򵥡�
3,canny���� �� John F. Canny �� 1986�꿪��������һ���༶��Ե����㷨��Ҳ���ܶ�����Ϊ�Ǳ�Ե���� �����㷨,
���ű�Ե����������Ҫ���۱�׼��:
�ʹ�����: ��ʶ�������ܶ��ʵ�ʱ�Ե��ͬʱ�����ܵļ��������������󱨡�
�߶�λ��: ��ʶ���ı�ԵҪ��ͼ���е�ʵ�ʱ�Ե�����ܽӽ���
��С��Ӧ: ͼ���еı�Եֻ�ܱ�ʶһ�Ρ�
*�ͺ���ֵ: ���һ����Canny ʹ�����ͺ���ֵ���ͺ���ֵ��Ҫ������ֵ(����ֵ�͵���ֵ):
���ĳһ����λ�õķ�ֵ���� �� ��ֵ, �����ر�����Ϊ��Ե���ء�
���ĳһ����λ�õķ�ֵС�� �� ��ֵ, �����ر��ų���
���ĳһ����λ�õķ�ֵ��������ֵ֮��,�����ؽ��������ӵ�һ������ �� ��ֵ������ʱ��������
*Canny �Ƽ��� ��:�� ��ֵ���� 2:1 ��3:1֮�䡣
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

	/// ת��Ϊ�Ҷ�ͼ
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

	/// ���� grad_x �� grad_y ����
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// �� X�����ݶ�
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// ��Y�����ݶ�
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// �ϲ��ݶ�(����)
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

