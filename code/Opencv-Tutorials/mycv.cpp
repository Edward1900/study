/**
work by yafeng
*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;



/*
ͼ�����ز�����
*/

void set_value(Mat& src, int value)
{


	int height = src.rows;                     // ͼ��߶� 
	int width = src.cols;                       // ͼ���ȣ�����Ϊ��λ�� 
	int step = src.step;                 // �����е�ͬ�е�֮����ֽ��� 
	int channels = src.channels();         // ��ɫͨ����Ŀ (1,2,3,4) 
	uchar *data = (uchar *)src.data;


	for (int i = 0; i != height; ++i)
	{
		for (int j = 0; j != width; ++j)
		{
			for (int k = 0; k != channels; ++k)
			{

				data[i*step + j*channels + k] = value;

			}
		}
	}

	/*���ߣ�
	for (int i = 0;i < src.rows;i++)
	{
	uchar* current_src = src.ptr< uchar>(i);
	uchar* current_dst = dst.ptr< uchar>(i);
	for (int j = 0;j < src.cols;j++)
	{

	//if (current_src[j] == 0)

	current_dst[j] = value;


	}
	}*/

}





int gammm(Mat& src, double g, Mat& dst)
{
	Mat table, d;
	for (int i = 0;i < 256;i++)
	{
		table.push_back((int)(255 * pow(i / 255.0, g)));
		//cout << (int)(255 * pow(i / 255.0, g)) << " ,";
	}
	LUT(src, table, d); //���
	d.convertTo(dst, CV_8U);
	return 0;
}

void elbp(Mat& src, Mat &dst, int radius, int neighbors)
{

	dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1);
	dst.setTo(0);

	for (int n = 0; n<neighbors; n++)
	{
		// ������ļ���
		float x = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
		float y = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
		// ��ȡ������ȡ����ֵ
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// С������
		float ty = y - fy;
		float tx = x - fx;
		// ���ò�ֵȨ��
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// ѭ������ͼ������
		for (int i = radius; i < src.rows - radius;i++)
		{
			for (int j = radius;j < src.cols - radius;j++)
			{
				// �����ֵ
				float t = static_cast<float>(w1*src.at<uchar>(i + fy, j + fx) + w2*src.at<uchar>(i + fy, j + cx) + w3*src.at<uchar>(i + cy, j + fx) + w4*src.at<uchar>(i + cy, j + cx));
				// ���б���
				dst.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) || (std::abs(t - src.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}


void clacHimg(Mat& src, Mat& histImage)
{

	//imshow("src", src);
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
	Mat histImage0(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
	histImage = histImage0.clone();
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
	//	imshow("calcHist Demo", histImage);
	//	waitKey(0);

}

bool FeatureConv(cv::Mat image, cv::Mat &result)
{
	CV_Assert(image.channels() == 1);
	image.convertTo(image, CV_32FC1);

	cv::Mat sobelx = (cv::Mat_<float>(3, 3) << -1, 0, 2,
		-1, 0, 2,
		-1, 0, 2);
	cv::Mat ConResMat;
	//�������
	cv::filter2D(image, ConResMat, image.type(), sobelx);
	//�����ݶȵķ���
	cv::Mat graMagMat;
	cv::multiply(ConResMat, ConResMat, graMagMat);
	//�����ݶȷ��ȼ�����������ֵ
	//int scaleVal =1.5;
	//double thresh = scaleVal * cv::mean(graMagMat).val[0];
	cv::Mat resulttemp = cv::Mat::zeros(graMagMat.size(), graMagMat.type());
	//float* pDataMag = (float*)graMagMat.data;
	//float* pDataRes = (float*)resulttemp.data;
	//int rows = ConResMat.rows, cols = ConResMat.cols;
	//for (int i = 1; i < rows - 1; i++)
	//{
	//	for (int j = 1; j < cols - 1; j++)
	//	{
	//		//����������Ӧ��ֵ�������ж�ֵ��
	//		pDataRes[i * cols + j] = 255 * ((pDataMag[i * cols + j] > thresh));
	//	}
	//}
	//resulttemp.convertTo(resulttemp, CV_8UC1);
	ConResMat.convertTo(resulttemp, CV_8UC1);
	result = resulttemp.clone();
	return true;
}




/*ͼ���ʽת��*/
#include <direct.h>
#include <io.h>
int con_vert(string src_path)
{
	String dirname1 = src_path + "*.bmp";
	String res_path = src_path + "convert";

	vector< String > files;
	glob(dirname1, files); //����ָ�����ļ��У���ȡ��������ģ��������ļ�·��
	const char *p = res_path.c_str();

	if (_mkdir(p) == 0)cout << "mkdir";
	int k = 0;
	for (int i = 0; i < files.size();i++)
	{
		Mat src = imread(files[i]);

		std::stringstream ss;
		std::string str;
		ss << k;
		ss >> str;
		string rpth = res_path + "/" + str + ".png";k++;
		cout << rpth << endl;
		imwrite(rpth, src);

	}

	return 0;
}

int main(int argc, char** argv)
{

	con_vert("H:/num/9/");
	waitKey(0);

}

int main_t(int argc, char** argv)
{

	for (int i = 1;i <= 7;i++)
	{
		string pth = format(".//t//%d.jpg", i);
		Mat img = imread(pth, 1);
		imshow("img", img);

		Mat dst, histImage;
		//clacHimg(img, histImage);
		//FeatureConv(dst, histImage);

		//compareHist(dst,);
		string pth1 = format(".//r//%d.jpg", i);
		imwrite(pth1, histImage);
	}

	waitKey(0);

}