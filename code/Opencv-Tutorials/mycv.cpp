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
图像像素操作：
*/

void set_value(Mat& src, int value)
{


	int height = src.rows;                     // 图像高度 
	int width = src.cols;                       // 图像宽度（像素为单位） 
	int step = src.step;                 // 相邻行的同列点之间的字节数 
	int channels = src.channels();         // 颜色通道数目 (1,2,3,4) 
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

	/*或者：
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
	LUT(src, table, d); //查表
	d.convertTo(dst, CV_8U);
	return 0;
}

void elbp(Mat& src, Mat &dst, int radius, int neighbors)
{

	dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1);
	dst.setTo(0);

	for (int n = 0; n<neighbors; n++)
	{
		// 采样点的计算
		float x = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
		float y = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
		// 上取整和下取整的值
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// 小数部分
		float ty = y - fy;
		float tx = x - fx;
		// 设置插值权重
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// 循环处理图像数据
		for (int i = radius; i < src.rows - radius;i++)
		{
			for (int j = radius;j < src.cols - radius;j++)
			{
				// 计算插值
				float t = static_cast<float>(w1*src.at<uchar>(i + fy, j + fx) + w2*src.at<uchar>(i + fy, j + cx) + w3*src.at<uchar>(i + cy, j + fx) + w4*src.at<uchar>(i + cy, j + cx));
				// 进行编码
				dst.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) || (std::abs(t - src.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}


void clacHimg(Mat& src, Mat& histImage)
{

	//imshow("src", src);
	/// 分割成3个单通道图像 ( R, G 和 B )
	vector<Mat> rgb_planes;
	split(src, rgb_planes);
	/// 设定bin数目
	int histSize = 255;
	/// 设定取值范围 ( R,G,B) )
	float range[] = { 0, 255 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat r_hist, g_hist, b_hist;
	/// 计算直方图:
	calcHist(&rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	// 创建直方图画布
	int hist_w = 400; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage0(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
	histImage = histImage0.clone();
	/// 将直方图归一化到范围 [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// 在直方图画布上画出直方图
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

	// 显示直方图
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
	//卷积运算
	cv::filter2D(image, ConResMat, image.type(), sobelx);
	//计算梯度的幅度
	cv::Mat graMagMat;
	cv::multiply(ConResMat, ConResMat, graMagMat);
	//根据梯度幅度及参数设置阈值
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
	//		//并根据自适应阈值参数进行二值化
	//		pDataRes[i * cols + j] = 255 * ((pDataMag[i * cols + j] > thresh));
	//	}
	//}
	//resulttemp.convertTo(resulttemp, CV_8UC1);
	ConResMat.convertTo(resulttemp, CV_8UC1);
	result = resulttemp.clone();
	return true;
}




/*图像格式转换*/
#include <direct.h>
#include <io.h>
int con_vert(string src_path)
{
	String dirname1 = src_path + "*.bmp";
	String res_path = src_path + "convert";

	vector< String > files;
	glob(dirname1, files); //遍历指定的文件夹，读取符合搜索模板的所有文件路径
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

//int main(int argc, char** argv)
//{
//
//	con_vert("H:/num/9/");
//	waitKey(0);
//
//}

void show_num(cv::Mat& src, CvPoint po, double shownum)

{
	char res[20];

	sprintf_s(res, "%.2f", shownum);

	cv::putText(src, res, po, CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 1);

}

double get_d_point(Point p1, Point p2)
{

	double d = sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
	return d;
}

double get_rot_angle(Mat& src)
{
	Mat srcImg;
	double angle = 0;
	cvtColor(src, srcImg, CV_BGR2GRAY);
	threshold(srcImg, srcImg, 150, 255, CV_THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(srcImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Mat rotate_dstImage;
	Mat rot_mat(2, 3, CV_32FC1);
	for (int i = 0; i < contours.size(); i++)
	{
		drawContours(src, contours, i, Scalar(0, 255, 0), 2, 8);
		RotatedRect rect = minAreaRect(contours[i]);
		Point2f p[4];
		rect.points(p);
		Point center = rect.center;
		double scale = 1;

		for (int j = 0;j <= 3;j++)
		{
			line(src, p[j], p[(j + 1) % 4], Scalar(0, 0, 255), 2);

		}
		double d1 = get_d_point(p[0], p[3]);
		double d2 = get_d_point(p[2], p[3]);

		angle = (d1 > d2) ? rect.angle : rect.angle + 90; //+ 90.0


	}

	return angle;

}

Mat rot(Mat& src, double src_angle)
{
	Mat srcImg;
	cvtColor(src, srcImg, CV_BGR2GRAY);
	threshold(srcImg, srcImg, 150, 255, CV_THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(srcImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Mat rotate_dstImage;
	Mat rot_mat(2, 3, CV_32FC1);
	for (int i = 0; i < contours.size(); i++)
	{
		drawContours(src, contours, i, Scalar(0, 255, 0), 2, 8);
		RotatedRect rect = minAreaRect(contours[i]);
		Point2f p[4];
		rect.points(p);
		Point center = rect.center;

		double scale = 1;


		for (int j = 0;j <= 3;j++)
		{
			line(src, p[j], p[(j + 1) % 4], Scalar(0, 0, 255), 2);

			//show_num(src, p[j], j);
			//cout << j << ":" << p[j] << endl;
		}
		double d1 = get_d_point(p[0], p[3]);
		double d2 = get_d_point(p[2], p[3]);

		double angle = (d1 > d2) ? rect.angle : rect.angle + 90; //+ 90.0

		cout << i << ":" << angle << endl;
		rot_mat = getRotationMatrix2D(center, -(src_angle - angle), scale);

	}

	warpAffine(src, rotate_dstImage, rot_mat, src.size());

	return rotate_dstImage;
}

int main(int argc, char** argv)
{

	//for (int i = 1;i <= 7;i++)
	//{
	//	string pth = format(".//t//%d.jpg", i);
	//	Mat img = imread(pth, 1);
	//	imshow("img", img);

	//	Mat dst, histImage;
	//	//clacHimg(img, histImage);
	//	//FeatureConv(dst, histImage);

	//	//compareHist(dst,);
	//	string pth1 = format(".//r//%d.jpg", i);
	//	imwrite(pth1, histImage);
	//}

	Mat img = imread("1.jpg", 1);
	double angle = get_rot_angle(img);
	imshow("src", img);

	Mat img2 = imread("3.jpg", 1);
	Mat dst2 = rot(img2, angle);
	imshow("src2", img2);
	imshow("dst2", dst2);
	waitKey(0);

}
