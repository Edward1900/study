//gamma颜色空间归一化(g可取0.5)，也可以用作数据扩展
int gamma(Mat& src, double g, Mat& dst)
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

void maint(int argc, char** argv)
{
	Mat img = imread("F:/2019-02-12_2019-02-26/hand/39AEAAF8EFCBCF519FA97D68FB0FA432.jpg");
	Mat dst, dst2;
	//Rect r(1000, 305, 80, 80);

	//Mat roi = img(r);
	//roi = roi*1.2;
	//roi = roi + 100;
	//rectangle(img,r,Scalar(255,0,0),10,8,0);


	resize(img, dst, Size(img.cols / 4, img.rows / 4));
	//resize(dst, dst2, Size(img.cols / 4, img.rows / 4));
	//dst2 = dst2 + 20;


	gamma(dst, 0.5, dst2);

	imshow("img", dst);
	imshow("src", dst2);
	//imwrite("dst.jpg",dst);

	waitKey(0);


}