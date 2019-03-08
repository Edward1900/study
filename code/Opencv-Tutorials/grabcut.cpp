/*
1.概述：
GrabCut是一种基于图切割的图像分割方法。

从围绕要分割的对象的用户指定的边界框开始，算法使用高斯混合模型估计目标对象和背景的颜色分布。
这用于在像素标签上构建马尔可夫随机场，其具有优选具有相同标签的连接区域的能量函数，
并且运行基于图切割的优化以推断它们的值。
由于这个估计可能比边界框中的原始估计更准确，所以重复这个两步程序直到收敛。
由于分割过程需要训练高斯混合模型，所以大图像比较耗时。
*/

//2,接口调用如下
static void getBinMask(const Mat& comMask, Mat& binMask)
{
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	binMask = comMask & 1;
}

int m = 0;

int main_(int argc, char** argv)
{
	string filename = "dm.png";
	if (filename.empty())
	{
		cout << "\nDurn, empty filename" << endl;
		return 1;
	}
	Mat image = imread(filename, 1);
	if (image.empty())
	{
		cout << "\n Durn, couldn't read image filename " << filename << endl;
		return 1;
	}

	Mat mask, bgModel, fgModel;
	Rect rect(Point(66, 100), Point(330, 300));
	grabCut(image, mask, rect, bgModel, fgModel, 500, GC_INIT_WITH_RECT);
	rectangle(image, rect, Scalar(255), 2, 8);

	//compare(mask, GC_PR_FGD, mask, CMP_EQ);
	imshow("src", image);
	Mat binMask;
	getBinMask(mask, binMask);

	imshow("dst", binMask * 255);

	cvWaitKey(0);
	return 0;
}
