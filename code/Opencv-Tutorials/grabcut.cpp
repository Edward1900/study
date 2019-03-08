/*
1.������
GrabCut��һ�ֻ���ͼ�и��ͼ��ָ����

��Χ��Ҫ�ָ�Ķ�����û�ָ���ı߽��ʼ���㷨ʹ�ø�˹���ģ�͹���Ŀ�����ͱ�������ɫ�ֲ���
�����������ر�ǩ�Ϲ�������ɷ���������������ѡ������ͬ��ǩ���������������������
�������л���ͼ�и���Ż����ƶ����ǵ�ֵ��
����������ƿ��ܱȱ߽���е�ԭʼ���Ƹ�׼ȷ�������ظ������������ֱ��������
���ڷָ������Ҫѵ����˹���ģ�ͣ����Դ�ͼ��ȽϺ�ʱ��
*/

//2,�ӿڵ�������
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
