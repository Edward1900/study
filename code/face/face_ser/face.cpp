// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#include"face.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <dlib/string.h>
//using namespace cv;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------



// ----------------------------------------------------------------------------------------
frontal_face_detector detector = get_frontal_face_detector();
shape_predictor sp;
anet_type net;


void init()
{
	deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
	// And finally we load the DNN responsible for face recognition.
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

}





std::vector<matrix<float, 0, 1>> comput_vector(matrix<rgb_pixel> img, std::vector<rectangle> face)
{
	std::vector<matrix<rgb_pixel>> faces;
	rectangle face_dst;
	int area = 0;
	for (int i = 0; i < face.size(); i++)
	{
		if (face[i].area() > area)
		{
			area = face[i].area();
			face_dst = face[i];
		}
	}

	auto shape = sp(img, face_dst);
	matrix<rgb_pixel> face_chip;
	extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
	faces.push_back(move(face_chip));


	if (faces.size() == 0)
	{
		cout << "No faces found in image!" << endl;

	}

	std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);

	return face_descriptors;

}



std::vector<matrix<float, 0, 1>> get_src_vector(std::string path)
{
	matrix<rgb_pixel> img;
	std::vector<matrix<float, 0, 1>> src_vector;
	//cv::Mat img = cv::imread("bald_guys.jpg",0);
	//load_image(img, path);
	cv::Mat src_img = cv::imread(path);
	assign_image(img, cv_image<rgb_pixel>(src_img));

	std::vector<rectangle> face = detector(img);
	if(face.size()>0)
	src_vector = comput_vector(img, face);

	return src_vector;
}


int save_em(std::map<std::string, matrix<float, 0, 1>> regfaces, std::string emdat)
{
	std::ofstream fout;
	fout.open(emdat, ios::out);

	std::map<std::string, matrix<float, 0, 1>>::iterator it;

	for (it = regfaces.begin(); it != regfaces.end(); ++it)
	{
		fout << it->first;
		fout << it->second << endl;
	}

	//fout << 'E';
	fout.flush();
	fout.close();

	return 1;
}



std::map<std::string, matrix<float, 0, 1>> load_em(std::string emdat)
{
	std::map<std::string, matrix<float, 0, 1>> regfaces;
	std::pair<std::string, matrix<float, 0, 1>> sample;
	
	std::ifstream fin;

	fin.open(emdat, ios::in);
		
	while (!fin.eof())
	//while (fin.peek()!='E')
	{
		
		fin >> sample.first;
		//cout << "l1:" << sample.first << endl;
		fin >> sample.second;
		//cout << "l2:"<< sample.second << endl;
		regfaces.insert(sample);

	}

	
	fin.close();

	return regfaces;
}

int Registeration(std::string id, std::string img, std::string emdat)
{
	std::map<std::string, matrix<float, 0, 1>> regfaces = load_em(emdat);
	std::vector<matrix<float, 0, 1>> emb_vector;

	emb_vector = get_src_vector(img);
	//regfaces[id] = emb_vector[0];
	if(emb_vector.size()>0)
	regfaces.insert(std::make_pair(id, emb_vector[0]));

	save_em(regfaces, emdat);

	return 1;
}

void show_emd(std::string emdat)
{
	std::map<std::string, matrix<float, 0, 1>> emd = load_em(emdat);

	
	std::map<std::string, matrix<float, 0, 1>>::iterator it;
	for (it = emd.begin(); it != emd.end(); ++it)
	{
		cout << it->first << it->second;
	}

}

std::vector<std::pair<std::string, float>>  search(std::string img, std::string emdat)
{
	std::map<std::string, matrix<float, 0, 1>> regfaces = load_em(emdat);
	std::vector<matrix<float, 0, 1>> emb_vector;
	std::vector<std::pair<std::string, float>> results;
	std::vector<std::pair<std::string, float>> results2;

	emb_vector = get_src_vector(img);

	if (emb_vector.size() > 0)
	{
		std::map<std::string, matrix<float, 0, 1>>::iterator it;
		for (it = regfaces.begin(); it != regfaces.end(); ++it)
		{
			//cout << it->first << it->second;
			float res = length(emb_vector[0] - it->second);
			if (res < 0.5)
			{
				//cout << it->first << ":" << res << endl;
				//cout << it->second - it->second  << endl;

				results.push_back(std::make_pair(it->first, res));
			}

		}

		
		std::pair<std::string, float> temp;
		for (int m = 0; m < results.size(); m++)
		{
			for (int n = m+1; n < results.size(); n++)
			{

				if (results[m].second > results[n].second)
				{
					
					temp = results[n];
					results[n] = results[m];
					results[m] = temp;
				}
			}

		}


	}


	return results;

}


std::vector<std::pair<std::string, float>>  search2(std::string img, std::map<std::string, matrix<float, 0, 1>> regfaces)
{
	
	std::vector<matrix<float, 0, 1>> emb_vector;
	std::vector<std::pair<std::string, float>> results;

	emb_vector = get_src_vector(img);

	if (emb_vector.size() > 0)
	{
		std::map<std::string, matrix<float, 0, 1>>::iterator it;
		for (it = regfaces.begin(); it != regfaces.end(); ++it)
		{
			//cout << it->first << it->second;
			float res = length(emb_vector[0] - it->second);
			if (res < 0.5)
			{
				//cout << it->first << ":" << res << endl;
				//cout << it->second - it->second  << endl;

				results.push_back(std::make_pair(it->first, res));
			}

		}

		std::pair<std::string, float> temp;
		for (int m = 0; m < results.size(); m++)
		{
			for (int n = m + 1; n < results.size(); n++)
			{

				if (results[m].second > results[n].second)
				{

					temp = results[n];
					results[n] = results[m];
					results[m] = temp;
				}
			}

		}

	}

	return results;

}

void clear(std::string emdat)
{

	std::map<std::string, matrix<float, 0, 1>> regfaces = load_em(emdat);

	regfaces.clear();
	save_em(regfaces, emdat);

}

void del(std::string id, std::string emdat)
{
	std::map<std::string, matrix<float, 0, 1>> regfaces = load_em(emdat);

	regfaces.erase(id);
	save_em(regfaces, emdat);
}




int compare(std::string path, matrix<rgb_pixel> img, std::vector<rectangle> face)
{
	int flag = 0;
//	double t1 = (double)cvGetTickCount();
	std::vector<matrix<float, 0, 1>> src_vector = get_src_vector(path);
//	t1 = (double)cvGetTickCount() - t1;
//	printf("get_src所用时间 = %gms\n", t1 / ((double)cvGetTickFrequency()*1000.));

	double t2 = (double)cvGetTickCount();
	std::vector<matrix<float, 0, 1>> face_vector = comput_vector(img, face);
//	t2 = (double)cvGetTickCount() - t2;
//	printf("comput_vector所用时间 = %gms\n", t2 / ((double)cvGetTickFrequency()*1000.));
	if (length(src_vector[0] - face_vector[0]) < 0.6)
	{
		flag = 1;
	}
	t2 = (double)cvGetTickCount() - t2;
	printf("comput_vector所用时间 = %gms\n", t2 / ((double)cvGetTickFrequency()*1000.));

	return flag;
}



int FaceID(char* src_name, char*  dst_name)
{
	string src, dst;
	int f = 0;
	src.assign(src_name);
	dst.assign(dst_name);
	

	matrix<rgb_pixel> img;
	matrix<rgb_pixel> img2;

	load_image(img, dst);
	cv::Mat src_img = cv::imread(dst);

	std::vector<rectangle> face = detector(img);

	if (face.size()>0)
	{
		f = compare(src, img, face);
		cv::rectangle(src_img, cv::Rect(face[0].left(), face[0].top(), face[0].width(), face[0].height()), cv::Scalar(0, 0, 255), 2, 8, 0);
		cv::imwrite("dst.jpg", src_img);
	}
	 
	return f;

}



void test()
{
	int ind = 71;
	int noface = 0;
	int right = 0;
	int wrong = 0;
	int all = 0;
	int k = 0;
	init();
	
	char *dir_path = "F:/face/2019.01.18/%d/";

	std::map<int, std::vector<cv::String>> reg_imgs;
	std::map<int, std::vector<cv::String>> test_imgs;

	for (int i = 1; i < ind; i++)
	{
		cv::String per_dir = cv::format(dir_path, i);
		std::vector<cv::String> imgs;
		//cout << per_dir + "*.jpg";
		glob(per_dir + "*.jpg", imgs);
		reg_imgs.insert(make_pair(i, imgs));
		//for (int j=0;j<imgs.size();j++)
		//{
		//	cv::Mat img = cv::imread(imgs[j]);
		//	cv::Mat dst;
		//	cv::resize(img,dst,cv::Size(img.cols/2,img.rows/2));
		//	cv::String dstpath = cv::format("./resize/%d.jpg", k);
		//	cv::imwrite(dstpath,dst);
		//	k++;
		//	cout << k << endl;
		//}
	}

	////注册
	//for (int k = 1; k < ind; k++)
	//{
	//	cout << k << ":" << endl;

	//	cout << reg_imgs[k].at(0) << endl;
	//	string name = cv::format("%d", k);
	//	Registeration(name, reg_imgs[k].at(0), "data.dat");

	//}


	//测试

	for (int k = 1; k < ind; k++)
	{
		cout << k << ":" << endl;
		string name = cv::format("%d", k);
		for (int i = 1; i < reg_imgs[k].size(); i++)
		{
			cout << reg_imgs[k].at(i) << endl;
			
			std::vector<std::pair<std::string, float>> res = search(reg_imgs[k].at(i), "data.dat");
			if(res.size()>0)
			{
				//for (int n = 0; n < res.size(); n++)
				{
			//		cout << res[0].first <<":"<< res[0].second << endl;
					if (res[0].first == name)
					{
						right++;

					}
					else {
						cout << "F:" << res[0].first << ":" << res[0].second << endl;
						wrong++;
						int tempi = atoi(res[0].first.c_str());
						cout << "tempi:" << tempi << endl;
						cout << "tempimg:" << reg_imgs[tempi].at(0) << endl;
						cv::Mat src = cv::imread(reg_imgs[k].at(i));
						cv::Mat dst = cv::imread(reg_imgs[tempi].at(0));
						string src_name = cv::format("./err/%d_%d_src.jpg", k,i);
						string dst_name = cv::format("./err/%d_%d_dst.jpg", k,i);
						cv::imwrite(src_name, src);
						cv::imwrite(dst_name, dst);
					}
				}
			}
			else
			{
				noface++;
			}
			all++;

		}
	}

	//cout << "src:" << reg_imgs[1].at(0) << endl;
	//Mat img = imread(reg_imgs[1].at(0));
	//imshow("src",img);
	//cvWaitKey(0);


	cout << "right:"<<right << endl;
	cout << "wrong:" << wrong << endl;
	cout << "noface:" << noface << endl;

	cout << "all:" << all << endl;
}



void test2()
{
	int ind = 71;
	int noface = 0;
	int right = 0;
	int wrong = 0;
	int all = 0;

	init();

	char *dir_path = "F:/face/2019.01.18/%d/";

	std::map<int, std::vector<cv::String>> reg_imgs;
	std::map<int, std::vector<cv::String>> test_imgs;

	for (int i = 1; i < ind; i++)
	{
		cv::String per_dir = cv::format(dir_path, i);
		std::vector<cv::String> imgs;
		//cout << per_dir + "*.jpg";
		glob(per_dir + "*.jpg", imgs);
		reg_imgs.insert(make_pair(i, imgs));
	}

	std::map<std::string, matrix<float, 0, 1>> regfaces;
	
	//注册
	for (int k = 1; k < ind; k++)
	{
		cout << k << ":" << endl;

		cout << reg_imgs[k].at(0) << endl;
		string name = cv::format("%d", k);

		std::vector<matrix<float, 0, 1>> emb_vector;

		emb_vector = get_src_vector(reg_imgs[k].at(0));
		//regfaces[id] = emb_vector[0];
		if(emb_vector.size()>0)
		regfaces.insert(std::make_pair(name, emb_vector[0]));

	}


	//测试
	for (int k = 1; k < ind; k++)
	{
		cout << k << ":" << endl;
		string name = cv::format("%d", k);
		for (int i = 1; i < reg_imgs[k].size(); i++)
		{
			cout << reg_imgs[k].at(i) << endl;

			std::vector<std::pair<std::string, float>> res = search2(reg_imgs[k].at(i), regfaces);
			if (res.size()>0)
			{
				//for (int n = 0; n < res.size(); n++)
				{
					//		cout << res[0].first <<":"<< res[0].second << endl;
					if (res[0].first == name)
					{
						right++;
					}
					else {
						cout << "F:" << res[0].first << ":" << res[0].second << endl;
						wrong++;
					}
				}
			}
			else
			{
				noface++;
			}
			all++;

		}
	}

	//cout << "src:" << reg_imgs[1].at(0) << endl;
	//Mat img = imread(reg_imgs[1].at(0));
	//imshow("src",img);
	//cvWaitKey(0);


	cout << "right:" << right << endl;
	cout << "wrong:" << wrong << endl;
	cout << "noface:" << noface << endl;

	cout << "all:" << all << endl;
}




int main(int argc, char** argv) try
{

	// We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
	
	
	test();
	
	//init();
	////matrix<rgb_pixel> img;
	////matrix<rgb_pixel> img2;
	////cv::Mat face_img = cv::imread("1.jpg");
	////cv::Mat face_img2 = cv::imread("face2.jpg");
	////cout << "first:"<< endl;
	////Registeration("001","face.jpg","1.dat");
	//////show_emd("1.txt");

	////cout << "second:" << endl;
	////Registeration("002", "face1.jpg", "1.dat");
	//////show_emd("1.txt");
	////
	////cout << "third:" << endl;
	////Registeration("003", "face0.jpg", "1.dat");
	////show_emd("1.dat");

	//Registeration("004", "J01_2019.01.18 14_56_44.jpg", "1.dat");

	////del("004", "1.dat");
	////search("face.jpg", "1.dat");

	//std::map<std::string, float> res = search("src.jpg", "1.dat");

	////search("face1.jpg", "1.dat");
	////	int f1 = FaceID("src.jpg", "src22.jpg");

	//cout << "end"  << endl;

	cin.get();
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}



