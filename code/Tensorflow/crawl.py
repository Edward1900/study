import re
import requests
from urllib import error
from bs4 import BeautifulSoup
import os

num = 0
numPicture = 0
file = 'J://study//face-swap//data//RobertDowney'
List = []


def Find(url):
    global List
    print('正在检测图片总数，请稍等.....')
    t = 0
    i = 1
    s = 0
    while t < 1000:
        Url = url + str(t)
        try:
            Result = requests.get(Url, timeout=7)
        except BaseException:
            t = t + 60
            continue
        else:
            result = Result.text
            pic_url = re.findall('"objURL":"(.*?)",', result, re.S)  # 先利用正则表达式找到图片url
            s += len(pic_url)
            if len(pic_url) == 0:
                break
            else:
                List.append(pic_url)
                t = t + 60
    return s


def recommend(url):
    Re = []
    try:
        html = requests.get(url)
    except error.HTTPError as e:
        return
    else:
        html.encoding = 'utf-8'
        bsObj = BeautifulSoup(html.text, 'html.parser')
        div = bsObj.find('div', id='topRS')
        if div is not None:
            listA = div.findAll('a')
            for i in listA:
                if i is not None:
                    Re.append(i.get_text())
        return Re


def dowmloadPicture(html, keyword):
    global num
    # t =0
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)  # 先利用正则表达式找到图片url
    print('找到关键词:' + keyword + '的图片，即将开始下载图片...')
    for each in pic_url:
        print('正在下载第' + str(num + 1) + '张图片，图片地址:' + str(each))
        try:
            if each is not None:
                pic = requests.get(each, timeout=7)
            else:
                continue
        except BaseException:
            print('错误，当前图片无法下载')
            continue
        else:
            string = file + r'\\' + keyword + '_' + str(num) + '.jpg'
            fp = open(string, 'wb')
            fp.write(pic.content)
            fp.close()
            num += 1
        if num >= numPicture:
            return


if __name__ == '__main__':  # 主函数入口
    # #tm = int(input('请输入每类图片的下载数量 '))
    # numPicture = 200
    # line_list = ['RobertDowney']
    # # with open('./name.txt', encoding='utf-8') as file:
    # #     line_list = [k.strip() for k in file.readlines()]  # 用 strip()移除末尾的空格
    #
    # for word in line_list:
    #     url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&pn='
    #     tot = Find(url)
    #     Recommend = recommend(url)  # 记录相关推荐
    #     print('经过检测%s类图片共有%d张' % (word, tot))
    #     file = word + '文件'
    #     y = os.path.exists(file)
    #     if y == 1:
    #         print('该文件已存在，请重新输入')
    #         file = word + '文件夹2'
    #         os.mkdir(file)
    #     else:
    #         os.mkdir(file)
    #     t = 0
    #     tmp = url
    #     while t < numPicture:
    #         try:
    #             url = tmp + str(t)
    #             result = requests.get(url, timeout=10)
    #             print(url)
    #         except error.HTTPError as e:
    #             print('网络错误，请调整网络后重试')
    #             t = t + 60
    #         else:
    #             dowmloadPicture(result.text, word)
    #             t = t + 60
    #     numPicture = numPicture + numPicture
    #
    # print('当前搜索结束，感谢使用')

    import glob
    import dlib
    import cv2

    paths = glob.glob("RobertDowney/*.jpg")
    facedet = dlib.get_frontal_face_detector()
    print(paths)
    i=0
    for p in paths:
        try:
            src = dlib.load_rgb_image(p)
            rects = facedet(src)
            print(rects)
            for r in rects:
                print(r)
                IMG = src[r.tl_corner().y : r.br_corner().y,r.tl_corner().x : r.br_corner().x,:]
                print(r.tl_corner() , r.br_corner())
                IMG = dlib.resize_image(IMG,256,256)
                dlib.save_image(IMG, "dst/dst{}.jpg".format(i))
                i = i+1
        except BaseException:
            print('错误，当前图片无法读取')
            continue
    # IMG_BGR = cv2.cvtColor(src,cv2.COLOR_RGB2BGR)
    # IMG = src[95:244,45:194,:]
    # dlib.save_image(IMG,"dst.jpg")

