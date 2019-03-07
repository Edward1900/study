import requests
import re


def getContent(url):#使用requests.get获取知乎首页的内容
    r = requests.get(url)#request.get().content是爬到的网页的全部内容
    return r.content

def get_imgs(baseurl,name,password,page_num):#post需要的表单数据，类型为字典
    login_data = {
            'account': name,
            'pwd': password,

    }
    #设置头信息
    headers_base = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4,zh-TW;q=0.2',
        'Connection': 'keep-alive',
        'Host': 'admin.****.net',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.130 Safari/537.36',
        'Referer': 'http://****.net/',
    }
#使用seesion登录，这样的好处是可以在接下来的访问中可以保留登录信息
    session = requests.session()#登录的URL
    baseurl += "/cj/sysuser/login.do"#requests 的session登录，以post方式，参数分别为url、headers、data
    content = session.post(baseurl, headers = headers_base, data = login_data)#成功登录后输出为 {"r":0,　　#"msg": "\u767b\u9646\u6210\u529f"　　#}
    print(content.text)
    #再次使用session以get去访问，一定要设置verify = False，否则会访问失败
    #s = session.get("http://******/list.do", verify = False)
    f = open('img_url.txt', 'w')
    toal_num = 0
    for i in range(page_num):
        page = '{}'.format(i+1)
        check_data = {
            'Date': '2018-11-01',
            'resId' : '1',
            'pindex'   : page ,
        }
        s = session.post("http://******/list.do",data = check_data, verify=False)     #2017-08-03
        html = s.text
        photo_list = re.findall(r'<a class="err_url" onclick="showImg[(](.*?)[)]">异常</a>', html)
        ptotalpages = re.findall(r'var ptotalpages = "(.*?)"', html)
        if int(page)>int(ptotalpages[0]):
            break
        print("总页数：",ptotalpages[0])
        print("当前页：", page)
        print(photo_list)
        #print(photo_list[0].split("'"))
        toal_num = toal_num + len(photo_list)
        for m in range(len(photo_list)):
            f.writelines(photo_list[m].split("'")[1]+'\n')
    print("总条目：", toal_num)
    f.close()
    #print(s.text)#.encode('utf-8'))#get得到的数据需要encode
    #f = open('t.txt', 'w')#'w'是对str,'wb+'是对byte
    #f.write(s.text)#.encode('utf-8'))

def down_image(url,file_name):
    img = requests.get(url)
    print('开始保存图片')
    f = open(file_name, 'ab')
    f.write(img.content)
    print(file_name, '图片保存成功！')
    f.close()

def down_images(img_txt,out_dir):
    with open(img_txt,'r') as f:
        for line in f:
            url = line.strip()
            print(url)
            name = url.split('/')[-1]
            image_name = out_dir+name
            down_image(url,image_name)



get_imgs("http://*****.net","name","pass",20)
down_images('img_url.txt',"./imgs/")



