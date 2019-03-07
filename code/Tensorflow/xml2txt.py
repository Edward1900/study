import xml.dom.minidom
import numpy as np
import os
import cv2

def readxml(path,path_img):

    img = cv2.imread(path_img)
    data = []
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement
    item1 = root.getElementsByTagName('filename')
    dat1 = item1[0].firstChild.data
    data.append(dat1)

    item2 = root.getElementsByTagName('xmin')
    dat2 = item2[0].firstChild.data
    data.append(dat2)

    item3 = root.getElementsByTagName('ymin')
    dat3 = item3[0].firstChild.data
    data.append(dat3)

    item4 = root.getElementsByTagName('xmax')
    dat4 = item4[0].firstChild.data
    data.append(dat4)

    item5 = root.getElementsByTagName('ymax')
    dat5 = item5[0].firstChild.data
    data.append(dat5)

    x = np.int32(data[1])
    y = np.int32(data[2])
    w = np.int32(data[3])-np.int32(data[1])
    h = np.int32(data[4])-np.int32(data[2])
    roi = img[y:y + h, x:x + w]
    pstr = "pos/" + data[0] + " 1 0 0 {} {} \n".format(w,h)

    return pstr,roi

def main():

    file_name = "J:\\tt\\xml\\"
    file_name2 = "J:\\tt\\img\\"
    out_path = "J:\\tt\\pos\\"
    f1 = os.listdir(file_name)
    #f2 = os.listdir(file_name2)

    f = open(out_path+"pos.txt",'w')
    for i in range(len(f1)):
        f_n1 = file_name + f1[i]
        f2 = f1[i].split(".")
        print(f2)
        f_n2 = file_name2 + f2[0]+".jpg"
        print(f_n2)
        str,roi = readxml(f_n1,f_n2)
        f.writelines(str)
        cv2.imwrite(out_path+f2[0]+".jpg",roi)


if __name__ == '__main__':
    main()