import  xml.dom.minidom
import numpy as np

def readxml(path):
    data = []
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement
    item1 = root.getElementsByTagName('F1')
    dat1 = item1[0].firstChild.data
    data.append(dat1)

    item2 = root.getElementsByTagName('F2')
    dat2 = item2[0].firstChild.data
    data.append(dat2)

    item3 = root.getElementsByTagName('F3')
    dat3 = item3[0].firstChild.data
    data.append(dat3)

    return data

# da=[]
# data = readxml('J:/study/tensorflow_learnning/dat.xml')
# data2 = readxml('J:/study/tensorflow_learnning/dat.xml')
# da.append(data)
# da.append(data2)
#
# x_dat = np.float32(da)
# print(x_dat)