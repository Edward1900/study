from xml.etree.ElementTree import Element, SubElement, tostring
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import cv2

def save_xml_obj(xml_name,jpg_name,obj_name,wh_rect):
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'obj'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = jpg_name

    node_pathname = SubElement(node_root, 'path')
    node_pathname.text = 'H:'

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'Unknown'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '{}'.format(wh_rect[0])

    node_height = SubElement(node_size, 'height')
    node_height.text = '{}'.format(wh_rect[1])

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    node_seg = SubElement(node_root, 'segmented')
    node_seg.text = '0'

    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = obj_name
    node_pose = SubElement(node_object, 'pose')
    node_pose.text = 'Unspecified'
    node_truncated = SubElement(node_object, 'truncated')
    node_truncated.text = '0'

    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_bndbox = SubElement(node_object, 'bndbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = '{}'.format(wh_rect[2])
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = '{}'.format(wh_rect[3])
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = '{}'.format(wh_rect[4])
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = '{}'.format(wh_rect[5])

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行

    f = open(xml_name,'w')
    dom1 = parseString(xml)
    f.write(dom1.toprettyxml(indent = ''))
    f.close()
    #print (xml)

def det_mouth(img, mouth_det):
    src = cv2.resize(img, (int(img.shape[1]/10), int(img.shape[0]/10)))
    gray = cv2.equalizeHist(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY))
    mouth = mouth_det.detectMultiScale(gray, 1.1, 2, 0, (50, 50))
    wh_rect = []
    for (x, y, w, h) in mouth:
        #print(x*10, y*10)
        wh_rect.append(img.shape[1])
        wh_rect.append(img.shape[0])
        wh_rect.append(x * 10)
        wh_rect.append(y * 10)
        wh_rect.append(x * 10 + w * 10)
        wh_rect.append(y * 10 + h * 10)
        #roi = img[y * 10:y * 10 + h * 10, x * 10:x * 10 + w * 10]
        #cv2.rectangle(img, (int(x*10), int(y*10)), (int(x*10+w*10), int(y*10+h*10)), (0, 255, 0), 1)
        return wh_rect
    return wh_rect


r = [10,10,1,2,3,4]
save_xml_obj("2.xml","2.jpg","m",r)