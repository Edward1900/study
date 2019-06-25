# -*- coding:utf-8 -*-
import os
import time
import cv2
import numpy as np
import tensorflow as tf


test_image_dir = './test_images/'

class TOD(object):

    def __init__(self):
        self.PATH_TO_CKPT = './frozen_inference_graph.pb'
        self.NUM_CLASSES = 3
        self.sess,self.image_tensor,self.output = self._load_model()


    def _load_model(self):
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                sess = tf.Session(graph=detection_graph)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                output = [boxes,scores,classes,num_detections]
                return sess,image_tensor,output


    def detect(self, image):
        start_time = time.time()
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            self.output,
            feed_dict={self.image_tensor: image_np_expanded})
        result = np.squeeze(boxes) * 300
        result1 = np.squeeze(classes)
        result2 = np.squeeze(scores)
        print('result0:{}'.format(result))
        print('result1:{}'.format(scores))
        print('result2:{}'.format(classes))
        print('result3:{}'.format(num_detections))
        for i in range(len(result)):
            if(result2[i]>0.3):
                cv2.rectangle(image,(int(result[i][1]),int(result[i][0])),(int(result[i][3]),int(result[i][2])),((result1[i]/90)*255,(result1[i]/90)*255,(result1[i]/90)*255),2)
        used_time = time.time() - start_time
        print('model_interpreter_time:{}'.format(used_time))
        rgb_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Resault", rgb_img)
        cv2.waitKey(0)


file_list = os.listdir(test_image_dir)

# 遍历文件
detecotr = TOD()
for file in file_list:
    print('=========================')
    full_path = os.path.join(test_image_dir, file)
    print('full_path:{}'.format(full_path))
    image = cv2.imread(full_path)
    image = cv2.resize(image, (300, 300))
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detecotr.detect(rgb_img)


# model_path = "./model/quantize_frozen_graph.tflite"
model_path = "./detect.tflite"
# # Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
#
# Get input and output tensors.
input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))

# with tf.Session( ) as sess:
if 1:
    file_list = os.listdir(test_image_dir)

    model_interpreter_time = 0
    start_time = time.time()
    # 遍历文件
    for file in file_list:
        print('=========================')
        full_path = os.path.join(test_image_dir, file)
        print('full_path:{}'.format(full_path))
        img = cv2.imread(full_path)
        res_img = cv2.resize(img, (300, 300))
        rgb_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(rgb_img, axis=0)
        image_np_expanded = image_np_expanded.astype('float32')  # 类型也要满足要求
        image_np_expanded = (image_np_expanded-128) / 128
         # 填装数据
        model_interpreter_start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], image_np_expanded)

        # 注意注意，我要调用模型了
        start_time1 = time.time()
        interpreter.invoke()
        used_time1 = time.time() - start_time1
        print('model_interpreter_time:{}'.format(used_time1))

        output_data0 = interpreter.get_tensor(output_details[0]['index'])
        output_data1 = interpreter.get_tensor(output_details[1]['index'])
        output_data2 = interpreter.get_tensor(output_details[2]['index'])
        output_data3 = interpreter.get_tensor(output_details[3]['index'])
        model_interpreter_time += time.time() - model_interpreter_start_time

        # 出来的结果去掉没用的维度
        result = np.squeeze(output_data0)*300
        result1 = np.squeeze(output_data1)
        result2 = np.squeeze(output_data2)
        print('result0:{}'.format(result))
        print('result1:{}'.format(output_data1))
        print('result2:{}'.format(output_data2))
        print('result3:{}'.format(output_data3))
        # print('result:{}'.format(sess.run(output, feed_dict={newInput_X: image_np_expanded})))
        for i in range(len(result)):
            if(result2[i]>0.3):
                cv2.rectangle(res_img,(int(result[i][1]),int(result[i][0])),(int(result[i][3]),int(result[i][2])),((result1[i]/90)*255,(result1[i]/90)*255,(result1[i]/90)*255),2)
        cv2.imshow("src", res_img)
        cv2.waitKey(0)

        # 输出结果是长度为10（对应0-9）的一维数据，最大值的下标就是预测的数字
        #print('result:{}'.format((np.where(result == np.max(result)))[0][0]))
    used_time = time.time() - start_time
    print('used_time:{}'.format(used_time))
    print('model_interpreter_time:{}'.format(model_interpreter_time))


