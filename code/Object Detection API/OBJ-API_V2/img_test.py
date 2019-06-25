import os
import time
import cv2
import numpy as np
import tensorflow as tf
import cv2
import numpy as np
import tensorflow as tf

test_image_dir = './data/train/jpgs/'

class TOD(object):

    def __init__(self):
        self.PATH_TO_CKPT = './out/frozen_inference_graph.pb'

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
        image_np_expanded = image_np_expanded.astype('float32')
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
                cv2.rectangle(image,(int(result[i][1]),int(result[i][0])),(int(result[i][3]),int(result[i][2])),((result1[i]/3)*255,255,(result1[i]/3)*255),2)
        used_time = time.time() - start_time
        print('model_interpreter_time:{}'.format(used_time))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Resault", image)
        cv2.waitKey(800)


file_list = os.listdir(test_image_dir)

# 遍历文件
detecotr = TOD()
while True:
    for file in file_list:
        print('=========================')
        full_path = os.path.join(test_image_dir, file)
        print('full_path:{}'.format(full_path))
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300, 300))
        detecotr.detect(image)