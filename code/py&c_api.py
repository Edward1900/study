# import ftry as ft
#
# model = ft.preload()
# ft.pretest(model)
# ft.pretest(model)
# ft.pretest(model)
# ft.pretest(model)
# ft.pretest(model)
# ft.pretest(model)

# �����
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

# ���ͼƬָ���ߴ�
target_size = (229, 229) #fixed size for InceptionV3 architecture

# Ԥ�⺯��
# ���룺model��ͼƬ��Ŀ��ߴ�
# �����Ԥ��predict
def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)   #���Է�������������ֻ��һ��ͼƬ������preds[0]�ǽ��
  return preds[0]


def preload():
    global model
    model = load_model('J:/keras_project/fine-tuning/kmodel.h5')

def pretest(path):
  # ����ͼƬ
  print(path)
  img = Image.open(path)
  preds = predict(model, img, target_size)
  print(np.max(preds))
  return np.max(preds)

preload()
pretest('J:/keras_project/fine-tuning/test.jpg')