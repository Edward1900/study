from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
#from matplotlib import pyplot as plt

from models.keras_ssdmob import build_model
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from keras.preprocessing import image
import cv2

img_height = 224 # Height of the input images
img_width = 224 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 20 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range,iou_threshold=0.1)

# 2: Optional: Load some weights
weights_path = 'I:/ssd_keras/ssdm_epoch-09_loss-3.3463_val_loss-2.4732.h5'

model.load_weights(weights_path, by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# TODO: Set the path to the weights you want to load.
# TODO: Set the path to the `.h5` file of the model to be loaded.
#model_path = 'ssdm.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
#
# K.clear_session() # Clear previous models from memory.
#
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'compute_loss': ssd_loss.compute_loss})


import time
orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
img_path = 'I:/ssd_keras/test/1.jpg'

imge = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
orig_images.append(imge)
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img)
input_images.append(img)
input_images = np.array(input_images)

start1 = time.perf_counter()
y_pred1 = model.predict(input_images)
dur1 = time.perf_counter() - start1
print("dur1:", dur1)

start = time.perf_counter()
y_pred = model.predict(input_images)
dur = time.perf_counter() - start
print("dur:", dur)

# y_pred_decoded = decode_detections(y_pred,
#                                    confidence_thresh=0.8,
#                                    iou_threshold=0.3,
#                                    top_k=200,
#                                    normalize_coords=normalize_coords,
#                                    img_height=img_height,
#                                    img_width=img_width)
print(y_pred)
confidence_threshold = 0.5


y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])

#model.save("mouthm.h5")

# Display the image and draw the predicted boxes onto it.
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background',
           'aeroplane', 'bicycle', 'tooth', 'boat',
           'bottle', 'throat', 'palate', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


plt.figure(figsize=(20,12))
plt.imshow(orig_images[0])

current_axis = plt.gca()

for box in y_pred_thresh[0]:
    # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
    xmin = box[2] * orig_images[0].shape[1] / img_width
    ymin = box[3] * orig_images[0].shape[0] / img_height
    xmax = box[4] * orig_images[0].shape[1] / img_width
    ymax = box[5] * orig_images[0].shape[0] / img_height
    color = colors[int(box[0])*2]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

plt.show()


def get_images0(img_path_list):
    orig_images = []  # Store the images here.
    roi_images = []
    input_images = []  # Store resized versions of the images here.
    x = []
    y = []
    for i in range(len(img_path_list)):
        #print(img_path_list[i])
        src = cv2.imread(img_path_list[i])
        img_roi = src
        orig_images.append(src)
        roi_images.append(img_roi)
        x.append(0)
        y.append(0)

        imge = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
        #img = image.load_img(img_path, target_size=(img_height, img_width))
        img = cv2.resize(imge, (img_height, img_width))
        img = image.img_to_array(img)
        input_images.append(img)

    return orig_images, roi_images, input_images, x, y

import os

def test_imgs(path):
    flag = 0
    start = time.perf_counter()
    path_list = []
    path_list1 = os.listdir(path)
    for i in range(len(path_list1)):
        path_list.append(path+path_list1[i])
        print(path+path_list1[i])


    orig_images, roi_images, input_images, x, y = get_images0(path_list)
    input_images = np.array(input_images)

    if(input_images.size>0):
        y_pred = model.predict(input_images)

        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh)

        # model.save("m.h5")
        #
        # Display the image and draw the predicted boxes onto it.

        for i in range(len(input_images)):
            for box in y_pred_thresh[i]:
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin = box[2] * roi_images[i].shape[1] / img_width + x[i]
                ymin = box[3] * roi_images[i].shape[0] / img_height + y[i]
                xmax = box[4] * roi_images[i].shape[1] / img_width + x[i]
                ymax = box[5] * roi_images[i].shape[0] / img_height + y[i]
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                #cv2.circle(orig_images[0], (int(xmin), int(ymin)), 5, (0,0,255),3)
                # cv2.rectangle(orig_images[i], (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)
                # cv2.putText(orig_images[i], label, (int(xmin), int(ymin)), 1, 1, (0, 255, 0), 1)
                if box[0]==6 and box[1]>0.5:
                    cv2.rectangle(orig_images[i], (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
                    cv2.putText(orig_images[i], label, (int(xmin), int(ymin-5)), 1, 1.3, (0, 255, 0), 2)
                    flag = 1
                if box[0]==7:
                    cv2.rectangle(orig_images[i], (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)
                    cv2.putText(orig_images[i], label, (int(xmin), int(ymin-5)), 1, 1.3, (255, 0, 0), 2)
                    flag = 2
                if box[0]==3:
                    cv2.rectangle(orig_images[i], (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 3)
                    cv2.putText(orig_images[i], label, (int(xmin), int(ymin-5)), 1, 1.3, (255, 255, 0), 2)
                    flag = 3

            if not (os.path.exists(path + 'res')):
                os.makedirs(path + 'res')
            name = path + 'res/'+ path_list1[i]

            #name = '{}.jpg'.format(i)
            if flag == 1 or flag == 2 or flag == 3:
                cv2.imwrite(name, orig_images[i])
                flag = 0

        #cv2.waitKey(0)
        dur = time.perf_counter() - start
        print("dur:", dur)



test_imgs('F:/')
#test('I:/ssd_keras_release/examples/get/2019-02-22-07-54-06m.jpg')
#print(resault)

