import sys
import os
import platform
import numpy as np
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet,preprocess_input
import keras
import keras.backend as K
import cv2

import xml.dom.minidom as xmld

dir = "J:/Project/Eyes/eyes2.0/DetEyes_train/opencvproject/eye/"


dom = xmld.parse("t.xml")
ro = dom.documentElement
img_iteml = ro.getElementsByTagName('image')
print(img_iteml[0].getAttribute('file'))
part = img_iteml[0].getElementsByTagName('part')
print(len(part))

path = img_iteml[0].getAttribute('file')
img = cv2.imread(dir+path)
img_res = cv2.resize(img,(224,224))
points=[]
for p in part:
    #print(p.getAttribute('name'),p.getAttribute('x'),p.getAttribute('y'))
    pt = (int(int(p.getAttribute('x'))*224/img.shape[1]),int(int(p.getAttribute('y'))*224/img.shape[0]))
    points.append(pt)
    cv2.circle(img_res,pt,2,(255),1)
print(points)

# cv2.imshow("src",img_res)
# cv2.waitKey(0)

####数据增强#######################
from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, \
    adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random
import matplotlib.pylab as pl

def randRange(a, b):
    '''
    a utility functio to generate random float values in desired range
    '''
    return random.random() * (b - a) + a


def randomAffine(im):
    '''
    wrapper of Affine transformation with random scale, rotation, shear and translation parameters
    '''
    tform = AffineTransform(scale=(randRange(0.75, 1.3), randRange(0.75, 1.3)),
                            rotation=randRange(-0.25, 0.25),
                            shear=randRange(-0.2, 0.2),
                            translation=(randRange(-im.shape[0] // 10, im.shape[0] // 10),
                                         randRange(-im.shape[1] // 10, im.shape[1] // 10)))
    return warp(im, tform.inverse, mode='reflect')


def randomPerspective(im):
    '''
    wrapper of Projective (or perspective) transform, from 4 random points selected from 4 corners of the image within a defined region.
    '''
    region = 1 / 4
    A = pl.array([[0, 0], [0, im.shape[0]], [im.shape[1], im.shape[0]], [im.shape[1], 0]])
    B = pl.array([[int(randRange(0, im.shape[1] * region)), int(randRange(0, im.shape[0] * region))],
                  [int(randRange(0, im.shape[1] * region)), int(randRange(im.shape[0] * (1 - region), im.shape[0]))],
                  [int(randRange(im.shape[1] * (1 - region), im.shape[1])),
                   int(randRange(im.shape[0] * (1 - region), im.shape[0]))],
                  [int(randRange(im.shape[1] * (1 - region), im.shape[1])), int(randRange(0, im.shape[0] * region))],
                  ])

    pt = ProjectiveTransform()
    pt.estimate(A, B)
    return warp(im, pt, output_shape=im.shape[:2])


def randomCrop(im):
    '''
    croping the image in the center from a random margin from the borders
    '''
    margin = 1 / 10
    start = [int(randRange(0, im.shape[0] * margin)),
             int(randRange(0, im.shape[1] * margin))]
    end = [int(randRange(im.shape[0] * (1 - margin), im.shape[0])),
           int(randRange(im.shape[1] * (1 - margin), im.shape[1]))]
    return im[start[0]:end[0], start[1]:end[1]]

def randomCrop2(im,label):
    '''
    croping the image in the center from a random margin from the borders
    '''
    margin = 1 / 10
    start = [int(randRange(0, im.shape[0] * margin)),
             int(randRange(0, im.shape[1] * margin))]
    end = [int(randRange(im.shape[0] * (1 - margin), im.shape[0])),
           int(randRange(im.shape[1] * (1 - margin), im.shape[1]))]
    lab = []
    for i in range(0,len(label),2):
        if label[i]<start[1] or label[i+1]<start[0] or label[i]>end[1] or label[i+1]>end[0]:
            return im,label
        else:
            lab.append(label[i]-start[1])
            lab.append(label[i+1]-start[0])
    return im[start[0]:end[0], start[1]:end[1]],lab

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
def randomGamma(im):
    '''
    Gamma filter for contrast adjustment with random gamma value.
    '''
    return adjust_gamma(im, gamma=randRange(0.5, 1.5))

def randomGamma2(im):
    '''
    Gamma filter for contrast adjustment with random gamma value.
    '''
    return random_gamma_transform(im, gamma_vari=randRange(0.5, 1.5))

def randomGaussian(im):
    '''
    Gaussian filter for bluring the image with random variance.
    '''
    return gaussian(im, sigma=randRange(0, 5))


def randomFilter(im):
    '''
    randomly selects an exposure filter from histogram equalizers, contrast adjustments, and intensity rescaler and applys it on the input image.
    filters include: equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid, gaussian
    equalize_adapthist, equalize_hist,'''
    Filters = [ adjust_log, adjust_sigmoid, randomGamma,
               ]
    filt = random.choice(Filters)
    return filt(im)


def randomNoise(im):
    '''
    random gaussian noise with random variance.
    '''
    var = randRange(0.001, 0.01)
    return random_noise(im, var=var)

def blur(img):
    img = cv2.blur(img, (3, 3))
    return img

def add_noise(img):
    for i in range(500): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 200
    return img

def augment(im, Steps = [blur,add_noise,randomGamma2]):
    '''
    image augmentation by doing a sereis of transfomations on the image.
    '''
    for step in Steps:
        if np.random.random() < 0.5:
            im = step(im)
    return im


def get_data(xml_name,dir):
    dom = xmld.parse(xml_name)
    ro = dom.documentElement
    img_iteml = ro.getElementsByTagName('image')
    data = []
    for im in img_iteml:
        path = im.getAttribute('file')
        img = cv2.imread(dir + path)
        #img_res = cv2.resize(img, (224, 224))

        part = im.getElementsByTagName('part')
        #print(len(part))
        if len(part)==16:
            points = []
            for p in part:
                # print(p.getAttribute('name'),p.getAttribute('x'),p.getAttribute('y'))
                pt = (int(p.getAttribute('x')), int(p.getAttribute('y')))
                points.append(pt[0])
                points.append(pt[1])
                #cv2.circle(img, pt, 2, (0,255,0), 1)

            # cv2.imshow("src", img_res)
            #
            # img, label = resize_data(img, points)
            # for i in range(0,len(label),2):
            #     cv2.circle(img, (label[i],label[i+1]), 2, (0, 255, 0), 1)
            # c_img,c_point = randomCrop2(img,label)
            # for i in range(0,len(c_point),2):
            #     cv2.circle(c_img, (c_point[i],c_point[i+1]), 2, (0, 255, 0), 1)
            # cv2.imshow("src1", c_img)
            # cv2.waitKey(0)
            data.append((dir + path,points))

    return data

def resize_data(im,label):
    img = cv2.resize(im, (224, 224))
    lab = []
    for i in range(0,len(label),2):
        lab.append(int(label[i] * 224 / im.shape[1]))
        lab.append(int(label[i+1] * 224 / im.shape[0]))
    #print(label)
    return img,lab

def data_gen(data,batch_size):
    while True:
        imgs = []
        labels = []
        batch = 0
        for d in data:
            img = cv2.imread(d[0])
            img = augment(img)
            img, label = randomCrop2(img, d[1])  # 数据增强
            img, label= resize_data(img, label)
            img = img/128 - 1
            #img = preprocess_input(img)
            imgs.append(img)
            labels.append(label)
            batch += 1
            if batch%batch_size == 0:
                imgs = np.array(imgs)
                labels = np.array(labels)
                yield (imgs,labels)
                imgs = []
                labels = []
                batch = 0

def get_val_data(xml_name,dir):
    dom = xmld.parse(xml_name)
    ro = dom.documentElement
    img_iteml = ro.getElementsByTagName('image')
    imgs = []
    labels = []
    for im in img_iteml:
        path = im.getAttribute('file')
        img = cv2.imread(dir + path)
        img = cv2.resize(img, (224, 224))
        img = img / 128 - 1

        part = im.getElementsByTagName('part')
        #print(len(part))
        if len(part) == 16:
            imgs.append(img)
            points = []
            for p in part:
                # print(p.getAttribute('name'),p.getAttribute('x'),p.getAttribute('y'))
                pt = (int(int(p.getAttribute('x')) * 224 / img.shape[1]), int(int(p.getAttribute('y')) * 224 / img.shape[0]))
                points.append(pt[0])
                points.append(pt[1])

            labels.append(points)
        #print(points)
        # cv2.imshow("src", img_res)
        # cv2.waitKey(10)

    return np.array(imgs[0:16]),np.array(labels[0:16])


#print(train_data_gen(d,16))


def build_model(lr=0.002):
    model_pre =  MobileNet(input_shape=(224, 224, 3), weights=None, include_top=False)
    #model_pre.load_weights("G:/keras_weights/mobilenet_1_0_224_tf_no_top.h5")
    x = model_pre.output
    #x = model_pre.get_layer("conv_pw_5_relu").output
    x = Reshape((1, 1, -1))(x)
    x = Conv2D(32,(1,1),strides=(1, 1), padding="same")(x)
    #x = Flatten()(x)
    #x = Activation('relu')(x)
    y = Reshape((32,))(x)
    #y = Dense(32, activation='relu')(x)
    model = Model(model_pre.input,y)
    model.compile(Adam(lr),loss='mean_squared_error')
    return model


model1 = build_model(0.00002)#初始0.002，过度0.0002，最终调整学习率0.00002
model1.summary()
model1.load_weights('save.h5')

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(save_best_only=True,monitor='val_loss',filepath='save.h5')

data = get_data("train2.xml",dir)
val_data = get_data("test.xml",dir)
print(data[0][0],data[0][1])
model1.fit_generator(data_gen(data,16),steps_per_epoch=500,
                     shuffle=True,validation_data=data_gen(val_data,16),
                     validation_steps=100,epochs=100,verbose=1,callbacks=[checkpoint])


def test(path):
    inputs = []
    src1 = cv2.imread(path)
    src = cv2.resize(src1,(224,224))
    img = src/128 -1
    inputs.append(img)
    inputs = np.array(inputs)
    resault = model1.predict(inputs)
    res = resault[0]
    print(len(res))

    for i in range(1,len(res),2):
        cv2.circle(src1, (int(res[i-1]*src1.shape[1]/224),int(res[i]*src1.shape[0]/224)), 2, (0,255,0), 1)
    # cv2.circle(src, (res[8], res[9]), 2, (0, 255, 0), 1)
    # cv2.circle(src, (res[16],res[17]), 2, (0,255,0), 1)
    # cv2.circle(src, (res[24], res[25]), 2, (0, 255, 0), 1)
    # src1 = randomGamma2(src1)
    # src1 = blur(src1)
    # src1 = add_noise(src1)
    cv2.imshow("res",src1)
    cv2.waitKey(0)
    return res[0]

#test('85.jpg')
path = 'J:/Project/Eyes/eyes2.0/DetEyes_train/opencvproject/eye/train1/'
img_list = os.listdir(path)
for p in img_list:
    print(path+p)
    test(path+p)
test('J:/Project/Eyes/eyes2.0/DetEyes_train/opencvproject/eye/train2/168.jpg')