import sys
import platform
old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
sys.stderr = old_stderr
import random
from keras import backend as K
from keras.preprocessing.image import img_to_array,array_to_img
from scipy.ndimage import affine_transform
IMG_SIZE = 384
img_shape    = (384,384,1) # The image shape used by the model
from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    Lambda, MaxPooling2D, Reshape
from keras.models import Model


def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    y = Add()([x, y])  # Add the bypass connection
    y = Activation('relu')(y)
    return y


def build_model(lr, l2, activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = Input(shape=img_shape)  # 384x384x1
    x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
    for _ in range(4): x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
    for _ in range(4): x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
    for _ in range(4): x = subblock(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
    for _ in range(4): x = subblock(x, 128, **kwargs)

    x = GlobalMaxPooling2D()(x)  # 512
    branch_model = Model(inp, x)

    ############
    # HEAD MODEL
    ############
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])
    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = Model([img_a, img_b], x)
    model.compile(optim, loss='binary_crossentropy', metrics=['acc'])
    return model, branch_model, head_model


model, branch_model, head_model = build_model(64e-5, 0)
head_model.summary()

from keras.utils import plot_model
from os.path import isfile
from PIL import Image as pil_image

# plot_model(head_model, to_file='head-model.png')
# pil_image.open('head-model.png')
branch_model.summary()
# plot_model(branch_model, to_file='branch-model.png')
# img = pil_image.open('branch-model.png')
# img.resize([x//2 for x in img.size])


def load_img(path, grayscale=True):
    if grayscale:
        src = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(src,(IMG_SIZE,IMG_SIZE))
        img = np.array(img, dtype="float") / 127.5
        img -=1
    else:
        src = cv2.imread(path)
        img = cv2.resize(src, (IMG_SIZE, IMG_SIZE))
        img = np.array(img,dtype="float") / 127.5
        img -= 1
    return img

a  = ['a','b',1]
a2  = ['a1','b1',0]
da = []
da.append(a)
da.append(a2)
#print(da[0])
d1,d2,d3 = da[1]
#print(d1,d2,d3 )

def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))

def get_lr(model):
    return K.get_value(model.optimizer.lr)

def generateData(batch_size,data):
    #print 'generateData...'
    while True:
        train_data_a = []
        train_data_b = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url_a,url_b,label = data[i]
            batch += 1
            img_a = load_img( '../input/train/' + url_a)
            img_a = img_to_array(img_a)

            img_b = load_img('../input/train/' + url_b)
            img_b = img_to_array(img_b)

            train_data_a.append(img_a)
            train_data_b.append(img_b)

            train_label.append(label)
            if batch % batch_size==0:
                #print 'get enough bacth!\n'
                train_data_a = np.array(train_data_a)
                train_data_b = np.array(train_data_b)

                train_label = np.array(train_label)
                yield ([train_data_a,train_data_b],train_label)
                train_data_a = []
                train_data_b = []
                train_label = []
                batch = 0

from keras.callbacks import ModelCheckpoint

model_checkpoint = ModelCheckpoint(filepath='model_epoch-{epoch:02d}_loss-{loss:.4f}.h5',
                                   monitor='loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
callback = [model_checkpoint]


# train_data = []
# train_pair = pd.read_csv("train_pair.csv")
# for i in train_pair.index:
#     a = train_pair.iloc[i]['ImageA']
#     b = train_pair.iloc[i]['ImageB']
#     l = train_pair.iloc[i]['Label']
#     train_data.append([a,b,l])
#
# random.shuffle(train_data)
# print(train_data[0:32])
# print(len(train_data))
# batch_size = 32

model.load_weights('model_epoch-81_loss-0.0095.h5')
# history = model.fit_generator(generator=generateData(batch_size,train_data),steps_per_epoch=len(train_data)//batch_size,initial_epoch=60,epochs=
#                     300,verbose=1,callbacks=callback).history
#

def get_feature(img):
    x = img_to_array(img)
    a = np.expand_dims(x, axis=0)
    feature = branch_model.predict(a)
    return feature

def prepareFeature(data, dataset):
    print("Preparing features")
    #features = []
    print("data.index.size:",data.index.size)
    features = np.zeros((data.index.size, 1, 512))
    count = 0
    for fig in data['Image']:
        # load images
        img = load_img("../input/" + dataset + "/" + fig)
        feature = get_feature(img)
        # print('feature.shape:',feature.shape)
        features[count] = feature
        if (count % 500 == 0):
            print("Processing feature: ", count + 1, ", ", fig)
        count += 1
        features = np.array(features)
    return features


def get_score(feature_a,feature_b):
    score = head_model.predict([feature_a,feature_b])
    return score[0][0]



img_a = load_img( '../input/train/52e3557b3.jpg')
img_b = load_img( '../input/train/a2b270cb3.jpg')

feature_a = get_feature(img_a)
feature_b = get_feature(img_b)
score1 = get_score(feature_a,feature_b)
#print(feature_a)
print(score1)


train_df = pd.read_csv("../input/train_sub.csv")
# train_features = prepareFeature(train_df, "train")
# #print(train_features[0:5])
# np.save('Features',train_features)

train_features = np.load('Features.npy')

test_df = pd.read_csv("../input/val.csv")
# test = os.listdir("../input/test/")
# print(len(test))
# col = ['Image']
# test_df = pd.DataFrame(test, columns=col)
# test_df['Id'] = ''
#
test_features = prepareFeature(test_df, "train")  #val
#test_features = prepareFeature(test_df, "test")  #test
#
file= open('res.txt', 'w')
all_res_list = []
all_score_list = []
cnt = 0
for f1 in test_features:
    f1_res = []
    for f2 in train_features:
        score = get_score(f1,f2)
        # print('...')
        f1_res.append(score)
    fnp = np.array(f1_res)
    index_img = fnp.argsort()[::-1]
    id_list = []
    score_list = []
    for i in index_img:
        name = train_df.iloc[i]['Id']
        if name not in id_list:
            id_list.append(name)
            score_list.append(fnp[i])
    all_res_list.append(id_list)
    all_score_list.append(score_list)
    file.write(str(id_list))
    file.write('\n')
    file.write(str(score_list))
    file.write('\n')
    print('working....',cnt)
    cnt+=1
file.close()





for th in np.arange(0.2, 0.8, 0.01):
    temp_list = all_res_list.copy()
    for i in range(len(temp_list)):
        for j in range(len(temp_list[i])):
            if all_score_list[i][j]<th:
                temp_list[i].insert(j,'new_whale')
                break
    score = 0
    for i in range(len(temp_list)):
        for j in range(5):
            if temp_list[i][j] == test_df['Id'][i]:
                score += (5 - j)/5
                break
    score = score/len(temp_list)
    print('score:',score)

# for i in range(len(all_res_list)):
#     test_df.loc[i, 'Id'] = ' '.join(all_res_list[i][0:5])
#
# test_df.head(10)
# test_df.to_csv('submission.csv', index=False)
