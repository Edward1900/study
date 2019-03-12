#part of the code is from https://github.com/maciejkula/triplet_recommendations_keras
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import glob
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image
import cv2

import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Embedding, merge, Conv2D, Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalMaxPool2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.applications.resnet50 import ResNet50


#样本数据生成，每组样本有三个样本组成，anchor样本，还有与之对应的正样本，和负样本,这里样本的选取是随机选取。
class sample_gen(object):
    def __init__(self, file_class_mapping, other_class = "new_whale"):
        self.file_class_mapping= file_class_mapping
        self.class_to_list_files = defaultdict(list)
        self.list_other_class = []
        self.list_all_files = list(file_class_mapping.keys())
        self.range_all_files = list(range(len(self.list_all_files)))

        for file, class_ in file_class_mapping.items():
            if class_ == other_class:
                self.list_other_class.append(file)
            else:
                self.class_to_list_files[class_].append(file)

        self.list_classes = list(set(self.file_class_mapping.values()))
        self.range_list_classes= range(len(self.list_classes))
        self.class_weight = np.array([len(self.class_to_list_files[class_]) for class_ in self.list_classes])
        self.class_weight = self.class_weight/np.sum(self.class_weight)

    def get_sample(self):
        class_idx = np.random.choice(self.range_list_classes, 1, p=self.class_weight)[0]
        examples_class_idx = np.random.choice(range(len(self.class_to_list_files[self.list_classes[class_idx]])), 2)
        positive_example_1, positive_example_2 = \
            self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[0]],\
            self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[1]]


        negative_example = None
        while negative_example is None or self.file_class_mapping[negative_example] == \
                self.file_class_mapping[positive_example_1]:
            negative_example_idx = np.random.choice(self.range_all_files, 1)[0]
            negative_example = self.list_all_files[negative_example_idx]
        return positive_example_1, negative_example, positive_example_2



#图像统一成256*256
batch_size = 8
input_shape = (256, 256)
base_path = "../input/train/"


#网络的损失函数，identity_loss其实没有y_true，只是对输出值进行优化（最小化），
#第二个值bpr_triplet_loss其实是并行的三个嵌入网络的输出，其计算表示了同类样本的相似度减去异类样本相似度的值（最大化）

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss


#以ResNet50网络为基础并使用imagenet权值迁移训练
def get_base_model(freeze=False):
    latent_dim = 1024
    base_model = ResNet50(include_top=False, weights='imagenet')  # use weights='imagenet' locally

    if freeze:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    #x = GlobalMaxPooling2D()(x)
    #x = Dropout(0.1)(x)
    dense_1 = Dense(latent_dim)(x)
    normalized = Lambda(lambda  x: K.l2_normalize(x,axis=1))(dense_1)
    base_model = Model(base_model.input, normalized, name="base_model")
    return base_model


#构建共享权值的并行网络
def build_model(freeze):
    base_model = get_base_model(freeze)

    positive_example_1 = Input(input_shape+(3,) , name='positive_example_1')
    negative_example = Input(input_shape+(3,), name='negative_example')
    positive_example_2 = Input(input_shape+(3,), name='positive_example_2')

    positive_example_1_out = base_model(positive_example_1)
    negative_example_out = base_model(negative_example)
    positive_example_2_out = base_model(positive_example_2)

    loss = merge(
        [positive_example_1_out, negative_example_out, positive_example_2_out],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_example_1, negative_example, positive_example_2],
        output=loss)
    model.compile(loss=identity_loss, optimizer=Adam(0.000001))

    print(model.summary())

    return model

#构建共享权值的并行网络
def build_model2(freeze,weight_path):

    latent_dim = 1024
    bbase_model = ResNet50(include_top=False)#, weights='imagenet')  # use weights='imagenet' locally

    x = bbase_model.output
    # x = GlobalMaxPooling2D()(x)
    # x = Dropout(0.5)(x)
    dense_1 = Dense(latent_dim)(x)
    normalized = Lambda(lambda  x: K.l2_normalize(x,axis=1))(dense_1)
    base_model = Model(bbase_model.input, normalized, name="base_model")
    #base_model = get_base_model(freeze)

    positive_example_1 = Input(input_shape+(3,) , name='positive_example_1')
    negative_example = Input(input_shape+(3,), name='negative_example')
    positive_example_2 = Input(input_shape+(3,), name='positive_example_2')

    positive_example_1_out = base_model(positive_example_1)
    negative_example_out = base_model(negative_example)
    positive_example_2_out = base_model(positive_example_2)

    loss = merge(
        [positive_example_1_out, negative_example_out, positive_example_2_out],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_example_1, negative_example, positive_example_2],
        output=loss)

    model.load_weights(weight_path)

    if freeze:
        for layer in bbase_model.layers:
            layer.trainable = False

    model.compile(loss=identity_loss, optimizer=Adam(0.000001))
    print(model.summary())

    return model


model_name = "triplet_model"

file_path = "weights_best.h5"

#interfa网络用于预测
def build_inference_model(weight_path=file_path):
    base_model = get_base_model()

    positive_example_1 = Input(input_shape+(3,) , name='positive_example_1')
    negative_example = Input(input_shape+(3,), name='negative_example')
    positive_example_2 = Input(input_shape+(3,), name='positive_example_2')

    positive_example_1_out = base_model(positive_example_1)
    negative_example_out = base_model(negative_example)
    positive_example_2_out = base_model(positive_example_2)

    loss = merge(
        [positive_example_1_out, negative_example_out, positive_example_2_out],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_example_1, negative_example, positive_example_2],
        output=loss)
    model.compile(loss=identity_loss, optimizer=Adam(0.000001))

    model.load_weights(weight_path)

    inference_model = Model(base_model.get_input_at(0), output=base_model.get_output_at(0))
    inference_model.compile(loss="mse", optimizer=Adam(0.000001))
    print(inference_model.summary())

    return inference_model


BBOX = '../input/bounding_boxes.csv'
bbox_df = pd.read_csv(BBOX).set_index('Image')


#读图像，corp, 数据增强
def read_and_resize(filepath):
    #im = Image.open((filepath)).convert('RGB')
    im = cv2.imread(filepath)
    #filename = filepath.split(os.sep)[-1]
    filename = filepath.split('/')[-1]
    bbox = bbox_df.loc[filename]
    x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
    if not (x0 >= x1 or y0 >= y1):
        im = im[y0:y1, x0:x1, :]
    #im = im.resize(input_shape)
    im = cv2.resize(im, input_shape)
    im_array = np.array(im, dtype="uint8")[..., ::-1]
    return np.array(im_array / (np.max(im_array)+ 0.001), dtype="float32")

def augment(im_array):
    # if np.random.uniform(0, 1) > 0.9:
    #     im_array = np.fliplr(im_array)
    return im_array


#生成数据对
def gen(triplet_gen):
    while True:
        list_positive_examples_1 = []
        list_negative_examples = []
        list_positive_examples_2 = []

        for i in range(batch_size):
            positive_example_1, negative_example, positive_example_2 = triplet_gen.get_sample()
            positive_example_1_img, negative_example_img, positive_example_2_img = read_and_resize(base_path+positive_example_1), \
                                                                       read_and_resize(base_path+negative_example), \
                                                                       read_and_resize(base_path+positive_example_2)

            positive_example_1_img, negative_example_img, positive_example_2_img = augment(positive_example_1_img), \
                                                                                   augment(negative_example_img), \
                                                                                   augment(positive_example_2_img)

            list_positive_examples_1.append(positive_example_1_img)
            list_negative_examples.append(negative_example_img)
            list_positive_examples_2.append(positive_example_2_img)

        list_positive_examples_1 = np.array(list_positive_examples_1)
        list_negative_examples = np.array(list_negative_examples)
        list_positive_examples_2 = np.array(list_positive_examples_2)
        yield [list_positive_examples_1, list_negative_examples, list_positive_examples_2], np.ones(batch_size)

# Read data，mapping the file with id
data = pd.read_csv('../input/clear.csv')
train, test = train_test_split(data, test_size=0.1, shuffle=True, random_state=1337)
file_id_mapping_train = {k: v for k, v in zip(train.Image.values, train.Id.values)}
file_id_mapping_test = {k: v for k, v in zip(test.Image.values, test.Id.values)}
train_gen = sample_gen(file_id_mapping_train)
test_gen = sample_gen(file_id_mapping_test)

# for p in data.Image.values:
#     im = cv2.imread('../input/train/'+ p)
#     bbox = bbox_df.loc[p]
#     x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
#     if not (x0 >= x1 or y0 >= y1):
#         im = im[y0:y1, x0:x1, :]
#     #im = im.resize(input_shape)
#     im = cv2.resize(im, input_shape)
#     cv2.imwrite('../input/pick/'+ p,im)



# 构建模型
model = build_model(freeze=False)
#model.load_weights(file_path)

checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min') #'val_loss'

early = EarlyStopping(monitor="val_loss", mode="min", patience=2)

def lr_schedule(epoch):
    if epoch < 10:
        return 0.001
    elif epoch < 20:
        return 0.0001
    elif epoch < 60:
        return 0.00001
    elif epoch < 100:
        return 0.000001
    else:
        return 0.0000001

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

#callbacks_list = [checkpoint, early]  # early
callbacks_list = [checkpoint,learning_rate_scheduler]

history = model.fit_generator(gen(train_gen), validation_data=gen(test_gen),initial_epoch=0, epochs=200,
                              verbose=1, workers=4,
                              callbacks=callbacks_list, steps_per_epoch=1000,
                              validation_steps=30)
#model.save_weights("onestage_model.h5")

# model2 = build_model2(freeze=True,weight_path=file_path)
#
# history2 = model2.fit_generator(gen(train_gen), validation_data=gen(test_gen),initial_epoch=5, epochs=20,
#                               verbose=1, workers=4,
#                               callbacks=callbacks_list, steps_per_epoch=1000,
#                               validation_steps=30)


#————————————————————我是分割线，训练部分结束————————————

model_name = "triplet_loss"
def data_generator(fpaths, batch=16):
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        img = read_and_resize(path)
        imgs.append(img)
        fnames.append(os.path.basename(path))
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)
        yield fnames, imgs
    raise StopIteration()


def sigmoid(x):
    return x #1/(1+np.exp(-x))

def compare(predicts_0,predicts_1):
    comp1 = sigmoid(np.sum(predicts_0 * predicts_1, axis=-1, keepdims=True))
    return comp1

imgs = []
img1 = read_and_resize('../input/test\\3fc59f4b1.jpg')
imgs.append(img1)
img2 = read_and_resize('../input/train\\c8baf072e.jpg')
imgs.append(img2)
img3 = read_and_resize('../input/train\\777c305a4.jpg')
imgs.append(img3)

imgs = np.array(imgs)
inference_model = build_inference_model()

predicts = inference_model.predict(imgs)
predicts1 = predicts.tolist()
print(predicts1[0])
print(predicts1[1])
print(predicts1[2])

#comp1 = sigmoid(np.sum(predicts[0] * predicts[1], axis=-1, keepdims=True))
comp1 = euclidean_distances(predicts[0].reshape(1, -1),predicts[1].reshape(1, -1))
#comp2 = sigmoid(np.sum(predicts[0] * predicts[2], axis=-1, keepdims=True))
comp2 = euclidean_distances(predicts[0].reshape(1, -1),predicts[2].reshape(1, -1))
print(comp1)
print(comp2)


def save_list(name,list):
    file = open(name, 'w')
    for lst in list:
        file.write(lst)
        file.write('\n')
    file.close()

def read_list(name):
    file = open(name, 'r')
    list_read = file.readlines()
    return list_read

#分别计算训练集和测试集

data = pd.read_csv('../input/train.csv')

file_id_mapping = {k: v for k, v in zip(data.Image.values, data.Id.values)}

train_files = glob.glob("../input/train/*.jpg")
test_files = glob.glob("../input/test/*.jpg")

train_preds = []
train_file_names = []
train_names = []
i = 1

for fnames in train_files:
    if i % 10 == 0:
        print(i/len(train_files)*100)
    i += 1
    t_id = file_id_mapping[fnames.split(os.sep)[-1]]
    if t_id != "new_whale":
        imgs = []
        img1 = read_and_resize(fnames)
        imgs.append(img1)
        imgs = np.array(imgs)
        predicts = inference_model.predict(imgs)
        predicts = predicts.tolist()
        train_preds += predicts
        train_file_names += fnames
        train_names.append(fnames)


train_preds = np.array(train_preds)
save_list('train_file_names.txt',train_names)
np.save('train',train_preds)



test_preds = []
test_file_names = []
i = 1
# for fnames, imgs in data_generator(test_files, batch=32):
for fnames in test_files:
    if i%10==0:
        print(i / len(test_files) * 100)
    i += 1
    imgs = []
    img1 = read_and_resize(fnames)
    imgs.append(img1)
    imgs = np.array(imgs)
    predicts = inference_model.predict(imgs)
    predicts = predicts.tolist()
    test_preds += predicts
    test_file_names += fnames

test_preds = np.array(test_preds)
np.save('test',test_preds)
save_list('test_file_names.txt',test_file_names)

# test_preds = np.load('test.npy')
# test_file_names = read_list('test_file_names.txt')
#
# train_preds = np.load('train.npy')
# train_names = read_list('train_file_names.txt')

#计算相似度分类（太耗时）
# preds_str = []
# i=0
# for pred in test_preds:
#     sample_result = []
#     sample_classes = []
#     for t_pred,name in zip(train_preds,train_names):
#         comp = compare(pred,t_pred)
#         name1 = name.split(os.sep)[-1].strip()
#         class_train = file_id_mapping[name1]
#         if class_train not in sample_classes:
#             sample_result.append((class_train, comp))
#             sample_classes.append(class_train)
#     sample_result.append(("new_whale", 0.999))
#     sample_result.sort(key=lambda x: x[1])
#     sample_result = sample_result[::-1]
#     sample_result = sample_result[:5]
#     print('{}:'.format(i),sample_result)
#     i+=1
#     preds_str.append(" ".join([x[0] for x in sample_result]))
#
# df = pd.DataFrame(preds_str, columns=["Id"])
# df['Image'] = [x.split(os.sep)[-1] for x in test_files]
# df.to_csv("sub_%s.csv"%model_name, index=False)


#
#使用nn分类
neigh = NearestNeighbors(n_neighbors=500)
neigh.fit(train_preds)

# Было
#distances, neighbors = neigh.kneighbors(train_preds)
#print(distances, neighbors)

distances_test, neighbors_test = neigh.kneighbors(test_preds)
distances_test, neighbors_test = distances_test.tolist(), neighbors_test.tolist()
preds_str = []


def merge_class(result):
    name = []
    dis = []
    f_result = []
    for x in result:
        if x[0] not in name:
            name.append(x[0])
            dis.append(x[1])
            f_result.append(x)
        else:
            ind = name.index(x[0])
            if x[1]<dis[ind]:
                dis[ind] = x[1]
                name[ind] = x[0]
                f_result[ind] = x
    return f_result


#选择适当的阈值加入new_whale类并保存文件
for filepath, distance, neighbour_ in zip(test_files, distances_test, neighbors_test):
    sample_result = []
    sample_classes = []
    for d, n in zip(distance, neighbour_):
        train_file = train_names[n].split(os.sep)[-1].strip()
        class_train = file_id_mapping[train_file]
        #if class_train not in sample_classes:
        sample_classes.append(class_train)
        sample_result.append((class_train, d))

    sample_result.append(("new_whale", 0.03))
    sample_result = merge_class(sample_result)
    sample_result.sort(key=lambda x: x[1])
    sample_result = sample_result[:5]
    print(filepath,sample_result)
    preds_str.append(" ".join([x[0] for x in sample_result]))

df = pd.DataFrame(preds_str, columns=["Id"])
df['Image'] = [x.split(os.sep)[-1] for x in test_files]
df.to_csv("sub_%s.csv"%model_name, index=False)



