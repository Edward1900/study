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
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

os.listdir("../input/")

# val_df = pd.read_csv("../input/val.csv")
# val_df.head()

train_df = pd.read_csv("../input/val.csv")
train_df.head()

IMG_SIZE = 197

def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, IMG_SIZE, IMG_SIZE, 3))
    count = 0

    for fig in data['Image']:
        # load images into images of size 100x100x3
        img = image.load_img("../input/" + dataset + "/" + fig,grayscale=False, target_size=(IMG_SIZE, IMG_SIZE, 3))

        x = image.img_to_array(img)
        x = preprocess_input(x)
        #x = x -110

        X_train[count] = x
        if (count % 500 == 0):
            print("Processing image: ", count + 1, ", ", fig)
        count += 1

    return X_train

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder

# get train_data
X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255
y, label_encoder = prepare_labels(train_df['Id'])
print(y.shape)



# # get val_data
# val_x = prepareImages(val_df, val_df.shape[0], "train")
# val_x /= 255
# val_y, _ = prepare_labels(val_df['Id'])
#
# print(val_y.shape)


#################################start#########################
from keras.applications.resnet50 import ResNet50
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.optimizers import Adam

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def pre_model():
    base_model = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3),weights=None, classes=5004)
    base_model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy', metrics=[categorical_accuracy])
    return base_model

model = pre_model()
model.summary()




from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau
#early_stopping = EarlyStopping(monitor='val_loss', mode='min')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

model_checkpoint = ModelCheckpoint(filepath='model_epoch-{epoch:02d}_loss-{loss:.4f}.h5',
                                   monitor='loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
callback = [model_checkpoint]
#history = model.fit(X, y, epochs=100, batch_size=48, verbose=1, callbacks=callback)
# history = model.fit(X, y, epochs=50, batch_size=32, verbose=1,validation_split=0.1,callbacks=callback)
#gc.collect()
#model.save("model.h5")
model.load_weights('model_epoch-92_loss-0.0045.h5')
#model= load_model("model.h5")
# plt.plot(history.history['acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.show()

train_ll = pd.read_csv("../input/train_sub.csv")
_,lab_en = prepare_labels(train_ll['Id'])


def map5(x, th=0):
    X = x.copy()
    score = 0
    for i, pred in enumerate(X):
        pl = lab_en.inverse_transform(pred.argsort()[-5:][::-1])
        # print(pred.argsort()[-5:][::-1])
        pred.sort()
        pre = pred[-5:][::-1]
        # print(pre[0],th)
        plist = pl.tolist()
        for j in range(5):
            if pre[j] < th:
                plist.insert(j,'new_whale')
                break
        #print(pl[0],':',pre[0],train_df['Id'][i],(pl[0] == train_df['Id'][i]))
        for j in range(5):
            # print(pl[j], ':', train_df['Id'][i])
            # print(pl[j] == train_df['Id'][i])
            if plist[j] == train_df['Id'][i]:
                score += (5 - j)/5
                break
    return score/X.shape[0]

predictions = model.predict(np.array(X), verbose=1)
print(predictions)


# for i, pred in enumerate(predictions):
#     pl = lab_en.inverse_transform(pred.argsort()[-5:][::-1])
#     print(pred.argsort()[-5:][::-1])
#
#     pred.sort()
#     print("b",pl[0])
#     # pre = pred[-5:][::-1]
#     plist = pl.tolist()
#     plist.insert(1,'www')
#     print(' '.join(pl))
#     #np.insert(pl, 1, 'www')
#     print(' '.join(plist[0:5]))

# sco1 = map5(predictions,0.3)
# print("org:",sco1)
#
# best_th = 0
# best_score = 0
# for th in np.arange(0.8, 1, 0.01):
#     score = map5(predictions, th)
#     if score > best_score:
#         best_score = score
#         best_th = th
#     print("Threshold = {:.3f}".format(th), "MAP5 = {}".format(score))
#
test = os.listdir("../input/test/")
print(len(test))

col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''

X = prepareImages(test_df, test_df.shape[0], "test")
X /= 255

predictions = model.predict(np.array(X), verbose=1)

file= open('log.txt', 'w')
for i, pred in enumerate(predictions):
    pl = lab_en.inverse_transform(pred.argsort()[-5:][::-1])
    pred.sort()
    pre = pred[-5:][::-1]
    file.write(str(pre))
    file.write('\n')
    # print(pre[0],th)
    plist = pl.tolist()
    for j in range(5):
        if pre[j] < 0.99:
            plist.insert(j, 'new_whale')
            break

    test_df.loc[i, 'Id'] = ' '.join(plist[0:5])

file.close()

test_df.head(10)
test_df.to_csv('submission.csv', index=False)
