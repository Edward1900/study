import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

os.listdir("../input/")

train_df = pd.read_csv("../input/train.csv")
train_df.head()

IMG_SIZE = 197

def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, IMG_SIZE, IMG_SIZE, 3))
    count = 0

    for fig in data['Image']:
        # load images into images of size 100x100x3
        img = image.load_img("../input/" + dataset + "/" + fig, target_size=(IMG_SIZE, IMG_SIZE, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

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

X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255

y, label_encoder = prepare_labels(train_df['Id'])

print(y.shape)

from keras.applications.resnet50 import ResNet50
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.optimizers import Adam

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def pre_model():
    base_model = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3),weights=None, classes=5005)
    base_model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy', metrics=[categorical_crossentropy, categorical_accuracy])
    return base_model

model = pre_model()
model.summary()

# model = Sequential()
# model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))
# model.add(BatchNormalization(axis = 3, name = 'bn0'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2), name='max_pool'))
# model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
# model.add(Activation('relu'))
# model.add(AveragePooling2D((3, 3), name='avg_pool'))
# model.add(Flatten())
# model.add(Dense(500, activation="relu", name='rl'))
# model.add(Dropout(0.8))
# model.add(Dense(y.shape[1], activation='softmax', name='sm'))
# model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
# model.summary()


from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau
#early_stopping = EarlyStopping(monitor='val_loss', mode='min')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

model_checkpoint = ModelCheckpoint(filepath='model_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
callback = [model_checkpoint]
#history = model.fit(X, y, epochs=100, batch_size=128, verbose=1, validation_split=0.1, callbacks=callback)
history = model.fit(X, y, epochs=100, batch_size=32, verbose=1,callbacks=callback)
#gc.collect()
model.save("model.h5")
# plt.plot(history.history['acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.show()



test = os.listdir("../input/test/")
print(len(test))

col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''

X = prepareImages(test_df, test_df.shape[0], "test")
X /= 255

predictions = model.predict(np.array(X), verbose=1)

for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))

test_df.head(10)
test_df.to_csv('submission.csv', index=False)
