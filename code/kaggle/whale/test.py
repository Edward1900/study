import numpy as np
import pandas as pd
import random
import os
import cv2
from keras.preprocessing import image
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

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
    return pl.rand() * (b - a) + a


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


def randomIntensity(im):
    '''
    rescales the intesity of the image to random interval of image intensity distribution
    '''
    return rescale_intensity(im,
                             in_range=tuple(pl.percentile(im, (randRange(0, 10), randRange(90, 100)))),
                             out_range=tuple(pl.percentile(im, (randRange(0, 10), randRange(90, 100)))))


def randomGamma(im):
    '''
    Gamma filter for contrast adjustment with random gamma value.
    '''
    return adjust_gamma(im, gamma=randRange(0.5, 1.5))


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
               randomIntensity]
    filt = random.choice(Filters)
    return filt(im)


def randomNoise(im):
    '''
    random gaussian noise with random variance.
    '''
    var = randRange(0.001, 0.01)
    return random_noise(im, var=var)


def augment(im, Steps = [randomFilter, randomNoise,randomCrop]):
    '''
    image augmentation by doing a sereis of transfomations on the image.
    '''
    for step in Steps:
        im = step(im)
    return im


def aug_img(train_df):
    train_new_df = pd.DataFrame([], columns=['Image', 'Id'])
    n = 0
    training_pts_per_class = train_df.groupby('Id').size()
    training_groups = train_df.groupby('Id').groups
    train_aug_index = training_pts_per_class.nlargest(5005).index
    for id in train_aug_index:
        print(id)
        grop_list = training_groups[id]
        #print(grop_list)
        if len(grop_list)<10:
            for i in range(len(grop_list)):
                img_name = train_df.iloc[grop_list[i]]['Image']
                img_id = train_df.iloc[grop_list[i]]['Id']
                #print(i,':',img_name)
                img = cv2.imread("../input/train/" + img_name)
                for k in range(10 - len(grop_list)):
                    img_aug = augment(img)
                    img_aug_name =  img_id + '{}_{}.jpg'.format(i,k)
                    pl.imsave("../input/train/" +img_aug_name, img_aug)
                    train_new_df.loc[n, 'Image'] = img_aug_name
                    train_new_df.loc[n, 'Id'] = img_id
                    n+=1
                    print(n)

    train_new_df.to_csv('train_aug.csv', index=False)


# train_new_df.sample(frac=1).reset_index(drop=True)
# train_new = pd.concat([train_df,train_new_df],ignore_index= True)
#
# train_new.to_csv('train_new.csv', index=False)

# train_new = pd.read_csv("train_new.csv")
# train_new.head()
#

def sub_class(train_df,dst_name=None):
    train_new_sub = pd.DataFrame([], columns=['Image', 'Id'])
    n = 0
    for id in train_df['Id']:
        #print(id)
        if not id == 'new_whale':
            train_new_sub.loc[n, 'Image'] = train_df.iloc[n]['Image']
            train_new_sub.loc[n, 'Id'] = train_df.iloc[n]['Id']
        n+=1
        if not dst_name == None:
            train_new_sub.to_csv(dst_name, index=False)
    return train_new_sub


def gen_pd_csv(train_df,pd_list,name):
    col = ['Image','Id']
    val_df = pd.DataFrame([],columns=col)
    # val_df['Id'] = ''
    for i in range(len(pd_list)):
        val_df.loc[i, 'Image'] = train_df.iloc[pd_list[i]]['Image']
        val_df.loc[i, 'Id'] = train_df.iloc[pd_list[i]]['Id']
    val_df.to_csv(name, index=False)
    print(val_df.head(5))


def get_train_pair(train_df):
    training_pts_per_class = train_df.groupby('Id').size()
    training_groups = train_df.groupby('Id').groups
    print("Min example a class can have: ", training_pts_per_class.index)
    # print("Max example a class can have: \n",training_pts_per_class.nlargest(500))
    # print("Max example index: \n",training_pts_per_class.nlargest(5005).index[200:])
    group_id = training_pts_per_class.index

    train_id_list = []
    for id in group_id:
        group_list = training_groups[id]
        if len(group_list) >= 2:
            if not id == 'new_whale':
                train_id_list.append(id)

    print(train_id_list)
    print(len(train_id_list))

    train_index_list = []
    for id in train_id_list:
        group_list = training_groups[id]
        for i in range(len(group_list)):
            train_index_list.append(group_list[i])
    print(train_index_list)
    print(len(train_index_list))

    col = ['ImageA', 'ImageB', 'Label']
    train_pair = pd.DataFrame([], columns=col)
    k = 0
    for id in train_id_list:
        group_list = training_groups[id]
        # print(group_list)
        for i in range(len(group_list)):
            img_name_a = train_df.iloc[group_list[i]]['Image']
            img_name_b = train_df.iloc[group_list[i - 1]]['Image']
            train_pair.loc[k, 'ImageA'] = img_name_a
            train_pair.loc[k, 'ImageB'] = img_name_b
            train_pair.loc[k, 'Label'] = 1
            k += 1

        neg_group_list = train_index_list.copy()
        for n in group_list:
            neg_group_list.remove(n)
        random.shuffle(neg_group_list)
        for i in range(len(group_list)):
            img_name_a = train_df.iloc[group_list[i]]['Image']
            img_name_b = train_df.iloc[neg_group_list[i]]['Image']
            train_pair.loc[k, 'ImageA'] = img_name_a
            train_pair.loc[k, 'ImageB'] = img_name_b
            train_pair.loc[k, 'Label'] = 0
            k += 1
            # img_id = train_df.iloc[group_list[i]]['Id']

    print(train_pair)
    train_pair.to_csv('train_pair.csv', index=False)


clear_img = os.listdir("../input/pick/pp/")
file = open("../input/pick.txt",'r')
list_name = file.readlines()
file.close()
for nam in list_name:
    clear_img.append(nam.strip())

print(len(clear_img),clear_img)

def sub_clear(train_df1,dst_name=None):
    train_new_sub = pd.DataFrame([], columns=['Image', 'Id'])
    n = 0
    for img in train_df1['Image']:
        #print(img)
        print(n)
        if img not in clear_img:
            train_new_sub.loc[n, 'Image'] = train_df1.iloc[n]['Image']
            train_new_sub.loc[n, 'Id'] = train_df1.iloc[n]['Id']
        n+=1
        if not dst_name == None:
            train_new_sub.to_csv(dst_name, index=False)
    return train_new_sub


def sub_clear2(train_df2,dst_name=None):
    nm = 0
    train_copy = train_df2.copy()
    train_new_sub = pd.DataFrame([], columns=['Image', 'Id'])
    training_groups = train_df2.groupby('Id').groups
    train_copy.set_index("Image")
    cclear_img = []

    imag_np = train_copy['Image'].values
    # print(np.where(imag_np=='0001f9222.jpg'))
    # print(train_copy['Id'].values[np.where(imag_np=='0001f9222.jpg')])

    for clear_i in clear_img:
        if clear_i in train_copy['Image'].values:
            id = train_copy['Id'].values[np.where(imag_np==clear_i)]
            print(id)
            grop_list = training_groups[id[0]]
            if len(grop_list) > 1:
                cclear_img.append(clear_i)
            else:
                nm+=1
    n = 0
    for img in train_df2['Image']:
        #print(img)
        print(n)
        if img not in cclear_img:
            train_new_sub.loc[n, 'Image'] = train_df2.iloc[n]['Image']
            train_new_sub.loc[n, 'Id'] = train_df2.iloc[n]['Id']
        n+=1
        if not dst_name == None:
            train_new_sub.to_csv(dst_name, index=False)
    print(nm)
    print(len(cclear_img))
    return train_new_sub



os.listdir("../input/")
train_df = pd.read_csv("../input/train.csv")
train_df.head()
print("Unique whales: ",train_df['Id'].nunique()) # it includes new_whale as a separate type.

train_sub = sub_class(train_df)
#
train_clear = sub_clear2(train_sub,"clear2.csv")











