# CNN卷积神经网络
import tensorflow as tf
import numpy as np
import glob
from tensorflow.python.platform import gfile
import cv2
from PIL import Image
import os


from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('data', one_hot=True)

train_path = 'J:/study/tensorflow_learnning/data/'
class_list = ['0','1','2','3','4','5','6','7','8','9']



def vectorize_imgs(img_path_list):
    image_arr_list = []
    for img_path in img_path_list:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img_arr = np.asarray(img, dtype='float32')
            image_arr_list.append(img_arr)
        else:
            print(img_path)
    return image_arr_list



def one_hot(labels,Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
    return one_hot_label


def get_data(file_dir,batch_size,n_classes):
    imge_paths = []
    labels = []
    for n in range(len(class_list)):
        sub_files = os.listdir(file_dir  + class_list[n])
        for m in range(len(sub_files)):
            imge_paths.append(file_dir  + class_list[n]+'/'+sub_files[m])
            labels.append(n)
    print(imge_paths)
    print(len(labels),labels)

    paths_lt = tf.cast(imge_paths, tf.string)
    labels_lt = tf.cast(labels, tf.int32)
    paths_list,label_list = tf.train.slice_input_producer([paths_lt,labels_lt],shuffle=True)

    print(paths_list)
    print(label_list)
    images1 = tf.read_file(paths_list)
    print(images1)
    images_decode = tf.image.decode_png(images1, channels=1)
    #images_decode = tf.image.decode_bmp(images1, channels=3)
    images_resize = tf.image.resize_image_with_crop_or_pad(images_decode, 28, 28)
    images_standardization = tf.image.per_image_standardization(images_resize)
    #print(images_standardization)
    images_batch,labels_batch = tf.train.batch([images_standardization,label_list],batch_size=batch_size,num_threads=64,
                                                       capacity=1024)
    images_batch = tf.cast(images_batch, tf.float32)
    labels_batch = tf.one_hot(labels_batch,depth=n_classes)
    labels_batch = tf.reshape(labels_batch, [batch_size, n_classes])
    print(images_batch)
    print(labels_batch)
    return images_batch,labels_batch

def get_paths(file_dir):
    imge_paths = []
    labels = []
    for n in range(len(class_list)):
        sub_files = os.listdir(file_dir  + class_list[n])
        for m in range(len(sub_files)):
            imge_paths.append(file_dir  + class_list[n]+'/'+sub_files[m])
            labels.append(n)
    print(imge_paths)
    print(len(labels),labels)

    return len(labels),imge_paths,labels

def get_data_batch(imge_paths,labels,start,batch_size,n_classes):
    end = (start + batch_size) % len(labels)
    if start < end:
        path_batch = imge_paths[start : end]
        label_batch = labels[start : end]
    else:
        path_batch = np.concatenate([imge_paths[start:], imge_paths[:end]])
        label_batch = np.concatenate([labels[start:], labels[:end]])


    images_batch = np.asarray(vectorize_imgs(path_batch), dtype='float32') / 255.

    labels_batch = one_hot(label_batch, n_classes)

    return images_batch,labels_batch

def read_img(path):
    images1 = tf.read_file(path)
    images_decode = tf.image.decode_png(images1, channels=1)
    images_resize = tf.image.resize_image_with_crop_or_pad(images_decode, 28, 28)
    images_standardization = tf.image.per_image_standardization(images_resize)
    img = tf.expand_dims(images_standardization,axis=0)
    return img

# 模型构建函数
def weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.01))

def baise(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x,w):
    out = tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")
    return out

def deconv2d(x,w,shape):
    out = tf.nn.conv2d_transpose(x,w,shape,strides=[1,1,1,1],padding="SAME")
    return out

def max_pool_2x2(x):
    out,argmax = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    return out,argmax


def un_max_pool(net, mask, stride):
    '''
    定义一个反最大池化的函数，找到mask最大的索引，将max的值填到指定位置
    args:
        net:最大池化后的输出，形状为[batch, height, width, in_channels]
        mask：位置索引组数组，形状和net一样
        stride:步长，是一个int32类型，这里就是max_pool_with_argmax传入的stride参数
    '''
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret

# cnn模型构建，变量定义
def cnn(x):
    with tf.name_scope("model"):
        with tf.variable_scope("layer1") as scope:
            # layer1

            weigh1 = weights([5, 5, 1, 32])
            bas_1 = baise([32])
            conv2_1 = tf.nn.relu(conv2d(x, weigh1) + bas_1,name=scope.name)  # 28*28*32
            max_pool_1,argmax1 = max_pool_2x2(conv2_1)  # out 14*14*32
            v_conv2_1 = tf.nn.relu(conv2_1)  #
            v_layer1 = deconv2d(v_conv2_1, weigh1,x.shape)
            #v_layer1 = tf.squeeze(v_layer1)

        with tf.variable_scope("layer2") as scope:
            # layer2
            weight2 = weights([5, 5, 32, 64])
            bas_2 = baise([64])
            conv2_2 = tf.nn.relu(conv2d(max_pool_1, weight2) + bas_2,name=scope.name)  # 14*14*64
            v_conv2_2 = tf.nn.relu(conv2_2)  #
            v_layer2 = deconv2d(v_conv2_2, weight2, max_pool_1.shape)
            v_unpool2 = un_max_pool(v_layer2,argmax1,2)
            v_layer2_cov1 = tf.nn.relu(v_unpool2)  #
            v_layer2_1 = deconv2d(v_layer2_cov1, weigh1,x.shape)

            max_pool_2, _ = max_pool_2x2(conv2_2)  # 7*7*64

        with tf.variable_scope("layer_f") as scope:
            # layer_f
            w_f = weights([7 * 7 * 64, 1024])
            b_f = baise([1024])
            f_in = tf.reshape(max_pool_2, [-1, 7 * 7 * 64])
            # f_drop = tf.nn.dropout(f_in,0.5)
            f_out = tf.nn.relu(tf.matmul(f_in, w_f) + b_f,name=scope.name)

        with tf.variable_scope("layer_f_2") as scope:
            # layer_f_2
            w_f_2 = weights([1024, 10])
            b_f_2 = baise([10])
            y_pre = tf.nn.softmax(tf.matmul(f_out, w_f_2) + b_f_2, name="out")

    return v_layer1,v_layer2_1,y_pre

def loss(y_pre,labels):
    with tf.name_scope("loss"):
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pre, labels=labels,name='cross-entropy')
        # loss = tf.reduce_mean(cross_entropy, name='loss')
        loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(y_pre), reduction_indices=[1]))  # loss
    return loss

def train(loss):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    # train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
    return train_op

def accuracy(y_pre,labels):
    correct = tf.equal(tf.argmax(y_pre, 1), tf.argmax(labels, 1))
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct) * 100.0
    return accuracy


# 数据准备
#img_n,image_path,label_path = get_paths(train_path)
#images,labels = get_data(train_path,100,10)  #训练时选择 100， 如测试时选择图像总数目（不建议用此op测试）
images1,labels = mnist.test.images[:100], mnist.test.labels[:100]  #测试
images = tf.reshape(images1,shape=[100,28,28,1])
ckpt_path = './ckpt/'

# test_one_img = read_img('J:/study/tensorflow_learnning/data/0/1.png')
# print(test_one_img)
# y_pre = cnn(test_one_img)   #测试单张图像

# 构建损失函数及选择优化器

xs = tf.placeholder(tf.float32, [None, 28,28,1])
ys = tf.placeholder(tf.float32, [None, 10])

v_layer1,v_layer2,y_pre = cnn(images)
loss = loss(y_pre,labels)     # loss
train_op = train(loss)
accuracy = accuracy(y_pre,labels)


# 启动会话，初始化变量，训练

isTrain = False
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
save = tf.train.Saver()

# setup tensorboard

tf.summary.scalar('loss', loss) #加入loss变量
for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var) #加入权值变量
merged = tf.summary.merge_all()  #融合所有的总结
writer = tf.summary.FileWriter('logs/train.log', sess.graph) #创建writer写图信息

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)



if isTrain:
    ckpt1 = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt1 and ckpt1.model_checkpoint_path:
        save.restore(sess, ckpt1.model_checkpoint_path)
    for step in range(1000):
        sess.run(train_op)
        summary_str = sess.run(merged) #运行融合summary
        writer.add_summary(summary_str, step) #添加到writer中
        if step % 50 == 0:
            print(sess.run(loss),":",sess.run(accuracy))
            #print(sess.run(images).shape)
            save.save(sess,ckpt_path + 'test-model.ckpt',step)
else:
    ckpt1 = tf.train.get_checkpoint_state(ckpt_path)
    #saver = tf.train.import_meta_graph('my_test_model-1000.meta') 恢复图
    #graph = tf.get_default_graph()
    if ckpt1 and ckpt1.model_checkpoint_path:
        save.restore(sess, ckpt1.model_checkpoint_path)
        for step in range(1):
            #print(sess.run(images))
            print(sess.run(accuracy))
            res = sess.run(v_layer1)
            res2 = sess.run(v_layer2)
            #print(sess.run(tf.argmax(y_pre,1)))
            for i in range(res.shape[0]):
                #nn_image = tf.squeeze(res[i])
                #img_n8 = tf.cast(v_layer1[i],tf.uint8)
                #nn_image1 = np.array(nn_image)
                img_d = np.uint8(res[i])
                if not (os.path.exists('./layers1')):
                    os.makedirs('./layers1')
                cv2.imwrite("./layers1/{}.png".format(i),img_d*255)
            for i in range(res2.shape[0]):
                #nn_image = tf.squeeze(res[i])
                #img_n8 = tf.cast(v_layer1[i],tf.uint8)
                #nn_image1 = np.array(nn_image)
                img_d = np.uint8(res2[i])
                if not (os.path.exists('./layers2')):
                    os.makedirs('./layers2')
                cv2.imwrite("./layers2/{}.png".format(i),img_d*255)

    else:
        pass

coord.request_stop()
coord.join(threads)