# 分类
import tensorflow as tf
import numpy as np

import random


# AX=0 相当于matlab中 null(a','r')
def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol * s[0]).sum()
    return rank, v[rank:].T.copy()


# 符号函数，之后要进行向量化
def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    elif x < 0:
        return -1


# noisy=False，那么就会生成N的dim维的线性可分数据X，标签为y
# noisy=True, 那么生成的数据是线性不可分的,标签为y
def mk_data(N, noisy=False):
    rang = [-1, 1]
    dim = 5

    X = np.random.rand(dim, N) * (rang[1] - rang[0]) + rang[0]

    while True:
        Xsample = np.concatenate((np.ones((1, dim)), np.random.rand(dim, dim) * (rang[1] - rang[0]) + rang[0]))
        k, w = null(Xsample.T)
        y = sign(np.dot(w.T, np.concatenate((np.ones((1, N)), X))))
        print(y[0][5])
        if np.all(y):
            break
    day=[]
    if noisy == True:
        idx = random.sample(range(1, N), N / 10)
        y[idx] = -y[idx]

    for st in range(200):
        if(y[0][st]==1):
            y1 = [1,0]
        else:
            y1 = [0,1]
        day.append(y1)
    da_x = np.float32(X.transpose())
    da_y = np.float32(day)
    return da_x, da_y, w



# 模型构建函数
def add_layer(insize, outsize, input, function = None):
    weight = tf.Variable(tf.random_normal([insize,outsize]))
    basize = tf.Variable(tf.zeros([outsize]))
    out = tf.matmul(input, weight) + basize
    if(function == None):
        output = out
    else:
        output = function(out)
    return output


# 1,数据准备
#产生的数据为随机数据，训练结果可能会不稳定
sign = np.vectorize(sign)
x_dat, y_dat, w = mk_data(200)

#x_dat = np.float32(np.random.rand(200,5)*5+10)
#y_dat = np.float32(np.zeros([200,2]))
# x_dat2 = np.float32(np.random.rand(200,5))
# y_dat2 = np.float32(np.zeros([200,2])+1)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
print(x_dat.shape)
print(y_dat.shape)

# 2，模型构建，变量定义
y_pre = add_layer(5,2,x,tf.nn.softmax)


# 3，构建损失函数及选择优化器
loss = -tf.reduce_sum(y*tf.log(y_pre))
optim = tf.train.GradientDescentOptimizer(0.05)
train_op = optim.minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pre),tf.argmax(y)),tf.float32))


# 4，启动会话，初始化变量，训练
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(3000):
        sess.run([train_op],feed_dict={x:x_dat,y:y_dat})
        if step % 100 == 0 :
            print(sess.run([loss,accuracy],feed_dict={x:x_dat,y:y_dat}))
            print(accuracy.eval(feed_dict={x:x_dat,y:y_dat}))