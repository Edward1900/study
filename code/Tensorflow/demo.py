import tensorflow as tf
import numpy as np

# 1,数据准备
x_dat = np.float32(np.random.rand(2, 100))
y_dat = np.dot([2.0, 3.0], x_dat)+5.0
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 2，变量定义及模型构建
W=tf.Variable(tf.random_uniform([1, 2], -0.1, 0.1))
b=tf.Variable(tf.zeros([1]))
y_pre=tf.matmul(W, x)+b

# 3，构建损失函数及选择优化器
loss = tf.reduce_mean(tf.square(y_pre-y))
optim = tf.train.GradientDescentOptimizer(0.5)
train_op = optim.minimize(loss)

# 4，初始化变量启动会话，训练
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(1, 200):
    sess.run([train_op], feed_dict={x: x_dat, y: y_dat})
    if step % 20 == 0:
        print(sess.run([W, b]))
