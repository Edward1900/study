# 多元线性回归
import tensorflow as tf
import numpy as np

# 模型构建函数
def add_layer(inputs, insize, outsize, activation_fun=None):
    w = tf.Variable(tf.random_normal([outsize,insize]))
    b = tf.Variable(tf.zeros([outsize,1]))
    #print(inputs.shape)
    out = tf.matmul( w,inputs) + b #注意matmul的参数顺序
    if activation_fun is None:
        output = out
    else:
        output = activation_fun(out)
    #print(output.shape)
    return output,w,b


# 1,数据准备
x_dat = np.float32(np.random.rand(5,200))
y_dat = np.dot([2.0, 3.0, 2.0,4.0,9.0],x_dat)+5.0
x = tf.placeholder(tf.float32,shape=[5,None])
y = tf.placeholder(tf.float32,shape=[None])

print(x_dat.shape)
print(y_dat.shape)

# 2，模型构建，变量定义
y_pre,w,b = add_layer(x, 5, 1, activation_fun=None)
#layer2 = add_layer(layer1, 10, 2, activation_fun=tf.nn.relu)
#y_pre = tf.nn.softmax(layer1)

# 3，构建损失函数及选择优化器
loss = tf.reduce_mean(tf.square(y-y_pre))
#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pre),reduction_indices=[1]))
optmiz = tf.train.GradientDescentOptimizer(0.1)  #不收敛需调整变化率，这里0.5不收敛，太小则训练太慢。
train_op = optmiz.minimize(loss) #根据loss来确定改变变量的方向及数值

# 4，启动会话，初始化变量，训练
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(2000):
        sess.run([train_op],feed_dict={x: x_dat, y: y_dat})
        if step % 200 == 0:
            print(sess.run([loss,w,b],feed_dict={x: x_dat, y: y_dat}))

