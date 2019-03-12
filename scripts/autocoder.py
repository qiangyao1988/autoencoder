# -*- coding: utf-8 -*-
# @Time    : Tue Mar 12 17:01:48 2019
# @Author  : Yao Qiang
# @Email   : qiangyao1988wsu@gmail.com
# @File    : autocoder.py
# @Software: Spyder
# @Pythpon Version: python3.6


from __future__ import division, print_function, absolute_import
import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

# load dataset MNIST
mnist = input_data.read_data_sets("./data/", one_hot=True)

'''
input layer:(28，28）= 784
first hidden layer: 500
second hidden layer: 100
third hidden layer: 500
output layer:(28，28）= 784
'''
# parameters
input_n=784
hidden1_n=500
hidden2_n=100
hidden3_n=500
output_n=784

learning_rate=0.01
batch_size=100
train_epoch=30000

# Define placeholders
x=tf.placeholder(tf.float32,[None,input_n])
y=tf.placeholder(tf.float32,[None,output_n])

# define variables
weights1=tf.Variable(tf.truncated_normal([input_n,hidden1_n],stddev=0.1))
bias1=tf.Variable(tf.constant(0.1,shape=[hidden1_n]))

weights2=tf.Variable(tf.truncated_normal([hidden1_n,hidden2_n],stddev=0.1))
bias2=tf.Variable(tf.constant(0.1,shape=[hidden2_n]))

weights3=tf.Variable(tf.truncated_normal([hidden2_n,hidden3_n],stddev=0.1))
bias3=tf.Variable(tf.constant(0.1,shape=[hidden3_n]))

weights4=tf.Variable(tf.truncated_normal([hidden3_n,output_n],stddev=0.1))
bias4=tf.Variable(tf.constant(0.1,shape=[output_n]))

def get_result(x,weights1,bias1,weights2,bias2,weights3,bias3,weights4,bias4):
    '''
    function to caluculate the output
    params:
        x: input
        weights1,bias1,weights2,bias2,weights3,bias3,weights4,bias4: parameters
    return: output
    activate function: sigmoid
    '''
    a1=tf.nn.sigmoid(tf.matmul(x,weights1)+bias1)
    a2=tf.nn.sigmoid(tf.matmul(a1,weights2)+bias2)
    a3=tf.nn.sigmoid(tf.matmul(a2,weights3)+bias3)
    y_hat=tf.nn.sigmoid(tf.matmul(a3,weights4)+bias4)
    return y_hat

# get output
y_hat = get_result(x,weights1,bias1,weights2,bias2,weights3,bias3,weights4,bias4)

# loss function
loss = tf.reduce_mean(tf.pow(y_hat - y, 2))

# optomizer
train_op=tf.train.RMSPropOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    
    # initialize the variables
    tf.global_variables_initializer().run()
    
    # traning step
    for i in range(train_epoch):
        train_x,tarin_y=mnist.train.next_batch(batch_size)
        if i%1000 == 0:
            print('epoch:',i)
            print('loss:',sess.run(loss,feed_dict={x:train_x,y:train_x}))
        sess.run(train_op,feed_dict={x:train_x,y:train_x})
    
    # testing step
    test_x=mnist.test.images[:5]
    test_y=test_x 
    encode_decode=sess.run(y_hat,feed_dict={x:test_x,y:test_y})
    
    # show the result
    f,a =plt.subplots(2,5,figsize=(10,2))
    for i in range(5):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
        
    f.show()
    plt.draw()








