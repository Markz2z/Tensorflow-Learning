import tensorflow as tf
import pandas as pd
import numpy as np

#basic varibale in linear regression
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

#y = w*x + b
linear_model = W * x + b
y = tf.placeholder(tf.float32)

#cost function, loss = (linear_model - y) ^ 2
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

#using gradient descent to optimzie the cost function
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#init the environment
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

cur_w, cur_b, cur_loss = sess.run([W, b, loss], {x:x_train, y:y_train})

print ("w:%s b:%s loss:%s" % (cur_w, cur_b, cur_loss))
