import tensorflow as tf
import pandas as pd
import numpy as np

#basic varibale in linear regression
W = tf.Variable([.3, .3], dtype=tf.float32)
b = tf.Variable([.3], dtype=tf.float32)
x = tf.placeholder(tf.float32, shape=(2))

#y = w*x + b
logistic_model = tf.sigmoid(tf.reduce_sum(W * x) + b)
y = tf.placeholder(tf.float32)

#cost function, loss = (linear_model - y) ^ 2
deltas = -y * tf.log(logistic_model) + (y - 1) * tf.log(1 - logistic_model)

#using gradient descent to optimzie the cost function
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(deltas)

#init the environment
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
x_train = [[1,1],[1,0],[0,1],[0,0]]
y_train = [0, 1, 1, 0]

for i in range(50000):
    for idx in range(len(x_train)):
        sess.run(train, {x:x_train[idx], y:y_train[idx]})

w_val, b_val, y_val = sess.run([W, b, logistic_model], {x:[0, 0]})

print ("w:%s, b:%s, y:%s" % (w_val, b_val, y_val))

