# 데이터가 많아지니깐 소스코드에 직접 쓰기가 불가능해진다.
# 보통 csv파일을 많이 씀.
# ,로 구성된 파일이다

import tensorflow.compat.v1 as tf
import numpy as np
import os

dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir))

xy = np.loadtxt(os.path.join(dir,'data-01-test-score.csv'), delimiter=',', dtype=np.float32)
x_data = xy[:,0:-1] # 행은 모두다 쓰고, 열은 뒤에서 하나만(label) 빼고 쓰겠다.
y_data = xy[:,[-1]] # xy[:, -1]해도 무방하다.

with tf.compat.v1.Session() as sess:
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    W = tf.Variable(tf.random_normal([3,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    
    hypothesis = tf.matmul(X,W) + b
    
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)
    
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
        if step%100 == 0:
            print(step, "Cost : ", cost_val, "\nPrediction : \n", hy_val)


# 파일이 굉장히 커서 메모리에 한번에 올리기 어렵다?
# 이런 경우 tensorflow는 Queue Runners라는 시스템이 있음
# Queue에 쌓고 reader1 -> decoder로 어떤 배치만큼 읽어와서 학습을 시킴
# 굉장히 큰 데이터를 쓸 때 유용하게 쓸 수 있음
# Lab4_2_2에서 살펴보자.

