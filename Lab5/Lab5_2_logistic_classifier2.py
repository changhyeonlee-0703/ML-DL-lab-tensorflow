
import tensorflow.compat.v1 as tf
import numpy as np
import os

# 실제 데이터를 사용해서 이진 분류를 해보자.
dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir))

xy = np.loadtxt(os.path.join(dir,'data-03-diabetes.csv'), delimiter=',', dtype=np.float32)

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]


with tf.compat.v1.Session() as sess:
    X = tf.placeholder(tf.float32, shape=[None, 8])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    W = tf.Variable(tf.random_normal([8,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    
    # 만약 시그모이드 함수를 직접 구현하고 싶다면
    # tf.div(1., 1. + tf.exp(tf.matmul(X,W)+b) 로 구현이 가능하다.
    hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
    
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
    
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
    
    
    predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32) # tf.cast함수를 통해 예측값이 true면 1.0, false 면 0.0을 반환
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32)) 
    
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        if step% 200 ==0:
            print(step, cost_val)
            
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis : ", h, "\nCorect (Y): ", c, "\nAccuracy: ", a)
