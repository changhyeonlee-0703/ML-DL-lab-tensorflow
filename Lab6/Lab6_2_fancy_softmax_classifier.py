import tensorflow.compat.v1 as tf
import numpy as np
import os

dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir))

xy = np.loadtxt(os.path.join(dir,'data-04-zoo.csv'), delimiter=',', dtype=np.float32)

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

nb_classes = 7

# 기본적으로 제공되는 cross entropy 함수와 one-hot함수를 사용할 예정
with tf.compat.v1.Session() as sess:
    X = tf.placeholder(tf.float32, shape=[None, 16])
    Y = tf.placeholder(tf.int32, shape=[None,1])
    
    # 데이터의 레이블이 정수 0~6사이의 값
    # 그렇기에 결과값을 onehot으로 바꿔줘야함. = tf.one_hot(Y안에는, 7개의 클래스가 있다=nb_classes)
    # ex) [1,5]인 데이터 2개의 레이블을 클래스가 6개이면 tf.one_hot([1,5],6) = [[0,1,0,0,0,0], [0,0,0,0,0,1]]
    # 그러나 차원이 N이면 tf.one_hot은 N+1차원으로 마든다. [[[0,1,0,0,0,0]], [[0,0,0,0,0,1]]]
    # 그래서 차원 하나를 줄여주기 위해 tf.reshape를 해줌.
    Y_one_hot = tf.one_hot(Y, nb_classes)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # -1은 알아서, 7개로 class를 분류하고 앞은 알아서란 뜻
    # 결과로 [[0,1,0,0,0,0], [0,0,0,0,0,1]] 나온다.
    
    W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
    b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
    
    logits = tf.matmul(X, W)+b
    hypothesis = tf.nn.softmax(logits)
    
    # cross entropy 라이브러리
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
    cost = tf.reduce_mean(cost_i)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
    
    prediction = tf.argmax(hypothesis, 1) # softmax로 나온 확률을 클래스에 맞게 스칼라 값으로 만들어줌
    correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1)) # 맞게 예측한지 비교함.  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 맞게 예측한 것을 모아서 정확도를 측정
    
    sess.run(tf.global_variables_initializer())
    
    
    for step in range(2000):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step%100==0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X:x_data, Y:y_data})
            print('Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}'.format(step, loss, acc))
            
    pred = sess.run(prediction, feed_dict={X:x_data})
    for p,y in zip(pred, y_data.flatten()): # flatten()은 y가 [[1], [0]] 형태를 [1,0]으로 바꿔줌
        print('[{}] Prediction: {} True Y: {}'.format(p==int(y), p, int(y)))