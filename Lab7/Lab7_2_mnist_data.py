import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import random

# 그 유명한 Mnist data를 사용함
# 손으로 쓴 숫자들인데, 숫자들을 컴퓨터가 읽을 수 있게 하기 위해 만든 데이터셋
# 28*28 = 784의 픽셀 데이터
# class는 0~9까지 10개의 클래스가 있음
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


nb_classes = 10
with tf.compat.v1.Session() as sess:
    X = tf.placeholder(tf.float32, shape=[None, 784])
    Y = tf.placeholder(tf.float32, shape=[None, nb_classes])
    
    W = tf.Variable(tf.random_normal([784, nb_classes]))
    b = tf.Variable(tf.random_normal([nb_classes]))
    
    hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
    
    cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1)) # axis=1행으로 모든 것을 더한다.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
    
    is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    # parameters
    train_epochs = 15 # 전체 데이터셋을 한번 학습시킨것은 1 epoch이라 함. 여기선 전체 데이터셋을 15번 본다.
    batch_size = 100 # 데이터셋을 100개로 쪼갠것. ex) 데이터가 1000개일 경우 100개씩 10개로 쪼갠다.
    total_batch = int(mnist.train.num_examples / batch_size) # 전체 batch는 몇개인가
    
    
    sess.run(tf.global_variables_initializer())
    for epoch in range(train_epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #100개씩을 읽어온다.
            c, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c / total_batch
            
        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))
        
    # 평가하기
    # sess.run으로 실행할 수도 있고 tensor에서 eval이란 함수를 호출할 수 있다.
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X:mnist.test.image, Y:mnist.test.labels}))
    
        # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),)
    plt.imshow(mnist.test.images[r : r + 1].reshape(28, 28), cmap="Greys", interpolation="nearest",)
    plt.show()