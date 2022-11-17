import tensorflow.compat.v1 as tf


# Lab2_1에선 X, y는 직접 값을 주었는데, 전시간에 placeholder처럼 값을 직접 주지 않고 코드를 작성할 때
# placeholder의 경우 학습 데이터를 따로 던져줄 수 있다는 장점을 가지고 있다.
with tf.compat.v1.Session() as sess:
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    X = tf.placeholder(tf.float32, shape=[None]) #shape None은 값을 1차원 array로 갯수를 원하는대로 줄 수 있다.
    Y = tf.placeholder(tf.float32, shape=[None])
    
    hypothesis = X*W+b
    
    cost = tf.reduce_mean(tf.square(hypothesis-Y))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)
    
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        val_cost, val_W, val_b, _ =sess.run([cost, W, b, train], feed_dict={X:[1,2,3,4,5], Y:[2.1, 3.1, 4.1, 5.1, 6.1]})
        if step%100==0:
            print(step, val_cost, val_W, val_b)
