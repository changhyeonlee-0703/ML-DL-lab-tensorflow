import tensorflow.compat.v1 as tf


with tf.compat.v1.Session() as sess:
    X_data = [1,2,3]
    y_data = [1,2,3]

    W = tf.Variable(tf.random_normal([1]),name='weight')
    X = tf.placeholder(tf.float32, shape=[None])
    Y = tf.placeholder(tf.float32, shape=[None])

    hypothesis = W*X

    cost = tf.reduce_mean(tf.square(hypothesis-Y))

    # 수동으로 cost fun을 Minimize하는 것을 만들었음
    # Gradient descent algorithm
    # 즉 W의 값을 cost function이 작아지도록 계속해서 바꿔주고 있다.
    learning_rate = 0.1
    gradient = tf.reduce_mean((W*X-Y)*X)
    descent = W - learning_rate*gradient
    update_W = W.assign(descent)
    
    # 위에 작업을 아래의 코드로 구현할 수 있다.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # train = optimizer.minimize(cost)
    # 이것은 Lab3_3에서 살펴본다.
    
    
    sess.run(tf.global_variables_initializer())
    
    # 수동으로 W를 수정하고
    # 그것을 cost function에 적용함
    for step in range(21):
        sess.run(update_W, feed_dict={X:X_data, Y:y_data})
        print(step, sess.run(cost, feed_dict={X:X_data, Y:y_data}), sess.run(W))


