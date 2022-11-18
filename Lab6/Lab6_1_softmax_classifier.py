import tensorflow.compat.v1 as tf

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

with tf.compat.v1.Session() as sess:
    X = tf.placeholder('float',[None, 4])
    Y = tf.placeholder('float',[None, 3])
    nb_classes = 3
    
    # X*W = [None, 4]*[4,3] = [None, 3]
    # Y= [None, 3] 
    W = tf.Variable(tf.random_normal([4, nb_classes], name='weight')) # 결과값이 3개 이므로 nb_classes는 3이 된다.
    b = tf.Variable(tf.random_normal([nb_classes], name='bias'))
    
    hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
    
    cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
    
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step%100 ==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
            
    
    # Test & One-hot encoding
    a= sess.run(hypothesis, feed_dict={X:[[1,11,7,9]]})
    print(a, sess.run(tf.arg_max(a,1))) # tf.arg_max로 One-hot encoding을 해줌
    