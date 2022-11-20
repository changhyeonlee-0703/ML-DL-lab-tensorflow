import tensorflow.compat.v1 as tf

# Traing과 test dataset으로 나누기
x_data = [[1,2,3], [1,3,2], [1,3,4], [1,5,5], [1,7,5], [1,2,5], [1,6,6], [1,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

x_test = [[2,1,1], [3,1,2], [3,3,4]]
y_test = [[0,0,1], [0,0,1], [0,0,1]]

with tf.compat.v1.Session() as sess:
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 3])
    
    W = tf.Variable(tf.random_normal([3,3]), name='weight')
    b = tf.Variable(tf.random_normal([3]), name='bias')
    
    hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
    
    cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
    
    prediction = tf.arg_max(hypothesis, 1)
    is_correct = tf.equal(prediction, tf.arg_max(Y,1))
    accuarcy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    # training
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X:x_data, Y:y_data})
        print( step, cost_val, W_val)
        
    # test
    print("Prediction: ", sess.run(prediction, feed_dict={X:x_test}))
    print("Accuracy: ", sess.run(accuarcy, feed_dict={X:x_test, Y:y_test}))
    
    
    # learning_rate가 너무 크다면 Overshooting이 일어남.
    # learning_rate가 너무 작다면 학습이 굉장히 더디게 일어남 또는 학습이 안될 수도 있다.
    # 또한 조금의 굴곡이 있을 경우 local minimum에 빠질 수 있다.
    
    # 만약 이 코드에서 learning_rate만 1.5로 바꿀 경우 발산이 일어나게 됨.
    # 반대로 learning_rate를 1e-10으로 너무 작게 설정할 경우 거의 이동되지 않거나 local minimum에 빠져 학습이 제대로 되지 않는다.
    
    # 가끔 learning rate를 잘 설정했음에도 학습이 되지 않는 경우에는
    # Non-mormalized inputs인지 의심해봐야함 ( 하나의 변수가 다른 변수에 비해 매우 큰 값 )
    