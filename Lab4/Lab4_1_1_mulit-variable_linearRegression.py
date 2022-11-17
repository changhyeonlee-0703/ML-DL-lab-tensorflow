import tensorflow.compat.v1 as tf

# 실제 머신러닝에서 많이 사용하는 다양한 변수를 활용한 회귀분석 구현
with tf.compat.v1.Session() as sess:
    x1_data = [73., 93., 89., 96., 73.]
    x2_data = [80., 88., 91., 98., 66.]
    x3_data = [75., 93., 90., 100., 70.]
    y_data = [152., 185., 180., 196., 142.]
    
    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    x3 = tf.placeholder(tf.float32)
    
    Y = tf.placeholder(tf.float32)
    
    w1 = tf.Variable(tf.random_normal([1]), name='weight1')
    w2 = tf.Variable(tf.random_normal([1]), name='weight2')
    w3 = tf.Variable(tf.random_normal([1]), name='weight3')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    
    hypothesis = x1*w1 + x2*w2 + x3*w3 + b
    
    cost = tf.reduce_mean(tf.square(hypothesis-Y))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)
    
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, hy_val, _= sess.run([cost, hypothesis, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, Y:y_data})
        if step%100 == 0 :
            print(step, "Cost : ", cost_val, "\nPrediction : \n", hy_val)
    
    # 이런 방법은 거의 사용하지 않는다. 왜냐면 데이터가 많아지면 코드로 다 쓰기 어렵기 때문
    # 먼저 x 데이터를 matrix로 구현해보자.