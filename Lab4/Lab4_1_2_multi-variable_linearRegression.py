import tensorflow.compat.v1 as tf

# X데이터를 메트릭스로 구현
with tf.compat.v1.Session() as sess:
    x_data = [[73., 80., 75.],
              [88., 93., 93.],
              [75., 93., 90.],
              [95., 98., 100,],
              [73., 66., 70.]]
    y_data = [[152.], [185.], [180.], [196.], [142.]]
    
    X = tf.placeholder(tf.float32, shape=[None,3]) # shape=[행(instance)=data 수, variable]
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    W = tf.Variable(tf.random_normal([3,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    
    hypothesis =tf.matmul(X,W) + b #tf.matmul은 matrix 곱
    
    cost = tf.reduce_mean(tf.square(hypothesis-Y))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)
    
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, hy_val, _= sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
        if step%100 == 0 :
            print(step, "Cost : ", cost_val, "\nPrediction : \n", hy_val)
            
            
    # 다음으로는 다른 파일에 있는 데이터를 로딩