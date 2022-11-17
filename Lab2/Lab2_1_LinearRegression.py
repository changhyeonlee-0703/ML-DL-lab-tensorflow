import tensorflow.compat.v1 as tf

with tf.compat.v1.Session() as sess:
    X_train = [1,2,3]
    y_train = [1,2,3]
    
    
    # 기존의 프로그램에서 변수와는 다름. 
    # 텐서플로우가 사용하는 변수이다. 텐서플로우를 실행시키면 자체적으로 변경시키는 값이다.
    # 또한 trainable한 변수다라고 보면 좋다.
    # tf.random_normal([1])이란 1차원의 랜덤 값을 초기값으로 선언해줌.
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    
    # Our hypothesis XW+b
    hypothesis = X_train*W+b
    
    # cost function은 어떻게 코드화하는가?
    # tf.reduce_mean은 평균화시켜주는 tf 내장 함수
    # t= [1,2,3,4]
    # tf.reduce_mean(t) ==> 2.5 
    cost = tf.reduce_mean(tf.square(hypothesis - y_train))
    
    
    # Gradient Descent
    # 이제 cost function을 최소화해줘야한다.
    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)
    
    # 그래프가 구현되었으면 sess을 만들고 실행시켜야함.
    # variable을 사용하기 위해선 그전에
    # 반드시 tf.global_variables_initializer() 실행시켜줘야함.
    sess.run(tf.global_variables_initializer())
    
    
    for step in range(2001):
        sess.run(train)
        if step%100 ==0:
            print(step, sess.run(cost), sess.run(W), sess.run(b))
    