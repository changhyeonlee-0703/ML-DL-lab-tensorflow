import tensorflow.compat.v1 as tf

# 이번엔 위에서 수동으로 했던 gradient Descent algorithm을 tf안에 있는 함수들을 사용하여 구현함.
with tf.compat.v1.Session() as sess:

    # 처음 W를 5로 두었음
    W = tf.Variable(5.0,name='weight')
    X = [1,2,3]
    Y = [1,2,3]

    hypothesis = W*X

    cost = tf.reduce_mean(tf.square(hypothesis-Y))

    # Gradient Descent Magic
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)

    sess.run(tf.global_variables_initializer())
    
    # 수동으로 W를 수정하고
    # 그것을 cost function에 적용함
    for step in range(100):
        print(step, sess.run(W))
        sess.run(train)

    # 최종 결과로 1.0으로 바뀐 것을 볼 수 있음.
