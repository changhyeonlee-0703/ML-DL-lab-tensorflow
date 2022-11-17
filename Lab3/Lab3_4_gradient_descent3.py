import tensorflow.compat.v1 as tf

# 만약 gradient를 약간 수정하고 싶을 때(19번째 줄)
# gradient를 임의로 조정하고 싶을 때
with tf.compat.v1.Session() as sess:

    # 처음 W를 5로 두었음
    W = tf.Variable(5.0,name='weight')
    X = [1,2,3]
    Y = [1,2,3]

    hypothesis = W*X

    
    cost = tf.reduce_mean(tf.square(hypothesis-Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    
    # compute_gradients함수는 optimizer로 계산한 gradeint를 가져올 때 사용한다.
    gvs = optimizer.compute_gradients(cost)
    # gvs로 가져온 gradient를 내가 원하는 방법으로 수정한 후
    # apply_gradients()로 적용한다.
    apply_gradients = optimizer.apply_gradients(gvs)
    
    # 직접 gradient를 계산해서 위에 gvs와 같은지 비교해보자.
    gradient= tf.reduce_mean((W*X-Y)*X)*2
    
    train = optimizer.minimize(cost)

    sess.run(tf.global_variables_initializer())
    
    # 수동으로 W를 수정하고
    # 그것을 cost function에 적용함
    for step in range(100):
        print(step, sess.run([gradient, W, gvs]))
        sess.run(apply_gradients)

    # 최종 결과로 1.0으로 바뀐 것을 볼 수 있음.
