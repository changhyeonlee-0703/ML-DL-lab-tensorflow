import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

# 직접 cost function의 값이 어떻게 변하는지를 확인해보자.
with tf.compat.v1.Session() as sess:
    X = [1,2,3]
    Y = [1,2,3]
    
    W = tf.placeholder(tf.float32)
    
    hypothesis = X*W
    
    cost = tf.reduce_mean(tf.square(hypothesis-Y))
    
    sess.run(tf.global_variables_initializer())
    
    W_val = []
    cost_val = []
    
    # cost function의 값을 기록.
    for i in range(-30,50):
        feed_W = i*0.1
        curr_cost, curr_W = sess.run([cost,W], feed_dict={W:feed_W})
        W_val.append(curr_W)
        cost_val.append(curr_cost)
        
    plt.plot(W_val, cost_val)
    plt.show()