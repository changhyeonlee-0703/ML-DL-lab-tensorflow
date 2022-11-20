import tensorflow.compat.v1 as tf
import numpy as np

def MinMaxScaler(data):
    numerator = data - np.min(data, axis=0) #axis=0은 열, axis=1은 행
    denominator = np.max(data, axis=0) - np.min(data, axis=0)
    return numerator / (denominator + 1e-7)

# 가끔 learning rate를 잘 설정했음에도 학습이 되지 않는 경우에는
# Non-mormalized inputs인지 의심해봐야함 ( 하나의 변수가 다른 변수에 비해 매우 큰 값 )
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

# normalized 함.
xy=MinMaxScaler(xy)

x_data = xy[:, 0:-1]
y_data =xy[:, [-1]]


with tf.compat.v1.Session() as sess:
    X = tf.placeholder(tf.float32, shape=[None, 4])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    W = tf.Variable(tf.random_normal([4,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    
    hypothesis = tf.matmul(X,W) + b
    cost = tf.reduce_mean(tf.square(hypothesis-Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)
    
    prediction = tf.arg_max(hypothesis, 1)
    is_correct = tf.equal(prediction, tf.arg_max(Y,1))
    accuarcy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    # training
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, optimizer], feed_dict={X:x_data, Y:y_data})
        print( step, cost_val, hy_val)
        
    

    