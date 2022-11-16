import tensorflow as tf

# hello = tf.constant("Hello, ")
# Tensorflow = tf.constant("TensorFlow!")
# helloTensor = hello + Tensorflow
# #tf.Session() 대신에 TF2.0이상부턴 tf.compat.v2.Session()을 사용해야함
# #sess = tf.compat.v1.Session()

# #또한 sess = tf.
# with tf.compat.v1.Session() as sess:
#     print(sess.run(helloTensor))

 # Launch the graph in a session.
with tf.compat.v1.Session() as ses:

     # Build a graph.
     a = tf.constant('hello, ')
     b = tf.constant('tensor')
     c = a + b

     # Evaluate the tensor `c`.
     print(ses.run(c))