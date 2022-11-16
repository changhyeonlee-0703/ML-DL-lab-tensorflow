import tensorflow as tf
#import tensorflow.compat.v1 as tf

with tf.compat.v1.Session() as ses:
    # 그래프를 빌드했음
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    node3 = tf.add(node1, node2) # node1 +node2 로도 표현 가능
    
    
    # 그래프의 하나의 요소기 때문에 결과값이 나오는 것이 아닌 텐서가 나옴
    # node1 :  Tensor("Const:0", shape=(), dtype=float32) node2 :  Tensor("Const_1:0", shape=(), dtype=float32)
    # node3 :  Tensor("Add:0", shape=(), dtype=float32)
    print('node1 : ', node1, 'node2 : ', node2) 
    print('node3 : ', node3)
    
    # 실행시키기 위해서는 세션을 만들고 런을 통해서 실행을 시킴
    # sess.run(node1, node2) :  [3.0, 4.0]
    # sess.run(node3):  7.0
    print('sess.run(node1, node2) : ', ses.run([node1, node2]))
    print("sess.run(node3): ", ses.run(node3))


    # 상수로 넣지 말고 변수로 넣는 그래프를 만든다면? placeholder란 노드를 사용
    # placeholder라는 특수한 노드를 만든다.(tf 1.0만 가능)
    a = tf.placeholder(tf.float32) 
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    
    # feed_dict으로 값을 넘겨 받을 수 있다.
    print(ses.run(adder_node, feed_dict={a:3, b:4.5}))
    print(ses.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))
    

# 텐서라는 것은 [[1,2,3,], [4,5,6], [7,8,9]] 이런 array를 말한다.
# 텐서는 rank, shape type

# rank는 몇차원 array이냐?
# 1차원 = scalar s=463 , 2차원 Vector v=[1.1, 2.2, 3.3], 3차원 = Matrix m=[[1,2,3],[4,5,6],[7,8,9]]
# 3차원 = 3-Tensor, n차원 = n-Tensor


# shape은 각각의 element에 몇개씩 들어있느냐?
# [D0]= [5], [D0,D1] = [3,4]
# [[1,2,3,], [4,5,6], [7,8,9]] = shape은 (3,3) or [3,3]이 된다. 
# 뒤에 3이 내부 리스트 안의 엘리먼트 갯수 ex [1,2,3]에서 1,2,3이므로 갯수는 3
# 앞의 3이 전체 [[***],[***],[***]] 갯수[***] 3개이므로 3

# type은 거의 tf.float32를 대부분 사용함 또는 tf.int32