# Lab1_2_node.py에서는 tensorflow1.0 버전이므로 placeholder가 사용이 가능했다.
# 그러나 현재 tensorflow2.0 버전에선 꽤 많은 변화가 일어났으므로 변화에 맞는 코드로 수정해봤다.

import tensorflow as tf

node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)


# tf2.0에선 session.run을 할필요가 없다.
# tf.print에서 직접 sesssion.run을 실행시켜준다.
tf.print(node1, node2)
tf.print(node3)

# placeholder로 변수 노드를 만들지 않고
# @tf.function을 이용하여 함수를 정의함으로써 훨씬 간결하게 처리 가능하다.
@tf.function
def adder(a,b):
    return a+b

a = tf.constant(1)
b = tf.constant(2)
print(adder(a,b))
tf.print(adder(a,b))

c = tf.constant([1,2])
d = tf.constant([3,4])
print(adder(c,d))
tf.print(adder(c,d))