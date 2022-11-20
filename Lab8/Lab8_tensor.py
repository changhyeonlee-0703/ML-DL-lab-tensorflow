import numpy as np
import tensorflow.compat.v1 as tf

t = np.array([0., 1., 2., 3., 4., 5., 6.,])

pp.pprint(t)
print(t.ndim) # 차원(rank)
print(t.shape) # shape

print(t[2:5], t[4:-1]) # slicing


tf.constant([1,2,3,4]) # tf.constant는 배열을 넣는 함수

[[[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24]]]]
# 일때 차원(rank)는 여는 대괄호 갯수 = 4
# shape는 [1,2,3,4] 로 우선 안의 요소들의 갯수는 차원 값과 같다.
# 제일 안쪽에 대괄호에서 4이므로 마지막에 4가 들어가고 그 다음 대괄호가 3개이므로
# [1,2,3,4] 이다.
# axis는 일 때 0은 행, 1은 열 이렇게 한단계씩 들어간다.


# tf.reduce_mean()이란?
# 줄여서 구한다란 뜻.
tf.reduce_mean([1,2], axis=0).eval()
# 1.5가 나올텐데 왜? 지금 Integer형이기 떄문

x= [[1., 2.],
    [3., 4.]]

tf.reduce_mean(x).eval()
# 결과는 2.5 모든 것에 평균을 구해라

tf.reduce_mean(x, axis=0).eval()
# 결과는 [2., 3.]

tf.reduce_mean(x, axis=1).eval()
# 결과는 [1.5, 3.5]

tf.reduce_mean(x, axis=-1).eval()
# 가장 안쪽에 있는 애들의 평균을 구해라
# [1.5, 3.5]


x= [[0,1,2],
    [2,1,0]]

tf.argmax(x, axis=0).eval() #가장 맥시멈값의 index를 구해줌.
# [1,0,0]

tf.argmax(x, axis=1).eval()
# [2,0]

#reshape

t=np.array[[[0,1,2], [3,4,5]]
           [[6,7,8], [9,10,11]]]

t.shape
#(2,2,3)

tf.reshape(t, shape=[-1, 3]).eval()
# 보통 가장 안쪽에 있는 값인 3을 건들지 않고 -1은 그 외는 자동으로 shape을 조정해줘

tf.reshape(t, shape=[-1,1,3]).eval()
# rank를 늘릴 떄

#Reshape(squeeze, expand)
tf.squeeze([[0], [1], [2]]).eval()
# [0,1,2]

tf.expand_dims([0,1,2], 1).eval()
# [[0], [1], [2]]
# 차원을 추가하고 싶을 떄

t= tf.one_hot([[0],[1], [2],[0]], depth=3)
# [[[1., 0., 0.]],
#  [[0., 1., 0.]],
#  [[0., 0., 1.]],
#  [[1., 0., 0.]]]
# 이렇게 원핫으로 바꿔준다. 차원이 하나더 추가됨을 확인할 수 있음

tf.reshpae(t, shape=[-1,3]).eval()
# 로 차원을 축소시킨다.


# Casting
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
# [1,2,3,4]

tf.cast([True, False, 1==1, 0==1], tf.int32).eval()
# [1,0,1,0]

# 모양이 똑같은 텐서로 만들고 싶을 떄
# tf.ones_like(x)