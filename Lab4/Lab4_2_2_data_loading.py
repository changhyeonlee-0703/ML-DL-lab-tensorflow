# 파일이 굉장히 커서 메모리에 한번에 올리기 어렵다?
# 이런 경우 tensorflow는 Queue Runners라는 시스템이 있음
# Queue에 쌓고 reader1 -> decoder로 어떤 배치만큼 읽어와서 학습을 시킴
# 굉장히 큰 데이터를 쓸 때 유용하게 쓸 수 있음

import tensorflow.compat.v1 as tf
import os

dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir))

filename_queue = tf.train.string_input_producer([os.path.join(dir, 'data-01-test-score.csv')], shuffle = False, name='filename_queue')

# reader 부분 정의
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]] # 값 변수들의 데이터타입을 정의해줌. float타입
xy = tf.decode_csv(value, record_default=record_defaults) # csv타입으로 디코드하고 데이터 타입을 명시

# X, Y로 나누고 총 데이터를 10개로 쪼갠다.(batch=10)
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)


# 추후 수정
# 현재 버전에서 쓰이는지 확인해봐야함.(너무 오래된 버전임)