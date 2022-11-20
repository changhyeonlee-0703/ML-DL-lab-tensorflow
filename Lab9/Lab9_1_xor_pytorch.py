import torch
from torch.autograd import Variable
# torch.autograd 는 자동 미분
import numpy as np


torch.manual_seed(777)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = Variable(torch.from_numpy(x_data))
Y = Variable(torch.from_numpy(y_data))

# Hypothesis 설계
# torch.nn.Linear(2,1, bias=True)에서 
# 2는 input feature를 1은 output feature를 뜻함
linear = torch.nn.Linear(2,1, bias=True)
sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear, sigmoid)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(10001):
    optimizer.zero_grad() # gradient(미분으로 나온 변화도?)을 backpropagation할 때 계속 더해주므로 
    # gradient를 매번 초기화가 필요하다. 여기서 W, b는 gradient가 아님을 유의
    hypothesis = model(X)
    
    # cost fucntion
    cost = -(Y*torch.log(hypothesis)+(1-Y)*torch.log(1-hypothesis)).mean()
    cost.backward() # cost.backward()함수를 통해 W, b에 대한 gradient를 계산
    optimizer.step() # optimizer.step()을 통해, W, b를 gradeint로 조정한다.
    
    if step%100==0:
        print(step, cost.data.numpy())
        
    # accuarcy
predicted = (model(X).data>0.5).float()
accuracy = (predicted == Y.data).float().mean()
print("\nHypothesis: ", hypothesis.data.numpy(), "\nCorrect: ", predicted.numpy(), "\nAccuracy: ", accuracy)

# 버그가 없음애도 불구하고 정확도가 낮음.
# 정확도가 0.5로 결과가 안좋음.
# 잘 안되는 이유는 하나의 layer만 사용했기 때문이다.

