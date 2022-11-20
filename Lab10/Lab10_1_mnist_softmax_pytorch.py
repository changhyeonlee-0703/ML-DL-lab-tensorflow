import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

# DL을 하다보면 랜덤하게 초기화하는 경우가 많다.
# 그러나 매번 랜덤하게 초기화될 경우 같은 코드여도 값이 달라지는 경우가 발생할 수 있는데
# 이를 방지하기 위해 seed를 둔다.
torch.manual_seed(777)

learning_rate=0.001
training_epoch = 15
batch_size = 100
# transform : 어떤 형태로 데이터를 불러올 것인가 
# 일반 이미지는 0-255사이의 값을 갖고, (H, W, C)의 형태를 갖는 반면 pytorch는 0-1사이의 값을 가지고 (C, H, W)의 형태를 갖는다. 
# transform에 transforms.ToTensor()를 넣어서 일반 이미지(PIL image)를 pytorch tensor로 변환한다.
mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

# 불러온 데이터셋을 이용해 data_loader 객체 생성
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)


model = torch.nn.Linear(784, 10, bias=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(training_epoch):
    avg_cost =0
    total_batch = len(mnist_train)//batch_size
    
    for i, (batch_xs, batch_ys) in enumerate(data_loader):
        # batch_xs는 (batch_size, 1, 28, 28)의 형태이므로
        # view함수를 사용하여 (batch_size, 28*28)의 형태로 바꾸어 준다.
        X=Variable(batch_xs.view(-1, 28*28))
        Y=Variable(batch_ys)
        
        optimizer.zero_grad() #gradient 값 초기화
        hypothesis = model(X) #학습된 모델을 X에 넣고 결과값 반환
        cost = criterion(hypothesis, Y) #결과값과 레이블 비교해서 loss측정
        cost.backward() # gradient 갱신
        optimizer.step() # parameter 조정
        
        avg_cost +=cost/total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost.data[0]))
    
print('Learning Finished!')

X_test = Variable(mnist_test.test_data.view(-1,28*28).float())
Y_test = Variable(mnist_test.test_labels)

prediction = model(X_test)
correct_prediction = torch.argmax(prediction, 1) == Y_test
#correct_prediction = (torch.max(prediction.data, 1)[1]==Y_test.data)
accuracy = correct_prediction.float().mean()
print('Accuracy:', accuracy)

# Get one and predict
r = random.randint(0, len(mnist_test) - 1)
X_single_data = Variable(mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float())
Y_single_data = Variable(mnist_test.test_labels[r:r + 1])

print("Label: ", Y_single_data.data)
single_prediction = model(X_single_data)
print("Prediction: ", torch.max(single_prediction.data, 1)[1])

