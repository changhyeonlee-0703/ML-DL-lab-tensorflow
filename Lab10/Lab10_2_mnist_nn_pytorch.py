import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

torch.manual_seed(777)

learning_rate=0.001
training_epoch = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)

# neuarl networks layers
linear1 = torch.nn.Linear(784, 256, bias=True)
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 10, bias=True)
relu = torch.nn.ReLU()

model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(training_epoch):
    avg_cost =0
    total_batch = len(mnist_train)//batch_size
    
    for i, (batch_xs, batch_ys) in enumerate(data_loader):
        X=Variable(batch_xs.view(-1, 28*28))
        Y=Variable(batch_ys)
        
        optimizer.zero_grad() #gradient 값 초기화
        hypothesis = model(X) #학습된 모델을 X에 넣고 결과값 반환
        cost = criterion(hypothesis, Y) #결과값과 레이블 비교해서 loss측정
        cost.backward() # gradient 갱신
        optimizer.step() # parameter 조정
        
        avg_cost +=cost/total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
print('Learning Finished!')

X_test = Variable(mnist_test.test_data.view(-1,28*28).float())
Y_test = Variable(mnist_test.test_labels)

prediction = model(X_test)
correct_prediction = torch.argmax(prediction, 1) == Y_test
accuracy = correct_prediction.float().mean()
print('Accuracy:', accuracy)

# Get one and predict
r = random.randint(0, len(mnist_test) - 1)
X_single_data = Variable(mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float())
Y_single_data = Variable(mnist_test.test_labels[r:r + 1])

print("Label: ", Y_single_data.data)
single_prediction = model(X_single_data)
print("Prediction: ", torch.max(single_prediction.data, 1)[1])

