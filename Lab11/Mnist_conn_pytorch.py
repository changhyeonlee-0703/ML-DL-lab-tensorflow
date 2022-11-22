import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.init

torch.manual_seed(777)

learning_rate = 0.01
training_epochs = 15
batch_size = 100

mnist_train = dset.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dset.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)


data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
keep_prob=0.7
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self._build_net()
        
    def _build_net(self):
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        # nn.Conv2d(input_channel수, output_channel수, kernel_size=filter크기)
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    torch.nn.Dropout(p=1-keep_prob))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Dropout(p=1-keep_prob))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Dropout(p=1-keep_prob))
        
        self.fc1 = nn.Linear(4*4*128, 625, bias=True)
        nn.init.xavier_uniform(self.fc.weight)
        self.layer4 = nn.Sequential(self.fc1,
                                    nn.ReLU(),
                                    nn.Dropout(p=1-keep_prob))
        self.fc2 = nn.Linear(625, 10, bias=True)
        nn.init.xavier_uniform(self.fc2.weight)
        
        self.criterion = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1) #fc층을 위해 평평하게 만들어준다. 일자로 길게 쭉
        out = self.layer4(out)
        out = self.fc2(out)
        
        return out
    
    def predict(self, x):
        self.eval() # 평가 모드
        return self.forward(x)
    
    def get_accuracy(self, x, y):
        prediction = self.predict(x)
        correct_prediction = (torch.max(prediction.data, 1)[1]==y.data)
        self.accuracy = correct_prediction.float().mean()
        return self.accuracy
    
    def train_model(self, x, y):
        self.train() # 훈련 모드
        self.optimizer.zero_grad()
        self.hypothesis = self.forward(x)
        self.cost =self.criterion(self.hypothesis, Y)
        self.cost.backward()
        self.optimizer.step()
        
        return cost
    
model = CNN()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(mnist_train)//batch_size
    for i, (batch_xs, batch_ys) in enumerate(data_loader):
        X = Variable(batch_xs)
        Y = Variable(batch_ys)
        
        cost = model.train_model(X,Y)
        
        avg_cost += cost/total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
model.eval() # 모델 평가 모드는 dropout이 false가 된다.
 
X_test = Variable(mnist_test.data.view(len(mnist_test), 1, 28, 28, 1, 28,28)).float()
Y_test = Variable(mnist_test.targets)

print(model.get_accuracy(X_test,Y_test))