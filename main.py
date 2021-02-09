import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets

class Net(nn.Module):
    def __init__(self):
        # initialize nn.module
        super().__init__()

        # define our layers
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # Rectified linear activation function over each layer
        ## Each F.relu is run on the output of the respective layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


net = Net()
#print(net)

X = torch.rand((28,28))
X = X.view(-1, 784)
output = net(X)

#print(output)

# (param) net.parameters() -- everything adjustable
# (param) lr=0.001 -- learning rate, which dictates the size of the step the optimizer will take to get
##                    to the best state
optimizer = optim.Adam(net.parameters(), lr=0.001)

# MNIST is a hand drawn number dataset (28 x 28)
train = datasets.MNIST("", train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

# batch_size: how many pieces we're feeding to our model at a time
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# an Epoch is how many passes we make through our data
EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of featuresets and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()  # back propagation
        optimizer.step()  # adjust weights for us
    print(loss)


correct = 0
total = 0


with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("accuracy: ", round(correct/total, 3))


plt.imshow(X[0].view(28,28))
plt.show()

print("prediction: ", torch.argmax(net(X[0].view(-1, 784)[0])))

def print_hi():
    # MNIST is a hand drawn number dataset (28 x 28)
    train = datasets.MNIST("", train=True, download=True,
                           transform = transforms.Compose([transforms.ToTensor()]))

    test = datasets.MNIST("", train=False, download=True,
                          transform = transforms.Compose([transforms.ToTensor()]))

    # batch_size: how many pieces we're feeding to our model at a time
    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

    for data in trainset:
        #print( data )
        break

    total = 0
    # counter_dict is used to track how many of each drawn number we have
    counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    for data in trainset:
        Xs, Ys = data
        for y in Ys:
            counter_dict[int(y)] += 1

    print( counter_dict )

    #x, y = data[0][0], data[1][0]
    #plt.imshow(data[0][0].view(28,28))
    #plt.show()

# Press the green button in the gutter to run the script.
##if __name__ == '__main__':
##    print_hi()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
