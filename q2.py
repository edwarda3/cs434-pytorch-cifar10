import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import seaborn as sns
sns.set()
import sys

import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('lr',help='Learning rate for the model')
args = parser.parse_args()
learning_rate = float(args.lr)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device,file=sys.stderr)

batch_size = 32

import loader
train_loader, validation_loader = loader.get_training_validation_loaders()
test_loader = loader.get_testing_loader()


class Net(nn.Module):
    def __init__(self,d_in,d_hidden,d_out):
        super(Net, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = x.view(-1, self.d_in).float()
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x), dim=1)

model = Net(32*32*3,100,10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model,file=sys.stdout)

epochs = 10

def train():
    import modeltraining as mt
    losstr = []
    lossv, accv, = [], []
    losst, acct = [], []
    for epoch in range(1, epochs + 1):
        mt.train(model,optimizer,criterion,train_loader,epoch,losstr,log_interval=1000)
        mt.validate(model,criterion,validation_loader,lossv, accv)
        mt.validate(model,criterion,test_loader,losst, acct)

    return acct[-1].data.item(), losstr,lossv,accv,losst,acct

def plot():
    plt.plot(np.arange(1,epochs+1), losstr)
    plt.plot(np.arange(1,epochs+1), accv)
    plt.title('Loss & Accuracy over Epochs\nActivation: Relu\nLearning_rate={}'.format(learning_rate))
    plt.legend(['Training Loss','Validation Accuracy (%)'])

    plt.savefig('images/q2_lr_{}.png'.format(learning_rate))
    plt.show()

if __name__ == "__main__":
    end_acc,losstr,lossv,accv,losst,acct = train()
    print(end_acc,file=sys.stdout)
    plot(losstr,lossv,accv,losst,acct)