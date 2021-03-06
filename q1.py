import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
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
    
print('Using PyTorch version:', torch.__version__, ' Device:', device, file=sys.stderr)

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
        x = torch.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x), dim=1)

model = Net(32*32*3,100,10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model,file=sys.stderr)

epochs = 50

def train():
    import modeltraining as mt
    losstr = []
    lossv, accv, = [], []
    losst, acct = [], []
    for epoch in range(1, epochs + 1):
        mt.train(model,optimizer,criterion,train_loader,epoch,losstr,log_interval=1000)
        mt.validate(model,criterion,validation_loader,lossv, accv)
        mt.validate(model,criterion,test_loader,losst, acct,testing=True)

    return acct[-1].data.item(), losstr,lossv,accv,losst,acct

def plot(losstr,lossv,accv,losst,acct):
    plt.plot(np.arange(1,epochs+1), losstr)
    plt.plot(np.arange(1,epochs+1), accv)
    plt.title('Loss & Accuracy over Epochs\nActivation: Sigmoid\nLearning_rate={}'.format(learning_rate))
    plt.legend(['Training Loss','Validation Accuracy (%)'])

    plt.savefig('images/q1_lr_{}.png'.format(learning_rate))

    return plt

if __name__ == "__main__":
    end_acc,losstr,lossv,accv,losst,acct = train()
    print('Final testing acc after {} epochs: {}'.format(epochs,end_acc),file=sys.stdout)

    import os
    os.makedirs('images/',exist_ok=True)
    plt = plot(losstr,lossv,accv,losst,acct)