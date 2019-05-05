import torch
import sys

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def train(model,optimizer,criterion,train_loader,epoch,loss_vector,log_interval=200):
    model.train()
    allloss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device=device,dtype=torch.int64)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)
        allloss+=loss.data.item()

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()),file=sys.stderr)
    loss_vector.append(allloss/len(train_loader))

def validate(model,criterion,validation_loader,loss_vector, accuracy_vector, testing=False):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device=device,dtype=torch.int64)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    err_type='Validation'
    if(testing):
        err_type='Testing'
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(err_type,
        val_loss, correct, len(validation_loader.dataset), accuracy),file=sys.stderr)
