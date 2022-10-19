import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 512 
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()
  
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname

class cs19b047NN(nn.Module):
    def _init_(self, input_size, hidden_size, num_classes):
        super(cs19b047NN, self)._init_()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
  model = cs19b047NN(input_size, hidden_size, num_classes).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
  
  n_total_steps = len(train_data_loader)
  
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_data_loader):  
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
       
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    
  print ('Returning model... (rollnumber: cs19b047)')
  return model


            
            

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = cs19b047NN(config)
  # batch_size = 64
  optimizer = torch.optim.SGD(model.parameters(), lr = lr)
  loss_fn = nn.CrossEntropyLoss()

  for e in range(n_epochs):
      model.train()
      totalTrainLoss = 0
      totalValLoss = 0
      trainCorrect = 0
      valCorrect = 0
      for (x, y) in train_data_loader:
          (x, y) = (x.to(device), y.to(device))
          pred = model(x)
          loss = loss_fn(pred, y)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          totalTrainLoss += loss
          trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
  print ('Returning model... (rollnumber: xx)')
  return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):
  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  
  with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    accuracy_val= 100.0 * n_correct / n_samples
   
  
  print ('Returning metrics... (rollnumber: cs19b047)')
  return accuracy_val, precision_val, recall_val, f1score_val
