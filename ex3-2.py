#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision
class RBFNet(nn.Module):
    def __init__(self):
        super(RBFNet, self).__init__()
        self.hidden_dim = 100
        self.output_dim = 4
        self.num_centers = 4

        self.centers = nn.Parameter(torch.randn(4, 12288))
        self.beta = nn.Parameter(torch.ones(num_centers, 1) / num_centers)
        self.sigma = sigma

        self.fc = nn.Linear(num_centers, output_dim)

    def radial_basis(self, x):
        C = self.centers.view(self.num_centers, -1)
        return torch.exp(-torch.sum((x - C) ** 2, dim=1) / (2 * self.sigma ** 2))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        H = self.radial_basis(x)
        out = self.fc(H)
        return out
hidden_dim = 100
output_dim = 4
num_centers= 4
sigma = 1.0
import torchvision.transforms as transforms
import torchvision.datasets as datasets
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder('C:/Users/dllab.DLL-038/Downloads/Pistachio_Image_Dataset/Pistachio_Image_Dataset/Pistachio_Image_Dataset', transform=train_transforms)
test_dataset = datasets.ImageFolder('C:/Users/dllab.DLL-038/Downloads/Pistachio_Image_Dataset/Pistachio_Image_Dataset/Pistachio_Image_Dataset', transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape
#shape12288
criterion = nn.CrossEntropyLoss()
# specify optimizer
model=RBFNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# number of epochs to train the model
n_epochs = 10  # suggest training between 20-50 epochs
for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        #optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        #model=RBF(data)
        model.train()
        output = model(data)
        # calculate the loss
        loss = criterion(output.float(), target.float())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        #optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)

    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1,train_loss))


# In[2]:


pip install torchvision


# In[3]:


get_ipython().system('pip install torchvision')


# In[ ]:




