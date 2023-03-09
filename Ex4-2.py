#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("Pistachio_28_Features_Dataset.csv")
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
class KSOM(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(KSOM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = torch.randn(input_dim,output_dim)

    def forward(self, x):
        distances = torch.sum((x - self.weights)**2, axis=1)
        winner = torch.argmin(distances)
        return winner
    
    def weight_updation(self, x, winner, lr=0.02):
        d = x
        for i in range(self.output_dim):
            if i == winner:
                self.weights[i] += lr * d[i]
            return self.weights[i]
model = KSOM(input_dim=28, output_dim=28)
for epoch in range(2):
    for o in x:
        o = torch.Tensor(o)
        winner = model(o)
        print(model.weight_updation(o, winner, lr=0.2))


# In[ ]:




