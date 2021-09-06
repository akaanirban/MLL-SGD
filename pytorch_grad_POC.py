#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 21:29:00 2021

@author: Anirban Das
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset
np.random.seed(0)
torch.random.manual_seed(0)

def computeFullDataGradient(loader, model, criterion):
    data_size = len(loader.dataset)
    batch_size = loader.batch_size
    correction_factor = (batch_size * 1.0 / data_size)
    full_gradVec = list()
    for i, (data, target) in enumerate(loader):

        input_var = data
        target_var = target
        output = model(input_var.float())
        loss = criterion(output, target_var.long())
        batch_gradVec = torch.autograd.grad(loss, model.parameters())
        if i == 0:
            for param in batch_gradVec:
                full_gradVec.append(torch.zeros_like(param))

        for param1, param2 in zip(full_gradVec, batch_gradVec):
            param1 += correction_factor * param2
    model.zero_grad()
    return full_gradVec

def myFullDataGradient(loader, model, criterion):
    correction_factor = 1.0/len(loader)
    for i, (d, t) in enumerate(loader):
        input_var = d
        target_var = t
        output = model(input_var.float())
        loss1 = criterion(output, target_var.long()) * correction_factor
        loss1.backward()
    temp_gradients = []
    for name, param in model.named_parameters():
        temp_gradients.append(param.grad.data.detach().clone())
    model.zero_grad()
    return temp_gradients


data = torch.tensor(np.random.randn(12, 3))
target = torch.tensor(np.random.randint(0,2,12))
data
l = torch.nn.Linear(3,2)
model = torch.nn.Linear(3,2)
loss = torch.nn.CrossEntropyLoss()

dataloader2 = torch.utils.data.DataLoader(
        TensorDataset(data, target),
        batch_size=4)
grads1 = computeFullDataGradient(dataloader2, model, loss)
print("Batch size 3: other method", grads1)
grads2 = myFullDataGradient(dataloader2, model, loss)
print("Batch size 3: my method", grads2)

dataloader1 = torch.utils.data.DataLoader(
        TensorDataset(data, target),
        batch_size=4)
grads1 = computeFullDataGradient(dataloader1, model, loss)
print("Batch size full dataset: other method", grads1)

grads2 = myFullDataGradient(dataloader1, model, loss)
print("Batch size full dataset: my method", grads2)



