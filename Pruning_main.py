# -*- coding: utf-8 -*-
"""
@author: Timur Galimzyanov
"""

#%%
import numpy as np
import torch
import torch.nn as nn

import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from tqdm import tqdm
import torch.optim.lr_scheduler as sched

import gc

'''
Loading pruning functions
Loading MLP models and train data generation functions
'''
from prun_functions import prune_model, reprune_model_local, unprune_model, count_non_zero
from mlp_setup import Teacher, Student, gen_train

'''
importing functions for plotting results e.t.c.
'''
from plotting_results import plot_train, loss_landscape1D, loss_landscape2D, plot_landscape1D, plot_landscape2D


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

def seed_everyting(seed):

    '''
    Seed
    '''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#%%

'''
Testing pruning functions
'''

model = Student(n_in = 30, n_out = 5, hidden = 20)
model.to(device)
pruns = np.arange(0.9, -0.01, -0.1)
pruns = np.arange(0.1, 0.9, 0.1)

for i, alpha in enumerate(pruns):

    if i == 0:    
        prune_model(model, alpha=alpha)
    else:
        reprune_model_local(model, alpha=alpha)

    print(f'Amount of nonzero params = {count_non_zero(model):.3f}')

#%%

def train(model, train_loader, optimizer, n_epochs = 5, scheduler = None):

    '''
    Traning function
    
    Returns timeseries of losses - train, validation and OOD
    '''
    

    losses = []
    losses_val = []
    losses_ODD = []
    
    loss_fun = nn.L1Loss()
    #loss_fun = nn.CosineEmbeddingLoss()
    model.train()

    
    for epoch in range(n_epochs):
    
        for features, targets in tqdm(train_loader):
        
            res = model(features.to(device))
            loss = loss_fun(res, targets.to(device)) #, torch.ones(len(targets)).to(device)
            
            with torch.no_grad():   
                res_val = model(X_test.to(device))
                loss_val = loss_fun(res_val, y_test.to(device)).item()
                losses_val.append(loss_val/val_norm)
                
                res_ODD = model(train_ODD.to(device))
                loss_ODD = loss_fun(res_ODD, target_ODD.to(device)).item()
                losses_ODD.append(loss_ODD/ODD_norm)
                
            model.zero_grad()
            loss.backward()
            losses.append(loss.item()/train_norm)
            optimizer.step()
        
        if scheduler != None:
            scheduler.step()
    
    return losses, losses_val, losses_ODD


#%%


train_samples, target_orig, train_ODD, target_ODD = gen_train(new = True)
n_in = train_samples.size()[1]
n_out = target_orig.size()[1]

## Addition of the random noise to the target
std = torch.std(target_orig)
target = target_orig + 0.1*std*torch.randn_like(target_orig)

var = torch.var(target_ODD)
target_ODD = target_ODD + 0.1*var*torch.randn_like(target_ODD)

train_orig_norm = (abs(target_orig)).mean().item()

loss_fun = nn.L1Loss()
print('base loss = ', loss_fun(target, target_orig).item()/train_orig_norm)

seed_everyting(42)

X_train, X_test, y_train, y_test = train_test_split(train_samples, target, test_size = 0.1, train_size = 0.9)
train_set = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

#train_norm = (abs(y_train)).mean().item()
#val_norm = (abs(y_test)).mean().item()
val_norm = train_orig_norm
train_norm = train_orig_norm

ODD_norm = (abs(target_ODD)).mean().item()

#%%

'''
functions for model training
'''

def train_model(train_loader, seed=42, hidden_size = 100, plot_text=''):

    seed_everyting(seed)
    
    model = Student(n_in = n_in, n_out = n_out, hidden = hidden_size)
    #model = torch.load('teacher.pt')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-2, weight_decay=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    scheduler = sched.MultiStepLR(optimizer, milestones=[2,3,4,5,6,7,8,9,10], gamma=0.7)
    #scheduler = None
    
    res = train(model, train_loader, optimizer, n_epochs = 2, scheduler = scheduler)
    print(f'Amount of nonzero params = {count_non_zero(model):.3f}')
    plot_train(res, 2, plot_text)
    
    # model_copy = copy.deepcopy(model)
    return model

'''
functions for model training at consequative prunning levels 
'''

def train_pruning(pruns, train_loader, seed=42, hidden_size = 100, plot_text=''):
    
    seed_everyting(seed)
    
    model = Student(n_in = 30, n_out = 5, hidden = hidden_size)
    #model = torch.load('teacher.pt')
    model.to(device)
    
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
       
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-2, weight_decay=0.00) 
    
    # unprune_model(model)
    
    for i, alpha in enumerate(pruns):
        #seed_everyting(42)
        if i == 0:    
            prune_model(model, alpha=alpha)
        else:
            reprune_model_local(model, alpha=alpha)
                  
        # scheduler = sched.MultiStepLR(optimizer, milestones=[2,3,4,5,6,7,8,9,10], gamma=0.2)
        scheduler = None
        
        res = train(model, train_loader, optimizer, n_epochs = 1, scheduler = scheduler)
        plot_text = f'prun ratio = {alpha:.2f}'
        plot_train(res, 0.5, plot_text)
        
        print(f'Amount of nonzero params = {count_non_zero(model):.3f}')
    
    unprune_model(model)
        
    return model


#%%

'''
Main part where I train the model with concequativly decreasing of prunning ratio
'''

pruns = list(np.arange(0.9, -0.01, -0.1))
pruns = [0.95] + pruns
pruns = [0.9]

train_pruning(pruns, train_loader, seed=1, hidden_size = 200)

#train_model(train_loader, seed=42, hidden_size=200)

#%%

###
###
# Different debugs
###
###

# model = Student(n_in = 30, n_out = 5, hidden = 200)
# model.to(device)

# print(model.lin3.bias)

# optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-2, weight_decay=0.01)
# #train(model, optimizer, n_epochs = 1, scheduler = None)
# prune_model(model, alpha=0.8)

# print(model.lin3.bias)
# print(model.lin3.bias_mask)

# train(model, optimizer, n_epochs = 1, scheduler = None)

# print(model.lin3.bias)
# print(model.lin3.bias_mask)

# reprune_model_local(model, alpha=0.2)

# print(model.lin3.bias)
# print(model.lin3.bias_mask)


#%%
# model = Student(n_in = 30, n_out = 5, hidden = 200)
# old_model = Student(n_in = 30, n_out = 5, hidden = 200)

# model.to(device)

# # torch.save(model, 'old_model.pt')

# #print(model.lin3.bias)
# prune_model(model, alpha=0.2)
# print(model.lin3.bias_mask)
# print(model.lin3.bias)
# print(model.lin3.bias_orig)

# optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-2, weight_decay=0.01)
# train(model, optimizer, n_epochs = 20, scheduler = None)



# print(model.lin3.bias)
# print(model.lin3.bias_orig)

# print('-----Pruning-----')
# reprune_model_local(model, alpha=0.2)
# print(model.lin3.bias_mask)
# print(model.lin3.bias)
# print(model.lin3.bias_orig)

# print(f'Amount of nonzero params = {count_non_zero(model):.3f}')

# #print(list((dict(model.named_buffers()).keys())))

#%%

# model = Student(n_in = n_in, n_out = n_out, hidden = 200)
# print(model)
# model = torch.load('teacher.pt')
# print(model)
# model.to(device)

# loss_fun = nn.L1Loss()
# res_ODD = model(train_ODD.to(device))
# loss_ODD = loss_fun(res_ODD, target_ODD.to(device)).item()

# print(loss_ODD)

# print(f' number of nonzero params = {count_non_zero(model.lin3.weight):.3f}')


#%%

'''
Just looking on the loss landscape of the interpolation of three and two models
'''

model1 = train_model(train_loader, seed=42, hidden_size=200)
model2 = train_model(train_loader, seed=420, hidden_size=200)
model0 = Student(n_in = n_in, n_out = n_out, hidden = 100)
model0 = torch.load('teacher.pt')
model0.to(device)

model0 = train_model(seed=1420, hidden_size=200)

model_temp = Student(n_in = n_in, n_out = n_out, hidden = 200)

res = loss_landscape2D(model0, model1, model2, model_temp, X_train, y_train, train_ODD, target_ODD, train_ODD, target_ODD)
plot_landscape2D(res, n_cont=100, ODD=False)

#%%

losses = loss_landscape1D(model0, model2, X_train, y_train, train_ODD, target_ODD, train_ODD, target_ODD)
plot_landscape1D(losses)

#%%

model1 = train_pruning(seed=1)
model2 = train_pruning(seed=2)

#%%

losses = loss_landscape1D(model1, model2, X_train, y_train, train_ODD, target_ODD, train_ODD, target_ODD)
plot_landscape1D(losses)

#%%

'''
Here I tried to construct a classifier
'''

num_sample = 31

targets = [-5., -3., -1., 1., 3., 5.]

target_samples = []
train_samples = []

for target in targets:
    targer_sample = torch.normal(mean = target, std =  0.5, size=(num_sample, 30))
    train_samples.append(targer_sample)
    target_samples += num_sample*[target]

train_samples = torch.cat(train_samples)
target_samples = torch.tensor(target_samples)

#print(train_probs)

# train_probs = torch.softmax(train_res, dim=1)
# qq = torch.argmax(train_probs, dim=1)
# print(qq)
# print(qq.float().mean().item())


#%%
