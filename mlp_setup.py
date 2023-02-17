# -*- coding: utf-8 -*-
"""
@author: Timur Galimzyanov

MLP models and train data generation functions

"""

#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

def seed_everything(seed):

    '''
    Seed everything
    '''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Teacher(nn.Module):
    
    '''
    Larger teacher model - three layers perceptron
    '''
    
    def __init__(self, n_in, n_out, hidden):
        super().__init__()

        self.lin1 =  nn.Linear(n_in, hidden)
        self.lin2 =  nn.Linear(hidden, hidden)
        self.lin3 =  nn.Linear(hidden, n_out)
        
    def forward(self, x):
                
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))      
        x = self.lin3(x)
        #x = 0.1*x/torch.norm(x)
        
        return x

class Student(nn.Module):
    
    '''
    Smalle student model - three layers perceptron
    '''
    
    
    def __init__(self, n_in, n_out, hidden):
        super().__init__()

        self.lin1 =  nn.Linear(n_in, hidden)
        self.lin2 =  nn.Linear(hidden, hidden)
        self.lin3 =  nn.Linear(hidden, n_out)
        
    def forward(self, x):
                
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))      
        x = self.lin3(x)
        #x = 0.1*x/torch.norm(x)
        
        return x


def gen_train(hidden_size = 200, num_sample = 1000, seed=42, new = False):

    '''
    initializatin of the model
    '''

    '''
    If new
    
    Generates and saves data for training the student
    Initializes and saves teacher model
    generates feature from uniform distrbution
    calculates target
    
    generates OOD features from unifrom distribution outside first one
    '''
    
    '''
    If not new, reads all this data from files
    '''

    if new:
        
        seed_everything(seed)
        n_in = 30
        n_out = 5
        hidden = hidden_size
    
        teacher = Teacher(n_in = n_in, n_out = n_out, hidden = hidden)
        torch.save(teacher, 'teacher.pt')
    
        '''
        generation of the train and target
        '''
    
        x0 = -5
        x1 = 5 
    
        train_samples = (x1-x0)*(torch.rand(num_sample, n_in)) + x0
        target = teacher(train_samples)
    
        torch.save(train_samples, 'train.pt')
        torch.save(target, 'target.pt')
        
        x0 = 10
        x1 = 20
    
        train_samples = (x1-x0)*(torch.rand(10000, n_in)) + x0
        target = teacher(train_samples)
    
        torch.save(train_samples, 'train_ODD.pt')
        torch.save(target, 'target_ODD.pt')
    
    train_samples = torch.load('train.pt')#.to(device)
    target = torch.load('target.pt')#.to(device)
    target.requires_grad = False

    train_ODD = torch.load('train_ODD.pt')#.to(device)
    target_ODD = torch.load('target_ODD.pt')#.to(device)
    target_ODD.requires_grad = False
    
    return train_samples, target, train_ODD, target_ODD

#%%
