# -*- coding: utf-8 -*-
"""
@author: Timur Galimzyanov

functions for plotting results et cetera

"""

#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

def plot_train(res, lim = 0.5, text=''):
    
    '''
    Plots train and test losses
    '''
    
    losses, losses_val, losses_ODD = res
    
    print('Train, ODD = ', sum(losses[-50:])/50, sum(losses_ODD[-50:])/50)
    plt.title(text)
    plt.plot(losses[:], label="Train")    
    plt.plot(losses_ODD[:], label="OOD")
    plt.plot(losses_val[:], label="Val")
    plt.ylim([0, lim])
    plt.show()

def tau(alpha, theta1, theta2):
    ## just a linear interpolation function
    ## alpha = 0 corresponds to the optimal solution
    return (1 - alpha)*theta2 + alpha*theta1

def tau2(alpha1, alpha2, theta1, theta2, theta3):
    ## just a linear interpolation function
    ## alpha_i = 0 corresponds to the optimal solution
    return (1 - alpha1 - alpha2)*theta1 + alpha1*theta2 + alpha2*theta3

def loss_landscape1D(model0, model1, X_train, y_train, train_ODD, target_ODD, train_norm, ODD_norm, loss_fun = nn.L1Loss()):

    '''
    Calculates losses of the model which weights are interpolation of two models: model0 and model1
    '''
    

    ### Parameters [optimal] corresponding to three models
    model0.eval()
    model1.eval()
    theta_opt = Params2Vec(model0.parameters())
    theta = Params2Vec(model1.parameters())
    
    losses = []
    losses_ODD = []
    
    alphas = torch.arange(-0.5, 1.3, 0.01)
    
    for alpha in alphas:
        with torch.no_grad():
          Vec2Params(tau(alpha, theta, theta_opt),  model1.parameters())

          prediction = model1(X_train.to(device))
          loss = loss_fun(prediction, y_train.to(device)).item()
          losses.append(loss/train_norm)
          
          prediction = model1(train_ODD.to(device))
          loss = loss_fun(prediction, target_ODD.to(device)).item()
          losses_ODD.append(loss/ODD_norm)
    return alphas, losses, losses_ODD

def plot_landscape1D(res):
    
    '''
    plot loss of the interpolation of two models
    '''
    
    alphas, losses, losses_ODD = res
    #plt.title(f'Hidden size = {hidden_size}')
    plt.plot(alphas, losses)
    plt.plot(alphas, losses_ODD)
    plt.ylim([0, 1])
    plt.show()

def loss_landscape2D(model1, model2, model3, model_template, X_train, y_train, train_ODD, target_ODD, train_norm, ODD_norm, loss_fun = nn.L1Loss()):

    '''
    Calculates losses of the model which weights are interpolation of three models: model1,2,3
    '''
        

    ### Parameters [optimal] corresponding to three models
    model1.eval()
    model2.eval()
    model3.eval()
    theta1 = Params2Vec(model1.parameters())
    theta2 = Params2Vec(model2.parameters())
    theta3 = Params2Vec(model3.parameters())
    
    losses = []
    losses_ODD = []
    
    alphas1 = torch.arange(-0.8, 1.5, 0.05)
    alphas2 = torch.arange(-0.3, 1.1, 0.05)
    
    for alpha1 in alphas1:

        losses_row = []
        losses_row_ODD = []
        for alpha2 in alphas2:
            
            with torch.no_grad():
                
                Vec2Params(tau2(alpha1, alpha2, theta1, theta2, theta3),  model_template.parameters())
                  
                prediction = model_template(X_train.to(device))
                loss = loss_fun(prediction, y_train.to(device)).item()
                losses_row.append(loss/train_norm)
                
                prediction = model_template(train_ODD.to(device))
                loss = loss_fun(prediction, target_ODD.to(device)).item()
                losses_row_ODD.append(loss/ODD_norm)
        
        losses.append(losses_row)
        losses_ODD.append(losses_row_ODD)
        
                
    return list(alphas2), list(alphas1), losses, losses_ODD

def plot_landscape2D(res, n_cont=100, ODD=False):
    
    '''
    plot loss of the interpolation of two models
    '''    
    
    alphas1, alphas2, losses, losses_ODD = res
    
    if ODD:
        loss = losses_ODD
    else:
        loss = losses
    
    plt.contour(alphas1, alphas2, loss, n_cont)
    plt.clim(0,4)
    plt.show()
#%%
