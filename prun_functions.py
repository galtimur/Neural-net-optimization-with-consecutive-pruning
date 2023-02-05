# -*- coding: utf-8 -*-
"""
@author: Timur Galimzyanov

Functions for pruning and repruning the model

"""

#%%
import torch
import torch.nn.utils.prune as prune

def remask(old, alpha):
    
    '''
    make random mask on the basis of old mask
    1. Gemerate tensor containing
    2. Replace 1 to 0 randomly, but firstly we do it on the positions, originally occupied by 0
    3. After saturating these positions, we start changing ones to zeros.
    '''
    
    new = torch.ones_like(old) 
    '''
    torch.rand returns numbers from uniform distribution
    goal - get random zeros from the {0,1} tensor
    idea - generate tensor with the same shape with (0,0.1] uniform numbers
    topk will take random element from such a tensor
    (prob - old) moves zero elements to topk (1 goes to the bottom of the list),
    so firstly we zero original zeros and then 1
    That is needed for encreasing the ratoi of 1, keeping 0 in their place                                                  
    '''
    
    prob = 0.1*torch.rand(old.size(), dtype=old.dtype, layout=old.layout, device=old.device)
    prob = prob - old
    nparams_to_prune = round(alpha*old.nelement())
    topk = torch.topk(prob.view(-1), k=nparams_to_prune)
    new.view(-1)[topk.indices] = 0
    
    return new

def prune_model(model, alpha):
    
    '''
    basic prune all elements of model
    '''

    for i, (name, module) in enumerate(model.named_modules()):
        if i == 0:
            continue
        prune.random_unstructured(module, name='weight', amount=alpha)
        prune.random_unstructured(module, name='bias', amount=alpha)
    
    return model

#reprune(model.lin3, 'bias', 0.6)

def reprune(module, name, alpha):

    '''
    reprune module of the model according to the old mask:
    we make new mask in which all zeros are in the place of some old zeros, but not 1
    if new pruning ratio is higher than old, we zero 1 after we zero all old 0
    '''

    old_mask = getattr(module, name + '_mask')
    new_mask = remask(old_mask, alpha)
    
    #delta = new_mask - old_mask  
    #weigts_old = getattr(module, name).detach()
       
    prune.remove(module, name)
    prune.custom_from_mask(module, name, mask=new_mask)
    # setattr(module, name, weigts_old + 0.1*torch.randn_like(delta)*delta)
    
    return module

def reprune_model_local(model, alpha):

    '''
    reprune all elements of the model according to the old mask
    '''
    
    for i, (name, module) in enumerate(model.named_modules()):
        if i == 0:
            continue
        reprune(module, 'weight', alpha)
        reprune(module, 'bias', alpha)
    
    return model

def unprune_model(model):

    '''
    unprune all elements of the model
    '''
    
    for i, (name, module) in enumerate(model.named_modules()):
        if i == 0:
            continue
        prune.remove(module, 'weight')
        prune.remove(module, 'bias')
    
    return model

def count_non_zero_tens(t):
    '''
    count ratio of nonzero elements in a tensor
    '''
    return (torch.count_nonzero(t)/t.nelement()).item()

def count_non_zero(model):
    
    '''
    count the ratio of nonzero elements in a model
    '''
    
    nonzero = 0
    n_el = 0
    
    for i, (name, module) in enumerate(model.named_modules()):
        if i == 0:
            continue
        
        nonzero += torch.count_nonzero(module.weight)
        nonzero += torch.count_nonzero(module.bias)
        
        n_el += module.weight.nelement()
        n_el += module.bias.nelement()
        
    return (nonzero/n_el).item()

#mask_names = list((dict(model.named_buffers()).keys()))  # to verify that all masks exist
