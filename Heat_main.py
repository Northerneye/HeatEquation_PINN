import matplotlib.pyplot as plt
import numpy as np
import torch
import random

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from heat_network import Net, NetDiscovery, get_results

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#torch.manual_seed(42)
#np.random.seed(10)

dist = 1

starting_positions = np.array([[i/100*dist, 0.0] for i in range(101)])
left_boundary = np.array([[0.0, i/100*dist] for i in range(101)])
right_boundary = np.array([[1.0*dist, i/100*dist] for i in range(101)])
inputs = np.concatenate((starting_positions, left_boundary, right_boundary))
inputs = torch.from_numpy(inputs.astype(np.float32))

starting_values = np.zeros((101, 1))
starting_values[:51, :] = 1.0
left_values = np.zeros((101, 1))
left_values[:, :] = 1.0
right_values = np.zeros((101, 1))
labels = np.concatenate((starting_values, left_values, right_values))
labels = torch.from_numpy(labels.astype(np.float32))

def l2_reg(model: torch.nn.Module):
    return torch.sum(sum([p.pow(2.) for p in model.parameters()]))

def physics_loss(model: torch.nn.Module):
    a = [random.random()*dist for i in range(11)]
    b = [random.random()*dist for i in range(11)]
    c = np.array([[i, j] for i in a for j in b])
    x = torch.from_numpy(c.astype(np.float32)).requires_grad_(True).to(DEVICE)
    u = model(x)
    
    # Now take derivates to form the PDEs
    u_t = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,1] # Take the partial derivative across the network with pytorch
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0]
    pde = u_t - 0.05*u_xx # = 0
    
    return torch.mean(pde**2)

def total_loss(model: torch.nn.Module):
    # Physics Loss
    pde_loss = physics_loss(model)

    # Also regularize the parameters
    #l2_loss = l2_reg(model)
    
    return pde_loss #+ 0.01*l2_loss


net = Net(loss2=total_loss, epochs=10000, loss2_weight=1, lr=3e-3, embedding_dim=8, dist=dist).to(DEVICE)

losses = net.fit(inputs, labels)
plt.plot(losses)
plt.yscale('log')
plt.show()

get_results(net, inputs, labels, dist=dist)