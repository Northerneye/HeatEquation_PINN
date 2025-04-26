import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import math

#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="FALSE"

from heat_network import Net, NetDiscovery, get_results

DEVICE = "cpu"#torch.device('xpu' if torch.xpu.is_available() else 'cpu')
print(f"Using: {DEVICE}")

#torch.manual_seed(42)
#np.random.seed(10)

dist = 1

input_res = 100
starting_positions = np.array([[i/input_res*dist, 0.0] for i in range(input_res+1)])
left_boundary = np.array([[0.0, i/input_res*dist] for i in range(input_res+1)])
right_boundary = np.array([[1.0*dist, i/input_res*dist] for i in range(input_res+1)])
inputs = np.concatenate((starting_positions, left_boundary, right_boundary))
inputs = torch.from_numpy(inputs.astype(np.float32)).to(DEVICE)

starting_values = np.zeros((input_res+1, 1))
starting_values[:math.floor(input_res/2)+1, :] = 1.0
left_values = np.zeros((input_res+1, 1))
left_values[:, :] = 1.0
right_values = np.zeros((input_res+1, 1))
labels = np.concatenate((starting_values, left_values, right_values))
labels = torch.from_numpy(labels.astype(np.float32)).to(DEVICE)

def l2_reg(model: torch.nn.Module):
    return torch.sum(sum([p.pow(2.) for p in model.parameters()]))

def physics_loss(model: torch.nn.Module):
    c = np.random.random((101, 2))*dist
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


net = Net(loss2=physics_loss, epochs=10000, loss2_weight=1, lr=3e-3, embedding_dim=8, dist=dist, device=DEVICE)
net.to(DEVICE)

losses = net.fit(inputs, labels)
plt.plot(losses)
plt.yscale('log')
plt.show()

get_results(net, inputs, labels, dist=dist, device=DEVICE)