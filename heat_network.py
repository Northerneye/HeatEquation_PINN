import os
import imageio
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_results(model, inputs, labels, save=False, name="final", dist=1.0):
    plt.plot(inputs[:101, 0], labels[:101], alpha=0.8, label="Exact")
    for j in range(5):
        a = np.linspace(0, dist, 101)
        c = np.array([[i, j/5] for i in a])
        x = torch.from_numpy(c.astype(np.float32)).requires_grad_(True).to(DEVICE)
        preds = model.predict(x)

        plt.plot(a, preds[:,0], alpha=0.8, label="PINN, t="+str(j))
    plt.legend()
    plt.ylabel('Temperature')
    plt.xlabel('Position')
    if(save):
        plt.savefig(f"./heat_run/{name}.png")
        plt.close()
    else:
        plt.show()

class Net(nn.Module):
    def __init__(self, n_hidden=30, epochs=1000, loss=nn.MSELoss(), lr=1e-3, loss2=None, loss2_weight=1.0, embedding_dim=4, dist=1.0) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.dist = dist

        self.layers = nn.Sequential(
            #nn.Linear(2, self.n_hidden),
            nn.Linear(embedding_dim*4+2, self.n_hidden),
            nn.Softplus(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Softplus(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Softplus(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Softplus(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Softplus(),
        )
        self.out = nn.Linear(self.n_hidden, 1)

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)


    def forward(self, x):
        #h = self.layers(x)
        h = self.layers(self.positional_encoding(x, self.embedding_dim))
        out = self.out(h)

        return out

    def fit(self, X, y):
        newpath = "./heat_run" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(X)
            loss = self.loss(y, outputs)
            if self.loss2:
                loss += self.loss2_weight * self.loss2(self)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 50) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
                get_results(self, X, y, save=True, name=ep, dist=self.dist)
        
        # Save training graphs as gif
        images = []
        for i in range(self.epochs+1):
            if(f'{i}.png' in os.listdir(newpath)):
                file_path = os.path.join(newpath, f"{i}.png")
                for repeat in range(8): # Each image gets this many frames
                    images.append(imageio.imread(file_path))

        # Make it pause at the end so that the viewers can ponder
        for _ in range(10):
            images.append(imageio.imread(file_path))

        imageio.mimsave(f"{newpath}/movie.gif", images)
        return losses

    def predict(self, X):
        self.eval()
        out = self.forward(X)
        return out.detach().cpu().numpy()


class NetDiscovery(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__(
            input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight
        )

        self.r = nn.Parameter(data=torch.tensor([0.]))
