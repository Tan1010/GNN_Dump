import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import random


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h) 

        h_clone = torch.clone(h)
        
        for i in range(h_clone.shape[0]):
            for j in range(h_clone.shape[1]):
                if h_clone[i,j] >= 0.5:
                    h_clone[i,j] = 1.0
                    
                else:
                    h_clone[i,j] = 0.0
                    
        return h,h_clone

def train(g, model):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    acc = []
    features = g.ndata['h']
    labels = g.ndata['labels']
    epoch = 500
    
    for e in range(epoch):
        # print(f"Parameters: {model.parameters()}")
        if e == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)

        # Forward
        final_emb, final_emb_clone = model(g, features)
        # print(f"Final emb: {final_emb}")
        # print(f"Final emb clone: {final_emb_clone}")
        
        # Compute loss
        loss = F.cross_entropy(final_emb, labels)

        # Compute accuracy
        train_acc = (final_emb_clone == labels).float().mean()
        acc.append(train_acc)
        # print(f"Accuracy: {train_acc}\n")

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
    print(f"First accuracy: {acc[0]}")
    print(f"Last accuracy: {acc[-1]}")
    
    plt.plot(np.arange(0,epoch),acc)
    plt.show()

# 15 June


# 17 June


# 15 July 
 

# 18 July


# 20 July

g2 = dgl.graph(([1,2,3,4], [0,0,1,2]), num_nodes=5)
g2.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0], [3.0,3.0,3.0], [4.0,4.0,4.0], [5.0,5.0,5.0]])
g2 = dgl.add_self_loop(g2)

g2.ndata['labels'] = torch.tensor([[1.0,1.0,1.0],[1.0,0.0,1.0], [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])

dgl.seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

model = GCN(3, 3)
train(g2,model)
print('=================================================')





