import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import random

dgl.seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat) 
        h = F.relu(h) 

        h_clone = torch.clone(h)
        
        for i in range(h_clone.shape[0]):
            for j in range(h_clone.shape[1]):
                if h_clone[i,j] >= 0.5:
                    h_clone[i,j] = 1.0
                    
                else:
                    h_clone[i,j] = 0.0
                    
        return h,h_clone

def train(g, model):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    acc = []
    highest_acc = 0
    highest_acc_epoch = -1

    features = g.ndata['h']
    labels = g.ndata['labels']
    epoch = 1000
    

    for e in range(epoch):
        # print(f"Parameters: {model.parameters()}")
        if e == 1:
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

        if train_acc > highest_acc:
            highest_acc = train_acc
            highest_acc_epoch = e
        # print(f"Accuracy: {train_acc}\n")

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"First accuracy: {acc[0]}")
    print(f"Last accuracy: {acc[-1]}")
    print(f"\nHighest accuracy: {highest_acc} at epoch {highest_acc_epoch}")
    
    # plt.plot(np.arange(0,epoch),acc)
    # plt.show()

def test(g, model):
    features = g.ndata['h']
    labels = g.ndata['labels']
    final_emb, final_emb_clone = model(g, features)
    loss = F.cross_entropy(final_emb, labels)
    test_acc = (final_emb_clone == labels).float().mean()
    print(f"Test accuracy: {test_acc}")

######################### train #####################################################
# darren - 29/6
# 0001110

# darren - 1/7
# 0000000

# tan - 18/7 
# 1000101

# tan - 20/7 
# 1000100

# dr saw - 29/8
# 0000000

# dr saw - 31/8
# 0010000

train_graph= dgl.graph(([1,3,5,0,0,2,3], [0,2,4,2,3,0,0]), num_nodes=6)

# step | distance | run distance | calories | heartbeat | sleep | sp02
train_graph.ndata['h'] = torch.tensor([[0.584,0.603,0.241,0.483,0.364,0.228,0.970],
[0.783,0.793,0.362,0.729,0.175,0.175,0.960],
[0.124,0.112,0.190,0.207,0.339,0.049,1.000],
[0.016,0.014,0.090,0.029,0.150,0.310,0.990],
[0.070,0.066,0.217,0.075,0.165,0.345,0.960],
[0.292,0.285,0.569,0.310,0.297,0.203,0.960]])

train_graph = dgl.add_self_loop(train_graph)

# nausea | dizziness | headache | cough | weakness | feeling very tired | fever
train_graph.ndata['labels'] = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

model = GCN(7, 7)
train(train_graph,model)

######################### test# #####################################################
# Hamizah - 29/7
# 0001001

# Hamizah - 1/8
# 0001000

# Tan 22/7
# 1000100

# Tan 25/7 
# 1100000

# Darren 4/7 
# 0001000

# Darren 6/7
# 0001000

test_graph = dgl.graph(([1,3,5,0,0,4,5], [0,2,4,4,5,0,0]), num_nodes=6)

# step | distance | run distance | calories | heartbeat | sleep | sp02
test_graph.ndata['h'] = torch.tensor([[0.256,0.238,0.299,0.175,0.174,0.338,0.96],
[0.076,0.068,0.268,0.076,0.197,0.416,0.920],
[0.015,0.014,0.062,0.032,0.150,0.285,0.970],
[0.007,0.006,0.030,0.016,0.137,0.453,1.000],
[0.165,0.178,0.164,0.217,0.180,0.149,0.990],
[0.481,0.496,0.177,0.484,0.161,0.292,0.990]])

test_graph = dgl.add_self_loop(test_graph)

# nausea | dizziness | headache | cough | weakness | feeling very tired | fever
test_graph.ndata['labels'] = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

test(test_graph,model)
print('=================================================')






