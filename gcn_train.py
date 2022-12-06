
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph


from dgl.data import CoraGraphDataset

dataset = CoraGraphDataset()
g = dataset[0]
features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
test_mask = g.ndata['test_mask']

from dgl.nn import GraphConv
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GraphConv(1433, 8*16) #activation default=None
        self.layer2 = GraphConv(8*16, 7)    #activation default=None


    def forward(self, g, features):
        x1 = F.relu(self.layer1(g, features)) #ReLU activation function
        x2 = self.layer2(g, x1)
        return x2
net = Net()

import numpy as np
g.add_edges(g.nodes(), g.nodes())
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
loss_list=[]
acc_list=[]
all_logits=[]

for epoch in range(5):
    net.train()
    logits = net(g, features)
    
    #print(logits)
    logp = F.log_softmax(logits, 1)
    all_logits.append(logp)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = evaluate(net, g, features, labels, test_mask)
    loss_list.append(loss.item())
    acc_list.append(acc)