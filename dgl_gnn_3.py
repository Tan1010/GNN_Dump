import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn import GraphConv


g = dgl.graph(([0,0], [1,2]), num_nodes=3)

# g.ndata['h'] = torch.randn(3,3)
g.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0], [3.0,3.0,3.0]])
print(f"Node features: {g.ndata['h']}")

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv = GraphConv(in_feats, h_feats, allow_zero_in_degree=True,weight=False, bias=False, norm='right')

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat)
        return h

# modify 
def message_computation(g):
    sg1 = g.subgraph([1,2])
    return torch.mean(sg1.ndata['h'], dim=0)

def message_aggregation(g, m):
    return torch.add(g.ndata['h'][0], m)

m = message_computation(g)
print(m)
h = message_aggregation(g,m)
print(f'Final Embeddings for Node 0: {h}')
print()


g2 = dgl.graph(([1,2], [0,0]), num_nodes=3)
g2.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0], [3.0,3.0,3.0]])
model = GCN(3, 3)
# g2 = dgl.add_self_loop(g2)
# print(f'In Degree: {g2.in_degrees()}')
a = model(g2, g2.ndata['h'])
g3 = a+g2.ndata['h']
# print(f'a: {a}')
print(f'g3: {g3}')
print(f'Final Embeddings for Node 0: {g3[0]}')






