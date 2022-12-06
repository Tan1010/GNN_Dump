import dgl
import numpy as np
import torch


g = dgl.graph(([0,0], [1,2]), num_nodes=3)

g.ndata['h'] = torch.randn(3,3)
print(f"Node features: {g.ndata['h']}")


def message_computation(g):
    sg1 = g.subgraph([1,2])
    return torch.mean(sg1.ndata['h'], dim=0)

def message_aggregation(g, m):
    return torch.add(g.ndata['h'][0], m)

m = message_computation(g)
print(m)
h = message_aggregation(g,m)
print(h)






