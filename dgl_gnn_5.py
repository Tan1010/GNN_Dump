import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn import GraphConv

g = dgl.graph(([1,2], [0,0]), num_nodes=3)
g.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0], [3.0,3.0,3.0]])
print(f"Initial embeddings: {g.ndata['h']}")
g = dgl.add_self_loop(g)
in_degree = dgl.DGLGraph.in_degrees(g)

def message_computation(g):
    features = g.ndata['h'].numpy()
    message_array = np.zeros(shape=(dgl.DGLGraph.number_of_nodes(g) , dgl.DGLGraph.number_of_nodes(g)))
    
    # print(f"Empty: {message_array}")
    edge0 = g.edges()[0].numpy()
    edge1 = g.edges()[1].numpy()
    # print(f"Edge0: {edge0}")
    # print(f"Edge1: {edge1}")
   
    for i in range(dgl.DGLGraph.number_of_nodes(g)):
        index = np.where(edge1 == i)[0]
        # print(f"Index: {index}")
        for j in index:
            # print(f"J: {j}")
            # print(f"Features {j}: {features[edge1[j]]}")
            message_array[i] += features[edge0[j]]
        # print(f"Message_array {i}: {message_array[i]}")
        # print(f"In degree {i}: {in_degree[i]}")
        # print(f"After division: {message_array[i]/in_degree[i]}")
        message_array[i] /= int(in_degree[i])

    # print(f"Message array: {message_array}")

        
    return message_array
    # sg1 = g.subgraph([1,2])
    # return torch.mean(sg1.ndata['h'], dim=0)

def message_aggregation(g, m):
    return torch.add(g.ndata['h'], m)

m = torch.tensor(message_computation(g))
# print(m)
h = message_aggregation(g,m)
print(f'Final Embeddings: {h}')
print()


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        # self.conv = GraphConv(in_feats, h_feats, allow_zero_in_degree=True,weight=False, bias=False, norm='right')
        self.conv = GraphConv(in_feats, h_feats, weight=False, bias=False, norm='right')

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat) + g.ndata['h']
        return h

g2 = dgl.graph(([1,2], [0,0]), num_nodes=3)
g2.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0], [3.0,3.0,3.0]])
g2 = dgl.add_self_loop(g2)
print(f"Initial embeddings: {g2.ndata['h']}")
model = GCN(3, 3)
# print(f'In Degree: {g2.in_degrees()}')
a = model(g2, g2.ndata['h'])
# g3 = a+g2.ndata['h']
print(f'Final embeddings: {a}')
# print(f'g3: {g3}')
# print(f'Final Embeddings for Node 0: {g3[0]}')






