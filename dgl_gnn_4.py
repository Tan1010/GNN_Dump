import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv = GraphConv(in_feats, h_feats, weight=False, bias=False, norm='both',  allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat) 
        # h = self.conv(g, in_feat) + g.ndata['h']
        # h = in_feat
        # for i in range(2):
        #     h = self.conv(g, h)
        #     h += g.ndata['h']
        #     g.ndata['h'] = h
        #     print(f"H: {h}")
        
        return h

# g2 = dgl.graph(([1,2,3,4], [0,0,1,2]), num_nodes=5)
g2 = dgl.graph(([0,0,1,1,2,2], [1,2,0,2,0,1]), num_nodes=3)
# g2.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0], [3.0,3.0,3.0], [4.0,4.0,4.0], [5.0,5.0,5.0]])
g2.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0], [3.0,3.0,3.0]])

# g2.ndata['labels'] = torch.tensor([[1.0,0.0,1.0],[1.0,0.0,0.0], [1.0,1.0,0.0], [0.0,1.0,0.0], [1.0,1.0,1.0]])
g2.ndata['labels'] = torch.tensor([[1.0,0.0,1.0],[1.0,0.0,0.0], [1.0,1.0,0.0]])
# g2 = dgl.graph(([1,2,0,2], [0,1,0,2]), num_nodes=3)
# g2.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0], [3.0,3.0,3.0]])

# g2 = dgl.graph(([1,2,3], [0,1,2]), num_nodes=4)
# g2.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0], [3.0,3.0,3.0], [4.0,4.0,4.0]])

# g2 = dgl.graph(([1,2,3], [0,1,1]), num_nodes=4)
# g2.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0], [3.0,3.0,3.0], [4.0,4.0,4.0]])
# print(f"Predecessor: {g2.predecessors(1)}")
# print(f"Successor: {g2.successors(1)}")

# g2 = dgl.graph(([1], [0]), num_nodes=2)
# g2.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0]])

g2 = dgl.add_self_loop(g2)
# in_degree = dgl.DGLGraph.in_degrees(g2)
# print(f"In degre: {in_degree}")
for e in range(1):
    print(f"Initial embeddings: {g2.ndata['h']}")
    model = GCN(3, 3)
    final_emb = model(g2, g2.ndata['h'])
    print(f'\nFinal embeddings: {final_emb}')
    loss = F.cross_entropy(final_emb, g2.ndata['labels'])
    print(f"Loss: {loss}")
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    print('=================================================')







