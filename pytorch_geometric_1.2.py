# test pytorch geometric message passing
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        # self.conv2 = GCNConv(h_feats, h_feats)

    def forward(self, g):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                
        h = self.conv1(g.x, g.edge_index)
        # h = self.conv2(h, g.edge_index)
        return h

# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[1], [2], [3]], dtype=torch.float)

edge_index = torch.tensor([[0, 1],
                           [1, 0]], dtype=torch.long)
# edge_index = add_self_loops(edge_index = edge_index)[0]

x = torch.tensor([[1, 2], [2, 3]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
model= GCN(2, 1)
print(model(data))


# summary
# 1. no matter we add self loops (add_self_loops) or not, it will give the same result 
# message passing will add self loop (reflects in 1 + node degree in the function)
# 2. if the node has multiple features, the final embedding will be the 
# summation of 2 features -> will have 2 different weights in 1 conv layer
# 3. layer by layer message passing - conv 1 first followed by conv 2
# each layer has different weights