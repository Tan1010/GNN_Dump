import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import random
a = np.array([
    [1,2,3],
    [4,5,6]
])

b = torch.from_numpy(a)
print(b)

c = torch.sigmoid(b)
print(c)

d = F.softmax(b)
print(d)