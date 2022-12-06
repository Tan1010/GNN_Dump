import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        # h = F.relu(h)
        # h = self.conv2(g, h)
        h = F.leaky_relu(h) 

        h_clone = torch.clone(h)
        
        print(f"H: {h}")
        

        for i in range(h_clone.shape[0]):
            for j in range(h_clone.shape[1]):
                # h_clone[i,j] = abs(math.floor(h_clone[i,j]))
                if h_clone[i,j] >= 0.5:
                    h_clone[i,j] = 1.0
                    
                else:
                    h_clone[i,j] = 0.0
                    
        return h,h_clone

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    best_val_acc = 0
    best_test_acc = 0
    acc = []
    features = g.ndata['h']
    labels = g.ndata['labels']
    epoch = 500
    
    for e in range(epoch):
        
        # Forward
        logits, logits_clone = model(g, features)
        # pred = logits.argmax(1)
        print(f"Logits: {logits}")
        print(f"Logits clone: {logits_clone}")
        

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy on training/validation/test
        train_acc = (logits_clone == labels).float().mean()
        acc.append(train_acc)
        print(f"Accuracy: {train_acc}\n")

        # Save the best validation accuracy and the corresponding test accuracy.
        # if best_val_acc < val_acc:
        #     best_val_acc = val_acc
        #     best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if e % 5 == 0:
        #     print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
        #         e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
    print(f"First accuracy: {acc[0]}")
    print(f"Last accuracy: {acc[-1]}")
    
    plt.plot(np.arange(1,epoch+1),acc)
    plt.show()

g2 = dgl.graph(([1,2,3,4], [0,0,1,2]), num_nodes=5)
g2.ndata['h'] = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0], [3.0,3.0,3.0], [4.0,4.0,4.0], [5.0,5.0,5.0]])
g2 = dgl.add_self_loop(g2)

g2.ndata['labels'] = torch.tensor([[1.0,1.0,1.0],[1.0,0.0,1.0], [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])


model = GCN(3, 3)
train(g2,model)
# final_emb = model(g2, g2.ndata['h'])
# print(f'\nFinal embeddings: {final_emb}')
# final_emb = model(g2, g2.ndata['h'])
# print(f'\nFinal embeddings: {final_emb}')
print('=================================================')





