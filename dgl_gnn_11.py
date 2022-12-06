import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


dgl.seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        hidden_feats = math.floor((h_feats+in_feats)/2)
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, hidden_feats)
        self.conv3 = GraphConv(hidden_feats, hidden_feats)
        self.conv4 = GraphConv(hidden_feats, hidden_feats)
        self.conv5 = GraphConv(hidden_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat) 
        h = F.relu(h) 
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        h = F.relu(h)
        h = self.conv5(g, h)
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
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tp=0
    fp=0
    tn=0
    fn=0
    loss_arr = []
    precision = []
    recall = []
    specificity = []

    features = g.ndata['h']
    labels = g.ndata['labels']
    epoch = 1000

    for e in range(epoch):
        tp=0
        fp=0
        tn=0
        fn=0

        # if e == 0 or e==epoch-1:
        #     print(f"Epoch {e}")
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             print(name, param.data)
            

        # Forward
        final_emb, final_emb_clone = model(g, features)

        # if e == epoch-1:
        #     print(f"Final emb: {final_emb_clone}")
        #     print(f"Labels: {labels}")


        # Compute loss
        loss = F.cross_entropy(final_emb, labels)
        loss_arr.append(loss)

        # Compute precision, recall, specificity
        for row in range(final_emb_clone.shape[0]):
            for col in range(final_emb_clone.shape[1]):
                if final_emb_clone[row][col] == 1.0 and labels[row][col] == 1.0:
                    tp +=1
                elif final_emb_clone[row][col] == 1.0 and labels[row][col] == 0.0:
                    fp +=1
                elif final_emb_clone[row][col] == 0.0 and labels[row][col] == 0.0:
                    tn +=1
                elif final_emb_clone[row][col] == 0.0 and labels[row][col] == 1.0:
                    fn +=1

        try:
            precise = tp/(tp+fp)
            rec = tp/(tp+fn)
            spec = tn/(fp+tn)
        except:
            precise = 0
            rec = 0
            spec = 0

        precision.append(precise)
        recall.append(rec)
        specificity.append(spec)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print(f"\nFinal Embedding: {final_emb_clone}")
    # print(f"\nLabels: {labels}")

    print(f"Train Precision: {precision[-1]}")    
    print(f"Train Recall: {recall[-1]}")    
    print(f"Train Specificity: {specificity[-1]}")    
    
    # plt.plot(np.arange(0,epoch),precision)
    # plt.plot(np.arange(0,epoch),recall)
    # plt.plot(np.arange(0,epoch),specificity)
    # plt.legend(["Precision", "Recall", "Specificity"])
    # loss_arr = loss_arr.numpy()
    loss_arr= [loss.detach().numpy() for loss in loss_arr]
    plt.plot(np.arange(0,epoch),loss_arr)
    plt.show()

def test(g, model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    test_tp=0
    test_fp=0
    test_tn=0
    test_fn=0

    features = g.ndata['h']
    labels = g.ndata['labels']
    final_emb, final_emb_clone = model(g, features)
    
    for row in range(final_emb_clone.shape[0]):
        for col in range(final_emb_clone.shape[1]):
            if final_emb_clone[row][col] == 1.0 and labels[row][col] == 1.0:
                test_tp +=1
            elif final_emb_clone[row][col] == 1.0 and labels[row][col] == 0.0:
                test_fp +=1
            elif final_emb_clone[row][col] == 0.0 and labels[row][col] == 0.0:
                test_tn +=1
            elif final_emb_clone[row][col] == 0.0 and labels[row][col] == 1.0:
                test_fn +=1
    print(f"\nFinal Embedding: {final_emb}")
    print(f"\nFinal Embedding: {final_emb_clone}")
    print(f"\nLabels: {labels}")

    try:
        test_precise = test_tp/(test_tp+test_fp)
        test_rec = test_tp/(test_tp+test_fn)
        test_spec = test_tn/(test_fp+test_tn)
    except:
        test_precise = 0
        test_rec = 0
        test_spec = 0

    print(f"\nTest Precision: {test_precise}")    
    print(f"Test Recall: {test_rec}")    
    print(f"Test Specificity: {test_spec}")    

######################### train #####################################################
# darren - 29/6 - 97%
# 001100000 - 1/7

# tan - 18/7 - 100%
# 001010001 - 20/7 

# dr saw - 29/8 - 96%
# 000011000 - 31/8

# Hamizah - 1/8 - 92%
# 000100110 - 3/8

# elaine - 8/8 - 95%
# 000000000 - 10/8
# -----------------------------------
# Patient 005 - 11/4
# 000001001 - 14/4

# Patient 013 - 23/5
# 001111001 - 27/5

# Patient 008 - 28/5
# 000001000 - 30/5

# Patient 012 - 1/7
# 001000000 - 4/7

# Patient 002 - 2/6
# 111111111 - 4/6
# -----------------------------------
# -----------------------------------
# 001100010
# 010010001
# 000110000
# 000101110
# 000010100
# -----------------------------------


# train_graph= dgl.graph(([0,1,0,3,1,2], [1,0,3,0,2,1]), num_nodes=5)
# train_graph= dgl.graph(([0,1,0,3,1,2,0,6,0,8,0,9,1,5,1,6,1,8,1,9,2,5,2,6,2,7,2,9,3,6,3,9], [1,0,3,0,2,1,6,0,8,0,9,0,5,1,6,1,8,1,9,1,5,2,6,2,7,2,9,2,6,3,9,3]), num_nodes=10)
train_graph= dgl.graph(([0,1,0,3,1,2,0,5,0,7,0,8,1,5,1,6,1,7,1,9,2,6,2,7,2,8,2,9,3,5,3,7,3,8,3,9], [1,0,3,0,2,1,5,0,7,0,8,0,5,1,6,1,7,1,9,1,6,2,7,2,8,2,9,2,5,3,7,3,8,3,9,3]), num_nodes=10)

# step | distance | run distance | calories | heartbeat | sleep | sp02
# train_graph.ndata['h'] = torch.tensor([
# [0.087,0.093,0.073,0.121,0.202,0.228,0.970],
# [0.018,0.017,0.057,0.052,0.188,0.049,1.000],
# [0.080,0.081,0.198,0.113,0.179,0.345,0.960],
# [0.123,0.113,0.402,0.143,0.213,0.416,0.920],
# [0.693,0.695,0.271,0.571,0.218,0.259,0.950]
# ])


# # --------------------- combine train & test -----------------------------
# train_graph.ndata['h'] = torch.tensor([
# [0.087,0.093,0.073,0.121,0.202,0.228,0.970],
# [0.018,0.017,0.057,0.052,0.188,0.049,1.000],
# [0.080,0.081,0.198,0.113,0.179,0.345,0.960],
# [0.123,0.113,0.402,0.143,0.213,0.416,0.920],
# [0.693,0.695,0.271,0.571,0.218,0.259,0.950],
# [0.400,0.397,0.288,0.302,0.208,0.147,0.950],
# [0.181,0.193,0.111,0.224,0.203,0.221,0.950],
# [0.113,0.094,0.266,0.133,0.201,0.197,0.970],
# [0.109,0.120,0.089,0.137,0.174,0.278,0.960],
# [0.198,0.196,0.245,0.205,0.213,0.211,0.950],
# ])
# # --------------------- combine train & test -----------------------------


# ------------------------ synthetic data --------------------------------
train_graph.ndata['h'] = torch.tensor([
[0.087,0.093,0.073,0.121,0.202,0.228,0.970],
[0.018,0.017,0.057,0.052,0.188,0.049,1.000],
[0.080,0.081,0.198,0.113,0.179,0.345,0.960],
[0.123,0.113,0.402,0.143,0.213,0.416,0.920],
[0.693,0.695,0.271,0.571,0.218,0.259,0.950],
[0.097,0.103,0.179,0.021,0.180,0.328,0.980],
[0.038,0.087,0.107,0.082,0.218,0.149,0.980],
[0.108,0.101,0.175,0.083,0.149,0.245,0.970],
[0.093,0.103,0.380,0.165,0.198,0.316,0.950],
[0.803,0.705,0.231,0.461,0.188,0.179,0.960],
])
# ------------------------ synthetic data --------------------------------


train_graph = dgl.add_self_loop(train_graph)

# nausea | dizziness | headache | cough | weakness | feeling very tired | fever
# train_graph.ndata['labels'] = torch.tensor([
# [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
# [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
# [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
# [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# ])


# # --------------------- combine train & test -----------------------------
# train_graph.ndata['labels'] = torch.tensor([
# [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
# [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
# [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
# [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
# [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
# [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
# [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
# ])
# # --------------------- combine train & test -----------------------------


# ------------------------ synthetic data --------------------------------
train_graph.ndata['labels'] = torch.tensor([
[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
])
# ------------------------ synthetic data --------------------------------


model = GCN(7, 9)
train(train_graph,model)

######################### test# #####################################################
# Patient 005 - 11/4
# 000001001 - 14/4

# Patient 013 - 23/5
# 001111001 - 27/5

# Patient 008 - 28/5
# 000001000 - 30/5

# Patient 012 - 1/7
# 001000000 - 4/7

# Patient 002 - 2/6
# 111111111 - 4/6


# test_graph = dgl.graph(([0,0,1,2,1,1,2,3], [1,2,0,0,2,3,1,1]), num_nodes=4)
test_graph = dgl.graph(([0,0,0,1,4,2,1,1,1,4,2,3,4,4,2,3], [1,4,2,0,0,0,4,2,3,1,1,1,2,3,4,4]), num_nodes=5)

# step | distance | run distance | calories | heartbeat | sleep | sp02
test_graph.ndata['h'] = torch.tensor([[0.400,0.397,0.288,0.302,0.208,0.147,0.950],
[0.181,0.193,0.111,0.224,0.203,0.221,0.950],
[0.113,0.094,0.266,0.133,0.201,0.197,0.970],
[0.109,0.120,0.089,0.137,0.174,0.278,0.960],
[0.198,0.196,0.245,0.205,0.213,0.211,0.950],
])

test_graph = dgl.add_self_loop(test_graph)

# nausea | dizziness | headache | cough | weakness | feeling very tired | fever
test_graph.ndata['labels'] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
])

# test(train_graph,model)
# test(test_graph,model)
print('=================================================')






