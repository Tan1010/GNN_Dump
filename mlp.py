# mlp for binary classification
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import seaborn as sns

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    # # balanced 
    # number_of_x = 500
    # pos_x = np.random.uniform(low=0.5, high=1, size=(number_of_x,))
    # neg_x = np.random.uniform(low=0, high=0.5, size=(number_of_x,))

    # test_number_of_x = 100
    # test_pos_x = np.random.uniform(low=0.5, high=1, size=(test_number_of_x,))
    # test_neg_x = np.random.uniform(low=0, high=0.5, size=(test_number_of_x,))

    # x_train = np.concatenate((pos_x, neg_x))
    # x_train = np.reshape(x_train, (len(x_train), 1))
    # np.random.shuffle(x_train)
    # y_train = np.zeros(len(x_train))
    # for i in range(len(x_train)):
    #     if x_train[i] >= 0.5:
    #         y_train[i] = 1
    # y_train = np.reshape(y_train, (len(y_train), 1))

    # x_test = np.concatenate((test_pos_x, test_neg_x))
    # x_test = np.reshape(x_test, (len(x_test), 1))
    # np.random.shuffle(x_test)
    # y_test = np.zeros(len(x_test))
    # for i in range(len(x_test)):
    #     if x_test[i] >= 0.5:
    #         y_test[i] = 1
    # y_test = np.reshape(y_test, (len(y_test), 1))


    # imbalanced
    pos_x = np.random.uniform(low=0.5, high=1, size=(100,))
    neg_x = np.random.uniform(low=0, high=0.5, size=(1000,))

    test_pos_x = np.random.uniform(low=0.5, high=1, size=(80,))
    test_neg_x = np.random.uniform(low=0, high=0.5, size=(220,))

    x_train = np.concatenate((pos_x, neg_x))
    x_train = np.reshape(x_train, (len(x_train), 1))
    np.random.shuffle(x_train)
    y_train = np.zeros(len(x_train))
    for i in range(len(x_train)):
        if x_train[i] >= 0.5:
            y_train[i] = 1
    y_train = np.reshape(y_train, (len(y_train), 1))

    x_test = np.concatenate((test_pos_x, test_neg_x))
    x_test = np.reshape(x_test, (len(x_test), 1))
    np.random.shuffle(x_test)
    y_test = np.zeros(len(x_test))
    for i in range(len(x_test)):
        if x_test[i] >= 0.5:
            y_test[i] = 1
    y_test = np.reshape(y_test, (len(y_test), 1))



    # plot frequency vs x graph
    # figure, axis = plt.subplots(1,1)
    # # sns.histplot(data=np.array(x_train), kde=True)
    # plt.scatter(x_train,y_train)
    # plt.xlabel('x train')
    # plt.ylabel('y train')
    # plt.title('y train vs x train')
    # plt.draw()

    # figure, axis = plt.subplots(1,1)
    # # sns.histplot(data=np.array(y_train), kde=True)
    # plt.scatter(x_test,y_test)
    # plt.xlabel('x test')
    # plt.ylabel('y test')
    # plt.title('y test vs x test')
    # plt.draw()

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    figure, axis = plt.subplots(1,1)
    sns.histplot(data=np.array(x_train), kde=True)
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.title('x Train Histogram')
    plt.draw()
    
    figure, axis = plt.subplots(1,1)
    sns.histplot(data=np.array(y_train), kde=True)
    plt.xlabel('y')
    plt.ylabel('Frequency')
    plt.title('y Train Histogram')
    plt.draw()

    figure, axis = plt.subplots(1,1)
    sns.histplot(data=np.array(x_test), kde=True)
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.title('x Test Histogram')
    plt.draw()

    figure, axis = plt.subplots(1,1)
    sns.histplot(data=np.array(y_test), kde=True)
    plt.xlabel('y')
    plt.ylabel('Frequency')
    plt.title('y Test Histogram')
    plt.draw()


    mlp = MLP()
    
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    
    loss_arr = []
    precision_arr = []
    recall_arr = []
    accuracy_arr = []
    epoch = 4000
    for e in range(epoch):
        pred_y = mlp(x_train)
        loss = F.binary_cross_entropy(pred_y, y_train)
        loss_arr.append(loss)

        prediction = np.zeros(len(x_train))

        for i in range(len(pred_y)):
            if pred_y[i] >= 0.5:
                prediction[i] = 1

        prediction = np.reshape(prediction, (len(prediction), 1))
        prediction = torch.from_numpy(prediction).float()

        # accuracy
        train_accuracy = (prediction == y_train).float().mean()
        accuracy_arr.append(train_accuracy)

        # precision & recall
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(prediction)):
            if prediction[i][0] == 1.0 and y_train[i][0] == 1.0:
                tp += 1 
            elif prediction[i][0] == 1.0 and y_train[i][0] == 0.0:
                fp += 1
            elif prediction[i][0] == 0.0 and y_train[i][0] == 0.0:
                tn +=1
            elif prediction[i][0] == 0.0 and y_train[i][0] == 1.0:
                fn +=1
        try:
            train_precision = tp/(tp+fp)
            train_recall = tp/(tp+fn)
        except:
            train_precision = 0
            train_recall = 0
        
        precision_arr.append(train_precision)
        recall_arr.append(train_recall)

        print(f'Epoch {e+1}')
        print(f'Loss {loss}')
        print(f'train_accuracy {train_accuracy}')
        print(f'train_precision {train_precision}')
        print(f'train_recall {train_recall}')
        print()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if e == epoch-1:
        #     print(f'Prediction')
        #     print(f'{prediction.flatten()}')
        #     print(f'Label')
        #     print(f'{y_train.flatten()}')


    # loss_arr= [loss.detach().numpy() for loss in loss_arr]
    # plt.plot(np.arange(0,epoch),loss_arr)
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.show()

    test_pred_y = mlp(x_test)
    test_prediction = np.zeros(len(x_test))

    for i in range(len(test_pred_y)):
        if test_pred_y[i] >= 0.5:
            test_prediction[i] = 1

    test_prediction = np.reshape(test_prediction, (len(test_prediction), 1))
    test_prediction = torch.from_numpy(test_prediction).float()

    # accuracy
    test_accuracy = (test_prediction == y_test).float().mean()
    print(f'Test accuracy: {test_accuracy}')

    # precision & recall
    test_tp = 0
    test_fp = 0
    test_tn = 0
    test_fn = 0
    for i in range(len(test_prediction)):
        if test_prediction[i][0] == 1.0 and y_test[i][0] == 1.0:
            test_tp += 1 
        elif test_prediction[i][0] == 1.0 and y_test[i][0] == 0.0:
            test_fp += 1
        elif test_prediction[i][0] == 0.0 and y_test[i][0] == 0.0:
            test_tn +=1
        elif test_prediction[i][0] == 0.0 and y_test[i][0] == 1.0:
            test_fn +=1
    try:
        test_precision = test_tp/(test_tp+test_fp)
        test_recall = test_tp/(test_tp+test_fn)
    except:
        test_precision = 0
        test_recall = 0
    
    print(f'Test precision: {test_precision}')
    print(f'Test recall: {test_recall}')

    # plt.show()

