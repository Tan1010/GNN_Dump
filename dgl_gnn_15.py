import pandas as pd
import math
import numpy as np
import datetime as dt
import re

# date{email{sp02, symptoms}}

email_list = []
all_email_list = []

questionnaire_data = {}

date_list = []

spo2_list = []

# ---------------------------- symptoms --------------------------------------    
# patients_response_excel = '../../Health Data/Daily Symptoms Questionnaire (Responses).xlsx',
# patients_details_excel = '../../Health Data/Accounts.xlsx'

patients_response = pd.read_excel('../../Health Data/Daily Symptoms Questionnaire (Responses).xlsx')
patients_details = pd.read_excel('../../Health Data/Accounts.xlsx')

patients_details_emails = patients_details[['Email', 'Personal Email']]

row = patients_response.shape[0]
col = patients_response.shape[1]

timestamps = patients_response['Timestamp']

for timestamp in timestamps:
    date = str(timestamp).split(" ")[0]
    if date not in date_list:
        date_list.append(date)

date_data = {}

for i in range(row):
    people = {} 
    timestamp = patients_response.loc[i]['Timestamp']
    date = str(timestamp).split(" ")[0]
    
    if date not in date_data.keys():
        date_data[date] = []
    
    email = str(patients_response.loc[i]['Mi Account Email']).replace(" ","")

    found = True

    # check with covid email
    for k in range(len(patients_details_emails)):
        if email in str(patients_details_emails.loc[k]['Personal Email']).replace(" ",""):
            email = patients_details_emails.loc[k]['Email']  
            found = True
            break
        elif email in str(patients_details_emails.loc[k]['Email']).replace(" ",""):
            email = patients_details_emails.loc[k]['Email']
            found = True
            break
        else:
            found = False

    if not found:
        continue
    
    spo2 = patients_response.loc[i]['Oxygen level (SpO2)']
    
    if pd.isnull(spo2):
        spo2 = 0 
    
    # remove special characters
    spo2 = float(re.sub('[^0-9.]', '', str(spo2)))
    if 1 < spo2 <= 100:
        spo2 /= 100

    if spo2 != 0.0:
        spo2_list.append(spo2)

    symptoms = ''
    for j in range(3,12):
        if type(patients_response.iloc[i,j]) == str and patients_response.iloc[i,j] != 'Normal' and patients_response.iloc[i,j] != '-':
            symptoms += '1'
        else:
            symptoms += '0'

    people[email] = {
        'spo2': round(spo2,3),
        'symptoms': symptoms
    }

    date_data[date].append(people)

average_spo2 = round(np.mean(spo2_list),2)

keys = date_data.keys()

for date in date_list:
    data = date_data[date]
    people = {}
    for i in range(len(data)):
        email = list(data[i].keys())[0]
        spo2 = data[i][email]['spo2']
        if spo2 == 0.0:
            data[i][email]['spo2'] = average_spo2
        
        people[email] = data[i][email]

    questionnaire_data[date] = people

# --------------------------------- vital signs ---------------------------------
csv = [
    ['../../Health Data/New Data/20220901_patient001/ACTIVITY/ACTIVITY_1662001399818.csv', '../../Health Data/New Data/20220901_patient001/HEARTRATE_AUTO/HEARTRATE_AUTO_1662001399912.csv', '../../Health Data/New Data/20220901_patient001/SLEEP/SLEEP_1662001399868.csv'],
    ['../../Health Data/New Data/20220901_patient002/ACTIVITY/ACTIVITY_1662001453925.csv', '../../Health Data/New Data/20220901_patient002/HEARTRATE_AUTO/HEARTRATE_AUTO_1662001454073.csv', '../../Health Data/New Data/20220901_patient002/SLEEP/SLEEP_1662001454007.csv'],
    ['../../Health Data/New Data/20220901_patient003/ACTIVITY/ACTIVITY_1662001505311.csv', '../../Health Data/New Data/20220901_patient003/HEARTRATE_AUTO/HEARTRATE_AUTO_1662001506431.csv', '../../Health Data/New Data/20220901_patient003/SLEEP/SLEEP_1662001505904.csv'],
    ['../../Health Data/New Data/20220901_patient004/ACTIVITY/ACTIVITY_1662001555916.csv', '../../Health Data/New Data/20220901_patient004/HEARTRATE_AUTO/HEARTRATE_AUTO_1662001556727.csv', '../../Health Data/New Data/20220901_patient004/SLEEP/SLEEP_1662001556401.csv'],
    ['../../Health Data/New Data/20220901_patient005/ACTIVITY/ACTIVITY_1662001608635.csv', '../../Health Data/New Data/20220901_patient005/HEARTRATE_AUTO/HEARTRATE_AUTO_1662001609074.csv', '../../Health Data/New Data/20220901_patient005/SLEEP/SLEEP_1662001608814.csv'],
    ['../../Health Data/New Data/20220901_patient006/ACTIVITY/ACTIVITY_1662001661778.csv', '../../Health Data/New Data/20220901_patient006/HEARTRATE_AUTO/HEARTRATE_AUTO_1662001662357.csv', '../../Health Data/New Data/20220901_patient006/SLEEP/SLEEP_1662001662044.csv'],
    ['../../Health Data/New Data/20220901_patient007/ACTIVITY/ACTIVITY_1662001715412.csv', '../../Health Data/New Data/20220901_patient007/HEARTRATE_AUTO/HEARTRATE_AUTO_1662001716042.csv', '../../Health Data/New Data/20220901_patient007/SLEEP/SLEEP_1662001715717.csv'],
    ['../../Health Data/New Data/20220901_patient008/ACTIVITY/ACTIVITY_1662001763489.csv', '../../Health Data/New Data/20220901_patient008/HEARTRATE_AUTO/HEARTRATE_AUTO_1662001764074.csv', '../../Health Data/New Data/20220901_patient008/SLEEP/SLEEP_1662001763757.csv'],
    ['../../Health Data/New Data/20220901_patient009/ACTIVITY/ACTIVITY_1662001933723.csv', '../../Health Data/New Data/20220901_patient009/HEARTRATE_AUTO/HEARTRATE_AUTO_1662001934334.csv', '../../Health Data/New Data/20220901_patient009/SLEEP/SLEEP_1662001933983.csv'],
    ['../../Health Data/New Data/20220901_patient011/ACTIVITY/ACTIVITY_1662001986760.csv', '../../Health Data/New Data/20220901_patient011/HEARTRATE_AUTO/HEARTRATE_AUTO_1662001986811.csv', '../../Health Data/New Data/20220901_patient011/SLEEP/SLEEP_1662001986790.csv'],
    ['../../Health Data/New Data/20220901_patient012/ACTIVITY/ACTIVITY_1662002044180.csv', '../../Health Data/New Data/20220901_patient012/HEARTRATE_AUTO/HEARTRATE_AUTO_1662002044746.csv', '../../Health Data/New Data/20220901_patient012/SLEEP/SLEEP_1662002044399.csv'],
    ['../../Health Data/New Data/20220901_patient013/ACTIVITY/ACTIVITY_1662002099494.csv', '../../Health Data/New Data/20220901_patient013/HEARTRATE_AUTO/HEARTRATE_AUTO_1662002099751.csv', '../../Health Data/New Data/20220901_patient013/SLEEP/SLEEP_1662002099609.csv'],
    ['../../Health Data/New Data/20220901_patient014/ACTIVITY/ACTIVITY_1662002145145.csv', '../../Health Data/New Data/20220901_patient014/HEARTRATE_AUTO/HEARTRATE_AUTO_1662002145654.csv', '../../Health Data/New Data/20220901_patient014/SLEEP/SLEEP_1662002145339.csv'],
    ['../../Health Data/New Data/20220901_patient015/ACTIVITY/ACTIVITY_1662002193368.csv', '../../Health Data/New Data/20220901_patient015/HEARTRATE_AUTO/HEARTRATE_AUTO_1662002194285.csv', '../../Health Data/New Data/20220901_patient015/SLEEP/SLEEP_1662002193844.csv'],
    ['../../Health Data/New Data/20220901_patient016/ACTIVITY/ACTIVITY_1662002240954.csv', '../../Health Data/New Data/20220901_patient016/HEARTRATE_AUTO/HEARTRATE_AUTO_1662002241537.csv', '../../Health Data/New Data/20220901_patient016/SLEEP/SLEEP_1662002241380.csv'],
]

vitalsigns_data = {}

steps_arr = []
steps_arr_with_0 = []

distance_arr = []
distance_arr_with_0 = []

runDistance_arr = []
runDistance_arr_with_0 = []

calories_arr = []
calories_arr_with_0 = []

heartbeat_arr = []
heartbeat_arr_with_0 = []

sleep_arr = []
sleep_arr_with_0 = []

for i in range(len(csv)):
    filename = csv[i][0]
    patient_id = filename[43:46]
    email = 'iirgcovid+'+patient_id+'@hotmail.com'
    if email not in email_list:
        all_email_list.append(email)

    df0 = pd.read_csv(csv[i][0])
    df1 = pd.read_csv(csv[i][1])
    df2 = pd.read_csv(csv[i][2],on_bad_lines='skip')
 
    for date in date_list:
        try:
            steps = df0.loc[df0['date'] == date]['steps'].item()
            distance = df0.loc[df0['date'] == date]['distance'].item()
            runDistance = df0.loc[df0['date'] == date]['runDistance'].item()
            calories = df0.loc[df0['date'] == date]['calories'].item()
            heartbeat = math.ceil(df1.loc[df1['date'] == date]['heartRate'].mean())

            if email == 'iirgcovid+005@hotmail.com':
                start = str(df2.loc[date]['wakeTime'])
                start = start[:-5]
                start_date = start.split(' ')[0].split('-')
                start_time = start.split(' ')[1].split(':')
    
                stop = str(df2.loc[date]['start'])
                stop = stop[:-5]
                stop_date = stop.split(' ')[0].split('-')
                stop_time = stop.split(' ')[1].split(':')
                a = dt.datetime(int(start_date[0]),int(start_date[1]),int(start_date[2]), int(start_time[0]), int(start_time[1]),int(start_time[2]))
                b = dt.datetime(int(stop_date[0]),int(stop_date[1]),int(stop_date[2]), int(stop_time[0]), int(stop_time[1]),int(stop_time[2]))
                diff_sec = (b-a).total_seconds()
                sleep = round(diff_sec/3600,3)
                
            elif email == 'iirgcovid+013@hotmail.com' or email == 'iirgcovid+014@hotmail.com':
                start = str(df2.loc[date]['date']).split()[3]+' '+str(df2.loc[date]['date']).split()[4]
                start = start[:-5]
                start_date = start.split(' ')[0].split('-')
                start_time = start.split(' ')[1].split(':')

                stop = df2.loc[date]['date'].item()
                stop = stop[:-5]
                stop_date = stop.split(' ')[0].split('-')
                stop_time = stop.split(' ')[1].split(':')
                
                a = dt.datetime(int(start_date[0]),int(start_date[1]),int(start_date[2]), int(start_time[0]), int(start_time[1]),int(start_time[2]))
                b = dt.datetime(int(stop_date[0]),int(stop_date[1]),int(stop_date[2]), int(stop_time[0]), int(stop_time[1]),int(stop_time[2]))
                diff_sec = (b-a).total_seconds()
                sleep = round(diff_sec/3600,3)

            else:
                start = df2.loc[df2['date'] ==  date]['start'].item()
                start = start[:-5]
                start_date = start.split(' ')[0].split('-')
                start_time = start.split(' ')[1].split(':')

                stop = df2.loc[df2['date'] ==  date]['stop'].item()
                stop = stop[:-5]
                stop_date = stop.split(' ')[0].split('-')
                stop_time = stop.split(' ')[1].split(':')

                a = dt.datetime(int(start_date[0]),int(start_date[1]),int(start_date[2]), int(start_time[0]), int(start_time[1]),int(start_time[2]))
                b = dt.datetime(int(stop_date[0]),int(stop_date[1]),int(stop_date[2]), int(stop_time[0]), int(stop_time[1]),int(stop_time[2]))
                diff_sec = (b-a).total_seconds()
                sleep = round(diff_sec/3600,3)


            steps_arr_with_0.append(steps)
            distance_arr_with_0.append(distance)
            runDistance_arr_with_0.append(runDistance)
            calories_arr_with_0.append(calories)
            heartbeat_arr_with_0.append(heartbeat)
            sleep_arr_with_0.append(sleep)

            if steps != 0:
                steps_arr.append(steps)
            if distance != 0:
                distance_arr.append(distance)
            if runDistance != 0:
                runDistance_arr.append(runDistance)
            if calories != 0:
                calories_arr.append(calories)
            if heartbeat != 0:
                heartbeat_arr.append(heartbeat)
            if sleep != 0:
                sleep_arr.append(sleep)            
            
            people = {} 
            people = {
                'steps': steps,
                'distance': distance,
                'runDistance': runDistance,
                'calories': calories,
                'heartbeat': heartbeat,
                'sleep': sleep,
            }
            if vitalsigns_data.get(date) == None:
                vitalsigns_data[date] = {}
                vitalsigns_data[date][email] = people
            else:
                vitalsigns_data[date][email] = people
        except:
            continue


total_steps = np.sum(steps_arr_with_0)
average_steps = np.mean(steps_arr)

total_distance = np.sum(distance_arr_with_0)
average_distance = np.mean(distance_arr)

total_runDistance = np.sum(runDistance_arr_with_0)
average_runDistance = np.mean(runDistance_arr)

total_calories = np.sum(calories_arr_with_0)
average_calories = np.mean(calories_arr)

total_heartbeat = np.sum(heartbeat_arr_with_0)
average_heartbeat = np.mean(heartbeat_arr)

total_sleep = np.sum(sleep_arr_with_0)
average_sleep = np.mean(sleep_arr)

for date in date_list:
    for email in all_email_list:
        try:
            if vitalsigns_data[date][email]['steps'] != 0:
                vitalsigns_data[date][email]['steps'] = round(vitalsigns_data[date][email]['steps']/total_steps,10)
            else:
                vitalsigns_data[date][email]['steps'] = round(average_steps/total_steps,10)

            if vitalsigns_data[date][email]['distance'] != 0:
                vitalsigns_data[date][email]['distance'] = round(vitalsigns_data[date][email]['distance']/total_distance,10)
            else: 
                vitalsigns_data[date][email]['distance'] = round(average_distance/total_distance,10)
            
            if vitalsigns_data[date][email]['runDistance'] != 0:
                vitalsigns_data[date][email]['runDistance'] = round(vitalsigns_data[date][email]['runDistance']/total_runDistance,10)
            else: 
                vitalsigns_data[date][email]['runDistance'] = round(average_runDistance/total_runDistance,10)

            if vitalsigns_data[date][email]['calories'] != 0:
                vitalsigns_data[date][email]['calories'] = round(vitalsigns_data[date][email]['calories']/total_calories,10)
            else:
                vitalsigns_data[date][email]['calories'] = round(average_calorie/total_calories,10)

            if vitalsigns_data[date][email]['heartbeat'] != 0:
                vitalsigns_data[date][email]['heartbeat'] = round(vitalsigns_data[date][email]['heartbeat']/total_heartbeat,10)
            else:
                vitalsigns_data[date][email]['heartbeat'] = round(average_heartbeat/total_heartbeat,10)
            
            if vitalsigns_data[date][email]['sleep'] != 0.0:
                vitalsigns_data[date][email]['sleep'] = round(vitalsigns_data[date][email]['sleep']/24,10)
            else:
                vitalsigns_data[date][email]['sleep'] = round(average_sleep/24,10)
            
        except:
            continue


useable_data = {}        
for date in date_list:
    people = {} 
    for email in all_email_list:
        try:
            q_d = questionnaire_data[date][email]
            v_d = vitalsigns_data[date][email]
            people[email] = {
                'symptoms': q_d['symptoms'],
                'steps': v_d['steps'],
                'distance': v_d['distance'],
                'runDistance': v_d['runDistance'],
                'calories': v_d['calories'],
                'heartbeat': v_d['heartbeat'],
                'sleep': v_d['sleep'],
                'spo2': q_d['spo2']
            }
        except:
            continue
    useable_data[date] = people


# # --------------------------- get nodes ---------------------------------------
total_nodes = 0
node_symptoms_list=[]
node_steps_list=[]
node_distance_list=[]
node_runDistance_list=[]
node_calories_list=[]
node_heartbeat_list=[]
node_sleep_list=[]
node_spo2_list=[]
for date in date_list:
    emails = useable_data[date].keys()
    total_nodes += (len(useable_data[date].keys()))
    for email in emails:
        email_list.append(email)
        try:
            node_symptoms_list.append(useable_data[date][email]['symptoms'])
            node_steps_list.append(useable_data[date][email]['steps'])
            node_distance_list.append(useable_data[date][email]['distance'])
            node_runDistance_list.append(useable_data[date][email]['runDistance'])
            node_calories_list.append(useable_data[date][email]['calories'])
            node_heartbeat_list.append(useable_data[date][email]['heartbeat'])
            node_sleep_list.append(useable_data[date][email]['sleep'])
            node_spo2_list.append(useable_data[date][email]['spo2'])
        except:
            continue

node_list1=[]
node_list2=[]

for i in range(len(node_symptoms_list)):
    for j in range(i+1,len(node_symptoms_list)):
        for k in range(len(node_symptoms_list[i])):
            if node_symptoms_list[i][k] == '1' and node_symptoms_list[j][k] == '1':
                node_list1.append(i)
                node_list1.append(j)
                node_list2.append(j)
                node_list2.append(i)
                break

#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
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
import networkx as nx



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
        # self.conv1.weight.data.fill_(0.00)
        # self.conv1.bias.data.fill_(0.01)
        # self.conv1.weight.data = self.conv1.weight.data.clamp(-0.5,0.5)
        # print(f'weight 1 : {self.conv1.weight}')

        self.conv2 = GraphConv(hidden_feats, hidden_feats)
        # self.conv2.weight.data.fill_(0.00)
        # self.conv2.bias.data.fill_(0.01)
        # self.conv2.weight.data = self.conv2.weight.data.clamp(-0.5,0.5)
        # print(f'weight 2 : {self.conv2.weight}')

        self.conv3 = GraphConv(hidden_feats, h_feats)
        # self.conv3.weight.data.fill_(0.00)
        # self.conv3.bias.data.fill_(0.01)
        # self.conv3.weight.data = self.conv3.weight.data.clamp(-0.5,0.5)
        # print(f'weight 3 : {self.conv3.weight}')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        # print(f'conv1: {h}') 
        h = torch.relu(h) 
        # h = F.leaky_relu(h) 
        # h = torch.tanh(h) 
        # h = torch.sigmoid(h)
        # print(f'conv1 act: {h}') 

        h = self.conv2(g, h)
        # print(f'conv2: {h}') 
        h = torch.relu(h)
        # h = F.leaky_relu(h)
        # h = torch.tanh(h) 
        # h = torch.sigmoid(h)
        # print(f'conv2 act: {h}') 

        h = self.conv3(g, h)
        # print(f'conv3: {h}') 
        h = torch.sigmoid(h)
        # print(f'conv3 act: {h}') 

        h_clone = torch.clone(h)
        
        for i in range(h_clone.shape[0]):
            for j in range(h_clone.shape[1]):
                if h_clone[i,j] >= 0.5:
                    h_clone[i,j] = 1.0
                    
                else:
                    h_clone[i,j] = 0.0
                    
        return h,h_clone

def train(g, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tp=0
    fp=0
    tn=0
    fn=0
    loss_arr = []
    precision = []
    recall = []
    f1 = []

    features = g.ndata['h']
    labels = g.ndata['labels']
    labels_shape = labels.shape
    epoch = 100

    for e in range(epoch):
        tp=0
        fp=0
        tn=0
        fn=0

        # if e == 0 or e == epoch-1:
        #     print(f"Epoch {e}")
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             print(name, param.data)
        #     # print(f'Weight 1: {model.conv1.weight.grad}')
        #     # print(f'Weight 2: {model.conv2.weight.grad}')
        #     # print(f'Weight 3: {model.conv3.weight.grad}')

        # Forward
        final_emb, final_emb_clone = model(g, features)

        # if e == epoch-1:
            # print(f"Final emb: {final_emb}")
            # print(f"Final emb: {final_emb_clone}")
            # print(f"Labels: {labels}")

        # Compute loss
        loss = F.cross_entropy(final_emb, labels)
        loss_arr.append(loss)

        # Compute precision, recall
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
        except:
            precise = 0
            rec = 0

        precision.append(precise)
        recall.append(rec)
        f1_score = (2 * precise * rec) / (precise + rec)
        f1.append(f1_score)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Train Precision: {precision[-1]}")    
    print(f"Train Recall: {recall[-1]}")    
    # print(f"Train F1 Score: {f1[-1]}")    
    
    # plt.plot(np.arange(0,epoch),precision)
    # plt.plot(np.arange(0,epoch),recall)
    # plt.legend(["Precision", "Recall"])
    
    # loss_arr= [loss.detach().numpy() for loss in loss_arr]
    # plt.plot(np.arange(0,epoch),loss_arr)
    # plt.show()

def test(g, model):
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
    # print(f"\nFinal Embedding: {final_emb}")
    # print(f"\nFinal Embedding: {final_emb_clone}")
    # print(f"\nLabels: {labels}")

    try:
        test_precise = test_tp/(test_tp+test_fp)
        test_rec = test_tp/(test_tp+test_fn)
    except:
        test_precise = 0
        test_rec = 0

    test_f1_score = (2 * test_precise * test_rec) / (test_precise + test_rec)

    print(f"\nTest Precision: {test_precise}")    
    print(f"Test Recall: {test_rec}")  
    # print(f"Test F1 Score: {test_f1_score}")  

######################### train #####################################################
train_graph= dgl.graph((node_list1, node_list2), num_nodes=total_nodes-1)

features=np.zeros([total_nodes-1, 7])
node_labels=np.zeros([total_nodes-1, 9])

for i in range(total_nodes-1):
    features[i] = [
        node_steps_list[i],
        node_distance_list[i],
        node_runDistance_list[i],
        node_calories_list[i],
        node_heartbeat_list[i],
        node_sleep_list[i],
        node_spo2_list[i],
    ]

for i in range(1,total_nodes):
    for j in range(len(node_symptoms_list[i])):
        if node_symptoms_list[i][j] == '0':
            node_labels[i-1][j] = 0.0
        elif node_symptoms_list[i][j] == '1':
            node_labels[i-1][j] = 1.0

# step | distance | run distance | calories | heartbeat | sleep | sp02    
train_graph.ndata['h'] = torch.from_numpy(features).float()

train_graph = dgl.add_self_loop(train_graph)

# cardiovascular | skin | gastrointestinal symptoms | eye, ears, nose, throat symptoms | musculoskeletal | neuropsychiatric symptoms | respiratory | genitourinary, reproductive | systemic
train_graph.ndata['labels'] = torch.from_numpy(node_labels).float()
model = GCN(7, 9)
train(train_graph,model)

######################### test# #####################################################
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
test_graph= dgl.graph(([0,1,0,3,1,2], [1,0,3,0,2,1]), num_nodes=5)

features_array = np.array([
    [1028, 697, 52, 28, 87, average_sleep, 97],
    [21, 129, 41, 12, 81, 1.167, 100],
    [947, 609, 142, 26, 77, 8.283, 96],
    [1457, 849, 288, 33, 92, 9.983, 92],
    [7402, 4757, 266, 127, 94, average_sleep, 95],
])

for i in range(features_array.shape[0]):
    features_array[i,0] = round(features_array[i,0]/total_steps,10)
    features_array[i,1] = round(features_array[i,1]/total_distance,10)
    features_array[i,2] = round(features_array[i,2]/total_runDistance,10)
    features_array[i,3] = round(features_array[i,3]/total_calories,10)
    features_array[i,4] = round(features_array[i,4]/total_heartbeat,10)
    features_array[i,5] = round(features_array[i,5]/24,10)
    features_array[i,6] = features_array[i,6]/100
    

test_graph.ndata['h'] = torch.from_numpy(features_array).float()

test_graph = dgl.add_self_loop(test_graph)

test_graph.ndata['labels'] = torch.tensor([
[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

# G = test_graph.to_networkx()
# options = {
#     'node_color': 'black',
#     'node_size': 20,
#     'width': 1,
# }

test(test_graph,model)
print('=================================================')






