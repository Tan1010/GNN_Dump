# use k-fold validation 

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
no_zero_total_spo2 = []

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
        
        no_zero_total_spo2.append(data[i][email]['spo2'])


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

no_zero_total_steps = []
no_zero_total_distance = []
no_zero_total_runDistance = []
no_zero_total_calories = []
no_zero_total_heartbeat = []
no_zero_total_sleep = []


for date in date_list:
    for email in all_email_list:
        try:
            if vitalsigns_data[date][email]['steps'] != 0:
                no_zero_total_steps.append(vitalsigns_data[date][email]['steps'])
                vitalsigns_data[date][email]['steps'] = round(vitalsigns_data[date][email]['steps']/total_steps,10)
            else:
                no_zero_total_steps.append(average_steps)
                vitalsigns_data[date][email]['steps'] = round(average_steps/total_steps,10)

            if vitalsigns_data[date][email]['distance'] != 0:
                no_zero_total_distance.append(vitalsigns_data[date][email]['distance'])
                vitalsigns_data[date][email]['distance'] = round(vitalsigns_data[date][email]['distance']/total_distance,10)
            else: 
                no_zero_total_distance.append(average_distance)
                vitalsigns_data[date][email]['distance'] = round(average_distance/total_distance,10)
            
            if vitalsigns_data[date][email]['runDistance'] != 0:
                no_zero_total_runDistance.append(vitalsigns_data[date][email]['runDistance'])
                vitalsigns_data[date][email]['runDistance'] = round(vitalsigns_data[date][email]['runDistance']/total_runDistance,10)
            else: 
                no_zero_total_runDistance.append(average_runDistance)
                vitalsigns_data[date][email]['runDistance'] = round(average_runDistance/total_runDistance,10)

            if vitalsigns_data[date][email]['calories'] != 0:
                no_zero_total_calories.append(vitalsigns_data[date][email]['calories'])
                vitalsigns_data[date][email]['calories'] = round(vitalsigns_data[date][email]['calories']/total_calories,10)
            else:
                no_zero_total_calories.append(average_calorie)
                vitalsigns_data[date][email]['calories'] = round(average_calorie/total_calories,10)

            if vitalsigns_data[date][email]['heartbeat'] != 0:
                no_zero_total_heartbeat.append(vitalsigns_data[date][email]['heartbeat'])
                vitalsigns_data[date][email]['heartbeat'] = round(vitalsigns_data[date][email]['heartbeat']/total_heartbeat,10)
            else:
                no_zero_total_heartbeat.append(average_heartbeat)
                vitalsigns_data[date][email]['heartbeat'] = round(average_heartbeat/total_heartbeat,10)
            
            if vitalsigns_data[date][email]['sleep'] != 0.0:
                no_zero_total_sleep.append(vitalsigns_data[date][email]['sleep'])
                vitalsigns_data[date][email]['sleep'] = round(vitalsigns_data[date][email]['sleep']/24,10)
            else:
                no_zero_total_sleep.append(average_sleep)
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

patient_vitalsigns_symptoms = []

positive_no_zero_total_steps = []
positive_no_zero_total_distance = []
positive_no_zero_total_runDistance = []
positive_no_zero_total_calories = []
positive_no_zero_total_heartbeat = []
positive_no_zero_total_sleep = []

negative_no_zero_total_steps = []
negative_no_zero_total_distance = []
negative_no_zero_total_runDistance = []
negative_no_zero_total_calories = []
negative_no_zero_total_heartbeat = []
negative_no_zero_total_sleep = []

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

            pvs = {
                'date': date,
                'email': email,
                'symptoms' : useable_data[date][email]['symptoms'],
                'steps' : useable_data[date][email]['steps'],
                'distance' : useable_data[date][email]['distance'],
                'runDistance' : useable_data[date][email]['runDistance'],
                'calories' : useable_data[date][email]['calories'],
                'heartbeat' : useable_data[date][email]['heartbeat'],
                'sleep' : useable_data[date][email]['sleep'],
                'spo2' : useable_data[date][email]['spo2']
            }
            patient_vitalsigns_symptoms.append(pvs)


        except:
            continue



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
import seaborn as sns
import pandas as pd
import networkx as nx
from operator import add, truediv


dgl.seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# print(f'np.mean(node_steps_list): {np.mean(no_zero_total_steps)}')
# print(f'np.std(node_steps_list): {np.std(no_zero_total_steps)}')
# print(f'np.mean(node_distance_list): {np.mean(no_zero_total_distance)}')
# print(f'np.std(node_distance_list): {np.std(no_zero_total_distance)} ')
# print(f'np.mean(node_runDistance_list): {np.mean(no_zero_total_runDistance)}')
# print(f'np.std(node_runDistance_list): {np.std(no_zero_total_runDistance)} ')
# print(f'np.mean(node_calories_list): {np.mean(no_zero_total_calories)} ')
# print(f'np.std(node_calories_list): {np.std(no_zero_total_calories)} ')
# print(f'np.mean(node_heartbeat_list): {np.mean(no_zero_total_heartbeat)} ')
# print(f'np.std(node_heartbeat_list): {np.std(no_zero_total_heartbeat)} ')
# print(f'np.mean(node_sleep_list): {np.mean(no_zero_total_sleep)} ')
# print(f'np.std(node_sleep_list): {np.std(no_zero_total_sleep)} ')
# print(f'np.mean(node_spo2_list): {np.mean(no_zero_total_spo2)} ')
# print(f'np.std(node_spo2_list): {np.std(no_zero_total_spo2)} ')
figure, axis = plt.subplots(2,4)
# figure.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.7)
axis.flat[0].set(xlabel='Value', ylabel='Frequency')
axis.flat[1].set(xlabel='Value', ylabel='Frequency')
axis.flat[2].set(xlabel='Value', ylabel='Frequency')
axis.flat[3].set(xlabel='Value', ylabel='Frequency')
axis.flat[4].set(xlabel='Value', ylabel='Frequency')
axis.flat[5].set(xlabel='Value', ylabel='Frequency')
axis.flat[6].set(xlabel='Value', ylabel='Frequency')
axis[0, 0].set_title("Histogram Steps")
axis[0, 1].set_title("Histogram Distance")
axis[0, 2].set_title("Histogram Run Distance")
axis[0, 3].set_title("Histogram Calories")
axis[1, 0].set_title("Histogram Heartbeat Rate")
axis[1, 1].set_title("Histogram Sleep")
axis[1, 2].set_title("Histogram SpO2")
axis[0, 0].hist(node_steps_list) 
axis[0, 1].hist(node_distance_list) 
axis[0, 2].hist(node_runDistance_list) 
axis[0, 3].hist(node_calories_list) 
axis[1, 0].hist(node_heartbeat_list) 
axis[1, 1].hist(node_sleep_list) 
axis[1, 2].hist(node_spo2_list) 
plt.draw()



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        alpha = target * self.alpha + (1-target) * (1-self.alpha)
        pt = torch.where(target == 1.0, pred, 1-pred)
        return torch.mean(alpha * (1.0 - pt) ** self.gamma * ce)


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        hidden_feats = math.floor((h_feats+in_feats)/2)
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, hidden_feats)
        self.conv3 = GraphConv(hidden_feats, h_feats)


    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = torch.relu(h) 
        
        h = self.conv2(g, h)
        h = torch.relu(h)

        h = self.conv3(g, h)
        h = torch.sigmoid(h)

        h_clone = torch.clone(h)
        
        for i in range(h_clone.shape[0]):
            for j in range(h_clone.shape[1]):
                if h_clone[i,j] >= 0.5:
                    h_clone[i,j] = 1.0
                    
                else:
                    h_clone[i,j] = 0.0
                    
        return h,h_clone

    
all_train_loss_fl = []
all_validation_loss_fl = []
all_train_loss_ce = []
all_validation_loss_ce = []
def train(g, model, loss_function, validate_g=None, fold=0, fold_no=0, to_print=True, testing=False):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tp=0
    fp=0
    tn=0
    fn=0
    loss_arr = []
    validation_loss_arr = []

    overall_precision = []
    overall_recall = []
    precision = []
    recall = []

    validation_overall_precision = []
    validation_overall_recall = []
    validation_precision = []
    validation_recall = []

    f1 = []

    features = g.ndata['h']
    labels = g.ndata['labels']
    labels_shape = labels.shape
    epoch = 1000
    focal_loss = FocalLoss()


    for e in range(epoch):
        tp_list = [0,0,0,0,0,0,0,0,0]
        fp_list = [0,0,0,0,0,0,0,0,0]
        tn_list = [0,0,0,0,0,0,0,0,0]
        fn_list = [0,0,0,0,0,0,0,0,0]

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        # if (e == 0 or e == epoch-1) and testing==True:
        #     print(f"Epoch {e}")
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             print(name, param.data)
            # print(f'Weight 1: {model.conv1.weight.grad}')
            # print(f'Weight 2: {model.conv2.weight.grad}')
            # print(f'Weight 3: {model.conv3.weight.grad}')

        # Forward
        final_emb, final_emb_clone = model(g, features)

        if e == epoch-1:
            print(f"Final emb: {final_emb}")
            # print(f"Final emb clone: {final_emb_clone}")
            print(f"Labels: {labels}")

        # Compute loss
        # loss = F.cross_entropy(final_emb, labels)
        # loss = 0
        # for row in range(labels.shape[0]):
        #     for col in range(labels.shape[1]):
        #         if labels[row][col] == 0.0:
        #             pt = 1 - final_emb[row][col]
        #         else:
        #             pt = final_emb[row][col]
        #         # loss += -(1-pt)**2 * torch.log(pt)
        #         loss += -1 * torch.log(pt)
        # # loss = loss / (labels.shape[0] * labels.shape[1])
        if loss_function == 'focal_loss':
            loss = focal_loss(final_emb, labels)
        elif loss_function == 'ce':
            loss = F.cross_entropy(final_emb, labels)
        # print(f"Loss {e+1}: {loss}")
        loss_arr.append(loss)

        # Compute precision, recall
        for row in range(final_emb_clone.shape[0]):
            for col in range(final_emb_clone.shape[1]):
                if final_emb_clone[row][col] == 1.0 and labels[row][col] == 1.0:
                    tp_list[col] +=1
                    tp += 1 
                elif final_emb_clone[row][col] == 1.0 and labels[row][col] == 0.0:
                    fp_list[col] +=1
                    fp += 1
                elif final_emb_clone[row][col] == 0.0 and labels[row][col] == 0.0:
                    tn_list[col] +=1
                    tn +=1
                elif final_emb_clone[row][col] == 0.0 and labels[row][col] == 1.0:
                    fn_list[col] +=1
                    fn +=1
        try:
            precise = tp/(tp+fp)
            rec = tp/(tp+fn)
        except:
            precise = 0
            rec = 0
        overall_precision.append(precise)
        overall_recall.append(rec)

        add_tp_fp_list = []
        add_tp_fn_list = []

        for index in range(len(tp_list)):
            add_tp_fp_list.append(tp_list[index] + fp_list[index])
            add_tp_fn_list.append(tp_list[index] + fn_list[index])

        precision_list = []
        recall_list = []

        for index in range(len(add_tp_fp_list)):
            try:
                precise_div = tp_list[index] / add_tp_fp_list[index]
                recall_div = tp_list[index] / add_tp_fn_list[index]
            except:
                precise_div = 0
                recall_div = 0
            
            precision_list.append(precise_div)
            recall_list.append(recall_div)


        precision.append(precision_list)
        recall.append(recall_list)
        # f1_score = (2 * precise * rec) / (precise + rec)
        # f1.append(f1_score)
        if validate_g != None:
            validation_loss, validation_precision_list, validation_recall_list, validation_precise, validation_rec = validate(g=validate_g, model=model, loss_function = loss_function)

            validation_loss_arr.append(validation_loss)
            validation_overall_precision.append(validation_precise)
            validation_overall_recall.append(validation_rec)
            validation_precision.append(validation_precision_list)
            validation_recall.append(validation_recall_list)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if to_print:
        print(f"Train Precision: {precision[-1]}")    
        print(f"Train Recall: {recall[-1]}")    
        print(f"Overall Train Precision: {overall_precision[-1]}")    
        print(f"Overall Train Recall: {overall_recall[-1]}")    
       
        with open('output2.txt', 'a') as f:
            f.writelines(f"Train Precision: {precision[-1]}\n")
            f.writelines(f"Train Recall: {recall[-1]}\n")    
            f.writelines(f"Overall Train Precision: {overall_precision[-1]}\n")    
            f.writelines(f"Overall Train Recall: {overall_recall[-1]}\n")  
            f.close() 
        if validate_g != None:
            print(f"\nValidation Precision: {validation_precision[-1]}")    
            print(f"Validation Recall: {validation_recall[-1]}")  
            print(f"Overall Validation Precision: {validation_overall_precision[-1]}")    
            print(f"Overall Validation Recall: {validation_overall_recall[-1]}\n")  
            with open('output2.txt', 'a') as f:
                f.writelines(f"\nValidation Precision: {validation_precision[-1]}\n")    
                f.writelines(f"Validation Recall: {validation_recall[-1]}\n")  
                f.writelines(f"Overall Validation Precision: {validation_overall_precision[-1]}\n")    
                f.writelines(f"Overall Validation Recall: {validation_overall_recall[-1]}\n\n")  
                f.close() 
        # print(f"Train F1 Score: {f1[-1]}")    
            # validate


    validation_loss_arr= [loss.detach().numpy() for loss in validation_loss_arr]
    if loss_function == 'focal_loss':
        all_validation_loss_fl.append(validation_loss_arr)
    else:
        all_validation_loss_ce.append(validation_loss_arr)

    # plt.plot(np.arange(0,epoch),precision)
    # plt.plot(np.arange(0,epoch),recall)
    # plt.legend(["Precision", "Recall"])

    loss_arr= [loss.detach().numpy() for loss in loss_arr]
    if loss_function == 'focal_loss':
        all_train_loss_fl.append(loss_arr)
    else:
        all_train_loss_ce.append(loss_arr)

    # plt.plot(np.arange(0,epoch),loss_arr)
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.show()

    if not testing:
        if fold_no == 3:
            figure, axis = plt.subplots(2,2)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.7)

            for ax in axis.flat:
                ax.set(xlabel='epoch', ylabel='loss')
            
            if loss_function == 'focal_loss':
                axis[0, 0].set_title("Fold 1 (Focal Loss)")
                axis[0, 1].set_title("Fold 2 (Focal Loss)")
                axis[1, 0].set_title("Fold 3 (Focal Loss)")
                axis[1, 1].set_title("Fold 4 (Focal Loss)")
                axis[0, 0].plot(np.arange(0,epoch),all_train_loss_fl[0], label = "Training")
                axis[0, 1].plot(np.arange(0,epoch),all_train_loss_fl[1], label = "Training")
                axis[1, 0].plot(np.arange(0,epoch),all_train_loss_fl[2], label = "Training")
                axis[1, 1].plot(np.arange(0,epoch),all_train_loss_fl[3], label = "Training")

                axis[0, 0].plot(np.arange(0,epoch),all_validation_loss_fl[0], label = "Validation")
                axis[0, 1].plot(np.arange(0,epoch),all_validation_loss_fl[1], label = "Validation")
                axis[1, 0].plot(np.arange(0,epoch),all_validation_loss_fl[2], label = "Validation")
                axis[1, 1].plot(np.arange(0,epoch),all_validation_loss_fl[3], label = "Validation")

            else:
                axis[0, 0].set_title("Fold 1 (Cross Entropy)")
                axis[0, 1].set_title("Fold 2 (Cross Entropy)")
                axis[1, 0].set_title("Fold 3 (Cross Entropy)")
                axis[1, 1].set_title("Fold 4 (Cross Entropy)")
                axis[0, 0].plot(np.arange(0,epoch),all_train_loss_ce[0], label = "Training")
                axis[0, 1].plot(np.arange(0,epoch),all_train_loss_ce[1], label = "Training")
                axis[1, 0].plot(np.arange(0,epoch),all_train_loss_ce[2], label = "Training")
                axis[1, 1].plot(np.arange(0,epoch),all_train_loss_ce[3], label = "Training")  

                axis[0, 0].plot(np.arange(0,epoch),all_validation_loss_ce[0], label = "Validation")
                axis[0, 1].plot(np.arange(0,epoch),all_validation_loss_ce[1], label = "Validation")
                axis[1, 0].plot(np.arange(0,epoch),all_validation_loss_ce[2], label = "Validation")
                axis[1, 1].plot(np.arange(0,epoch),all_validation_loss_ce[3], label = "Validation")
            axis[0, 0].legend(loc="upper right")
            axis[0, 1].legend(loc="upper right")
            axis[1, 0].legend(loc="upper right")
            axis[1, 1].legend(loc="upper right")
        plt.draw()

        # if fold_no == 3:
        #     figure, axis = plt.subplots(2,2)
        #     for ax in axis.flat:
        #         ax.set(xlabel='epoch', ylabel='loss')
        #     if loss_function == 'focal_loss':
        #         axis[0, 0].set_title("Validation Fold 1 (Focal Loss)")
        #         axis[0, 1].set_title("Validation Fold 2 (Focal Loss)")
        #         axis[1, 0].set_title("Validation Fold 3 (Focal Loss)")
        #         axis[1, 1].set_title("Validation Fold 4 (Focal Loss)")
        #         axis[0, 0].plot(np.arange(0,epoch),all_validation_loss_fl[0])
        #         axis[0, 1].plot(np.arange(0,epoch),all_validation_loss_fl[1])
        #         axis[1, 0].plot(np.arange(0,epoch),all_validation_loss_fl[2])
        #         axis[1, 1].plot(np.arange(0,epoch),all_validation_loss_fl[3])
        #     else:
        #         axis[0, 0].set_title("Validation Fold 1 (Cross Entropy)")
        #         axis[0, 1].set_title("Validation Fold 2 (Cross Entropy)")
        #         axis[1, 0].set_title("Validation Fold 3 (Cross Entropy)")
        #         axis[1, 1].set_title("Validation Fold 4 (Cross Entropy)")
        #         axis[0, 0].plot(np.arange(0,epoch),all_validation_loss_ce[0])
        #         axis[0, 1].plot(np.arange(0,epoch),all_validation_loss_ce[1])
        #         axis[1, 0].plot(np.arange(0,epoch),all_validation_loss_ce[2])
        #         axis[1, 1].plot(np.arange(0,epoch),all_validation_loss_ce[3])
        # plt.draw()

# all_validation_loss = []

def validate(g, model, loss_function,fold=0, fold_no=0):
    validation_loss_arr = []

    validation_tp_list = [0,0,0,0,0,0,0,0,0]
    validation_fp_list = [0,0,0,0,0,0,0,0,0]
    validation_tn_list = [0,0,0,0,0,0,0,0,0]
    validation_fn_list = [0,0,0,0,0,0,0,0,0]

    validation_tp = 0
    validation_fp = 0
    validation_tn = 0
    validation_fn = 0

    features = g.ndata['h']
    labels = g.ndata['labels']
    final_emb, final_emb_clone = model(g, features)
    
    for row in range(final_emb_clone.shape[0]):
        for col in range(final_emb_clone.shape[1]):
            if final_emb_clone[row][col] == 1.0 and labels[row][col] == 1.0:
                validation_tp_list[col] +=1
                validation_tp +=1
            elif final_emb_clone[row][col] == 1.0 and labels[row][col] == 0.0:
                validation_fp_list[col] +=1
                validation_fp +=1
            elif final_emb_clone[row][col] == 0.0 and labels[row][col] == 0.0:
                validation_tn_list[col] +=1
                validation_tn+=1
            elif final_emb_clone[row][col] == 0.0 and labels[row][col] == 1.0:
                validation_fn_list[col] +=1
                validation_fn+=1

    focal_loss = FocalLoss()
    if loss_function == 'focal_loss':
        loss = focal_loss(final_emb, labels)
    elif loss_function == 'ce':
        loss = F.cross_entropy(final_emb, labels)
    # validation_loss_arr.append(loss)

    # validation_loss_arr= [loss.detach().numpy() for loss in validation_loss_arr]
    # all_validation_loss.append(validation_loss_arr)

    try:
        validation_precise =validation_tp/(validation_tp+validation_fp)
        validation_rec = validation_tp/(validation_tp+validation_fn)
    except:
        validation_precise = 0
        validation_rec = 0

    validation_add_tp_fp_list = []
    validation_add_tp_fn_list = []

    for index in range(len(validation_tp_list)):
        validation_add_tp_fp_list.append(validation_tp_list[index] + validation_fp_list[index])
        validation_add_tp_fn_list.append(validation_tp_list[index] + validation_fn_list[index])

    validation_precision_list = []
    validation_recall_list = []

    for index in range(len(validation_add_tp_fp_list)):
        try:
            validation_precise_div = validation_tp_list[index] / validation_add_tp_fp_list[index]
            validation_recall_div = validation_tp_list[index] / validation_add_tp_fn_list[index]
        except:
            validation_precise_div = 0
            validation_recall_div = 0
        
        validation_precision_list.append(validation_precise_div)
        validation_recall_list.append(validation_recall_div)

    # validation_f1_score = (2 * validation_precise * validation_rec) / (validation_precise + validation_rec)
    # print(f"\nValidation Precision: {validation_precision_list}")    
    # print(f"Validation Recall: {validation_recall_list}")  
    # print(f"Overall Validation Precision: {validation_precise}")    
    # print(f"Overall Validation Recall: {validation_rec}")  
    # with open('output2.txt', 'a') as f:
    #     f.writelines(f"\nValidation Precision: {validation_precision_list}\n")    
    #     f.writelines(f"Validation Recall: {validation_recall_list}\n")  
    #     f.writelines(f"Overall Validation Precision: {validation_precise}\n")    
    #     f.writelines(f"Overall Validation Recall: {validation_rec}\n\n")  
    #     f.close() 
    # print(f"Validation F1 Score: {validation_f1_score}")  

    # if fold_no == 3:
    #     figure, axis = plt.subplots(2,2)
    #     for ax in axis.flat:
    #         ax.set(xlabel='epoch', ylabel='loss')
    #     axis[0, 0].set_title("Validation Fold 1")
    #     axis[0, 0].plot(np.arange(0,epoch),all_validation_loss[0])

    #     axis[0, 1].set_title("Validation Fold 2")
    #     axis[0, 1].plot(np.arange(0,epoch),all_validation_loss[1])
        
    #     axis[1, 0].set_title("Validation Fold 3")
    #     axis[1, 0].plot(np.arange(0,epoch),all_validation_loss[2])

    #     axis[1, 1].set_title("Validation Fold 4")
    #     axis[1, 1].plot(np.arange(0,epoch),all_validation_loss[3])
    # plt.draw()
    return loss, validation_precision_list, validation_recall_list, validation_precise, validation_rec

def test(g, model):

    test_tp_list = [0,0,0,0,0,0,0,0,0]
    test_fp_list = [0,0,0,0,0,0,0,0,0]
    test_tn_list = [0,0,0,0,0,0,0,0,0]
    test_fn_list = [0,0,0,0,0,0,0,0,0]

    test_tp = 0
    test_fp = 0
    test_tn = 0
    test_fn = 0

    features = g.ndata['h']
    labels = g.ndata['labels']
    final_emb, final_emb_clone = model(g, features)
    
    for row in range(final_emb_clone.shape[0]):
        for col in range(final_emb_clone.shape[1]):
            if final_emb_clone[row][col] == 1.0 and labels[row][col] == 1.0:
                test_tp_list[col] +=1
                test_tp +=1
            elif final_emb_clone[row][col] == 1.0 and labels[row][col] == 0.0:
                test_fp_list[col] +=1
                test_fp +=1
            elif final_emb_clone[row][col] == 0.0 and labels[row][col] == 0.0:
                test_tn_list[col] +=1
                test_tn+=1
            elif final_emb_clone[row][col] == 0.0 and labels[row][col] == 1.0:
                test_fn_list[col] +=1
                test_fn+=1
    # print(f"\nFinal Embedding: {final_emb}")
    # print(f"\nFinal Embedding: {final_emb_clone}")
    # print(f"\nLabels: {labels}")

    try:
        test_precise =test_tp/(test_tp+test_fp)
        test_rec = test_tp/(test_tp+test_fn)
    except:
        test_precise = 0
        test_rec = 0

    test_add_tp_fp_list = []
    test_add_tp_fn_list = []

    for index in range(len(test_tp_list)):
        test_add_tp_fp_list.append(test_tp_list[index] + test_fp_list[index])
        test_add_tp_fn_list.append(test_tp_list[index] + test_fn_list[index])

    test_precision_list = []
    test_recall_list = []

    for index in range(len(test_add_tp_fp_list)):
        try:
            test_precise_div = test_tp_list[index] / test_add_tp_fp_list[index]
            test_recall_div = test_tp_list[index] / test_add_tp_fn_list[index]
        except:
            test_precise_div = 0
            test_recall_div = 0
        
        test_precision_list.append(test_precise_div)
        test_recall_list.append(test_recall_div)

    # test_f1_score = (2 * test_precise * test_rec) / (test_precise + test_rec)
    print(f"Test Precision: {test_precision_list}")    
    print(f"Test Recall: {test_recall_list}")  
    print(f"Overall Test Precision: {test_precise}")    
    print(f"Overall Test Recall: {test_rec}\n")  
    # print(f"Test F1 Score: {test_f1_score}")  
    with open('output2.txt', 'a') as f:
        f.writelines(f"Test Precision: {test_precision_list}\n")    
        f.writelines(f"Test Recall: {test_recall_list}\n")  
        f.writelines(f"Overall Test Precision: {test_precise}\n")    
        f.writelines(f"Overall Test Recall: {test_rec}\n")  
        f.close() 

######################### train #####################################################
open("output2.txt", "w").close()

features=[]
node_labels=[]

no_positive = [0,0,0,0,0,0,0,0,0]
no_negative = [0,0,0,0,0,0,0,0,0]

for i in range(total_nodes):
    current_patient_all_data = patient_vitalsigns_symptoms[i]
    current_patient_email = patient_vitalsigns_symptoms[i]['email']

    for j in range(i+1, len(patient_vitalsigns_symptoms)):
        if patient_vitalsigns_symptoms[j]['email'] == current_patient_email:
            patient_symptoms = patient_vitalsigns_symptoms[j]['symptoms']
            
            temp_node_labels = [0,0,0,0,0,0,0,0,0]
            for k in range (len(patient_symptoms)):
                if patient_symptoms[k] == '0':
                    temp_node_labels[k] = 0.0
                    no_negative[k] += 1
                elif patient_symptoms[k] == '1':
                    temp_node_labels[k] = 1.0
                    no_positive[k] += 1
            node_labels.append(temp_node_labels)

            temp_features = [
                patient_vitalsigns_symptoms[i]['steps'],
                patient_vitalsigns_symptoms[i]['distance'],
                patient_vitalsigns_symptoms[i]['runDistance'],
                patient_vitalsigns_symptoms[i]['calories'],
                patient_vitalsigns_symptoms[i]['heartbeat'],
                patient_vitalsigns_symptoms[i]['sleep'],
                patient_vitalsigns_symptoms[i]['spo2']
            ]
            features.append(temp_features)
            break

print(f'no positive: {no_positive}')      
print(f'no negative: {no_negative}')      
node_labels = np.array(node_labels)
features = np.array(features)

node_list1 = []
node_list2 = []
for i in range(len(node_labels)):
    for j in range(i+1,len(node_labels)):
        for k in range(len(node_labels[i])):
            if node_labels[i][k] == 1.0 and node_labels[j][k] == 1.0:
                node_list1.append(i)
                node_list1.append(j)
                node_list2.append(j)
                node_list2.append(i)
                break

# 4 fold
k=4
split_len = len(node_list1)//4 
ind = 0


for i in range(k):
    validation_first = ind # for validation set
    if i == k-1: # for validation set
        validation_last = len(node_list1)-1
    else:
        validation_last = ind+split_len-1 
    ind += split_len


    train_node_list1 = []
    train_node_list2 = []
    validation_node_list1 = []
    validation_node_list2 = []
    validation_node_labels = []
    train_node_labels = []
    validation_features = []
    train_features = []
    no_train_node = 0
    no_validation_node = 0

    for j in range(len(node_list1)):
        if validation_first <= j <= validation_last:
            validation_node_list1.append(node_list1[j])
            validation_node_list2.append(node_list2[j])
        else:
            train_node_list1.append(node_list1[j])
            train_node_list2.append(node_list2[j])
    print(f'### Fold {i}')
    with open('output2.txt', 'a') as f:
        f.writelines(f'### Fold {i}\n')
        f.close() 
    # print(f'validation_node_list1: {validation_node_list1[0]}')
    # print(f'validation_node_list2: {validation_node_list2[-1]}')
    # print(f'train_node_list1: {train_node_list1[0]}')
    # print(f'train_node_list2: {train_node_list2[-1]}\n')
    
    all_validation_index=[]
    all_train_index=[]

    all_nodes_index = list(range(0,len(node_labels)))
    
    if i == 0:
        validation_first_index = 0
    else:
        validation_first_index = validation_node_list1[0]
    if i == k-1: 
        validation_last_index = len(node_labels)-1
    else:
        validation_last_index = validation_node_list2[-1]

    node_index_not_in_validation_list = []
    for j in range(len(node_labels)):
        if j in validation_node_list1 or validation_first_index <= j <= validation_last_index:
            validation_node_labels.append(node_labels[j])
            validation_features.append(features[j])
            no_validation_node += 1
            all_validation_index.append(j)
        if j in train_node_list1 or j < validation_first or j > validation_last:
            train_node_labels.append(node_labels[j])
            train_features.append(features[j])
            no_train_node += 1
            all_train_index.append(j)

    validation_node_list1_set = list(set(validation_node_list1))
    validation_node_list2_set = list(set(validation_node_list2))

    validation_change_index_dict = {}
    all_validation_nodes_index = list(range(0,no_validation_node))
    all_validation_nodes_index_before_change_in_node1 = list(range(validation_first_index,validation_last_index+1))
    # for j in range(len(validation_node_list1_set)):
    #     if validation_node_list1_set[j] in all_validation_nodes_index:
    #         validation_change_index_dict[validation_node_list1_set[j]] = validation_node_list1_set[j]
    #     else:
            
    for j in range(len(all_validation_nodes_index_before_change_in_node1)):
        if all_validation_nodes_index_before_change_in_node1[j] in validation_node_list1_set:
            validation_change_index_dict[all_validation_nodes_index_before_change_in_node1[j]] = j

    validation_next_index = validation_last_index - validation_first_index + 1
    for j in range(len(validation_node_list1_set)):
        if validation_node_list1_set[j] > validation_last_index:
            validation_change_index_dict[validation_node_list1_set[j]] = validation_next_index
            validation_next_index+=1
            continue

    # print(f'no_validation_node: {no_validation_node}')
    # print(f'all_validation_nodes_index: {all_validation_nodes_index}')
    # print(f'validation_first_index: {validation_first_index}')
    # print(f'validation_last_index: {validation_last_index}')
    # print(f'all_validation_nodes_index_before_change_in_node1: {all_validation_nodes_index_before_change_in_node1}')
    # print(f'validation_node_list1_set: {validation_node_list1_set}')
    # print(f'validation_change_index_dict: {validation_change_index_dict}\n')
    for j in range(len(validation_node_list1)):
        validation_node_list1[j] = validation_change_index_dict[validation_node_list1[j]]
        validation_node_list2[j] = validation_change_index_dict[validation_node_list2[j]]

    train_node_list1_set = list(set(train_node_list1))
    train_node_list2_set = list(set(train_node_list2))

    train_change_index_dict = {}
    if i == 0:
        for j in range(no_train_node):
            if train_node_list1_set[j] >= j:
                if train_node_list1_set[j] not in train_change_index_dict.keys():
                    train_change_index_dict[train_node_list1_set[j]] = j

        for j in range(len(train_node_list1)):
            train_node_list1[j] = train_change_index_dict[train_node_list1[j]]
            train_node_list2[j] = train_change_index_dict[train_node_list2[j]]

    train_node_labels = np.array(train_node_labels)
    validation_node_labels = np.array(validation_node_labels)

    train_features = np.array(train_features)
    validation_features = np.array(validation_features)

    train_graph= dgl.graph((train_node_list1, train_node_list2), num_nodes=no_train_node)

    # step | distance | run distance | calories | heartbeat | sleep | spo2    
    train_graph.ndata['h'] = torch.from_numpy(train_features).float()

    train_graph = dgl.add_self_loop(train_graph)

    # cardiovascular | skin | gastrointestinal symptoms | eye, ears, nose, throat symptoms | musculoskeletal | neuropsychiatric symptoms | respiratory | genitourinary, reproductive | systemic
    train_graph.ndata['labels'] = torch.from_numpy(train_node_labels).float()
    model1 = GCN(7, 9)
    print(f'----------------------Focal Loss----------------------')
    with open('output2.txt', 'a') as f:
        f.writelines(f'----------------------Focal Loss----------------------\n')
        f.close() 

    

    # # validation
    validation_graph= dgl.graph((validation_node_list1, validation_node_list2), num_nodes=no_validation_node)
    validation_graph.ndata['h'] = torch.from_numpy(validation_features).float()
    validation_graph = dgl.add_self_loop(validation_graph)
    validation_graph.ndata['labels'] = torch.from_numpy(validation_node_labels).float()
    # validate(validation_graph,model1,fold=k, fold_no=i,loss_function='focal_loss')
    train(g=train_graph, validate_g =validation_graph, model = model1, loss_function = 'focal_loss', fold=k, fold_no=i)

    print(f'----------------------Cross Entropy Loss----------------------')
    with open('output2.txt', 'a') as f:
        f.writelines(f'----------------------Cross Entropy Loss----------------------\n')
        f.close() 
    model2 = GCN(7, 9)
    train(g=train_graph, validate_g =validation_graph, model=model2, loss_function = 'ce', fold=k, fold_no=i)
    # validate(validation_graph,model2, fold=k, fold_no=i, loss_function='ce')

print()
######################### test# #####################################################
print(f'Testing')
train_graph_2 = dgl.graph((node_list1, node_list2), num_nodes=len(node_labels))

# step | distance | run distance | calories | heartbeat | sleep | spo2    
train_graph_2.ndata['h'] = torch.from_numpy(features).float()

train_graph_2 = dgl.add_self_loop(train_graph_2)

# cardiovascular | skin | gastrointestinal symptoms | eye, ears, nose, throat symptoms | musculoskeletal | neuropsychiatric symptoms | respiratory | genitourinary, reproductive | systemic
train_graph_2.ndata['labels'] = torch.from_numpy(node_labels).float()
model1_2 = GCN(7, 9)
train(g = train_graph_2, model = model1_2, loss_function = 'focal_loss', to_print=False, testing=True)
model2_2 = GCN(7, 9)
train(g = train_graph_2, model = model2_2, loss_function = 'ce', to_print=False, testing=True)


# test
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
with open('output2.txt', 'a') as f:
    f.writelines(f"\n================Testing================")    
    f.close()
print(f'----------------------Focal Loss----------------------')
with open('output2.txt', 'a') as f:
    f.writelines(f'\n----------------------Focal Loss----------------------\n')
    f.close() 
test(test_graph,model1_2)

print(f'----------------------Cross Entropy Loss----------------------')
with open('output2.txt', 'a') as f:
    f.writelines(f'\n----------------------Cross Entropy Loss----------------------\n')
    f.close() 
test(test_graph,model2_2)

plt.show()