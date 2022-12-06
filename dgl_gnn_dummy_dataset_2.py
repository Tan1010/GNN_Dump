# class 6 classification

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
normalized_spo2 = []

keys = date_data.keys()

for date in date_list:
    data = date_data[date]
    people = {}
    for i in range(len(data)):
        email = list(data[i].keys())[0]
        spo2 = data[i][email]['spo2']
        if spo2 == 0.0:
            data[i][email]['spo2'] = average_spo2

        normalized_spo2.append(data[i][email]['spo2'])
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
max_steps = np.amax(steps_arr_with_0)
min_steps = np.amin(steps_arr_with_0)
average_steps = np.mean(steps_arr)

total_distance = np.sum(distance_arr_with_0)
max_distance = np.amax(distance_arr_with_0)
min_distance = np.amin(distance_arr_with_0)
average_distance = np.mean(distance_arr)

total_runDistance = np.sum(runDistance_arr_with_0)
max_runDistance = np.amax(runDistance_arr_with_0)
min_runDistance = np.amin(runDistance_arr_with_0)
average_runDistance = np.mean(runDistance_arr)

total_calories = np.sum(calories_arr_with_0)
max_calories = np.amax(calories_arr_with_0)
min_calories = np.amin(calories_arr_with_0)
average_calories = np.mean(calories_arr)

total_heartbeat = np.sum(heartbeat_arr_with_0)
max_heartbeat = np.amax(heartbeat_arr_with_0)
min_heartbeat = np.amin(heartbeat_arr_with_0)
average_heartbeat = np.mean(heartbeat_arr)

total_sleep = np.sum(sleep_arr_with_0)
max_sleep = np.amax(sleep_arr_with_0)
min_sleep = np.amin(sleep_arr_with_0)
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
                norm_steps = round((vitalsigns_data[date][email]['steps'] - min_steps)/(max_steps - min_steps),10)
                vitalsigns_data[date][email]['steps'] = norm_steps
            else:
                no_zero_total_steps.append(average_steps)
                norm_steps = round((average_steps - min_steps)/(max_steps - min_steps),10)
                vitalsigns_data[date][email]['steps'] = norm_steps

            if vitalsigns_data[date][email]['distance'] != 0:
                no_zero_total_distance.append(vitalsigns_data[date][email]['distance'])
                norm_distance = round((vitalsigns_data[date][email]['distance'] - min_distance)/(max_distance - min_distance),10)
                vitalsigns_data[date][email]['distance'] = norm_distance
            else: 
                no_zero_total_distance.append(average_distance)
                norm_distance = round((average_distance - min_distance)/(max_distance - min_distance),10)
                vitalsigns_data[date][email]['distance'] = norm_distance
            
            if vitalsigns_data[date][email]['runDistance'] != 0:
                no_zero_total_runDistance.append(vitalsigns_data[date][email]['runDistance'])
                norm_runDistance = round((vitalsigns_data[date][email]['runDistance'] - min_runDistance)/(max_runDistance - min_runDistance),10)
                vitalsigns_data[date][email]['runDistance'] = norm_runDistance
            else: 
                no_zero_total_runDistance.append(average_runDistance)
                norm_runDistance = round((average_runDistance - min_runDistance)/(max_runDistance - min_runDistance),10)
                vitalsigns_data[date][email]['runDistance'] = norm_runDistance

            if vitalsigns_data[date][email]['calories'] != 0:
                no_zero_total_calories.append(vitalsigns_data[date][email]['calories'])
                norm_calories = round((vitalsigns_data[date][email]['calories'] - min_calories)/(max_calories - min_calories),10)
                vitalsigns_data[date][email]['calories'] = norm_calories
            else:
                no_zero_total_calories.append(average_calorie)
                norm_calories = round((average_calorie - min_calories)/(max_calories - min_calories),10)
                vitalsigns_data[date][email]['calories'] = norm_calories

            if vitalsigns_data[date][email]['heartbeat'] != 0:
                no_zero_total_heartbeat.append(vitalsigns_data[date][email]['heartbeat'])
                norm_heartbeat = round((vitalsigns_data[date][email]['heartbeat'] - min_heartbeat)/(max_heartbeat - min_heartbeat),10)
                vitalsigns_data[date][email]['heartbeat'] = norm_heartbeat
            else:
                no_zero_total_heartbeat.append(average_heartbeat)
                norm_heartbeat = round((average_heartbeat - min_heartbeat)/(max_heartbeat - min_heartbeat),10)
                vitalsigns_data[date][email]['heartbeat'] = norm_heartbeat
            
            if vitalsigns_data[date][email]['sleep'] != 0.0:
                no_zero_total_sleep.append(vitalsigns_data[date][email]['sleep'])
                norm_sleep = round((vitalsigns_data[date][email]['sleep'] - min_sleep)/(max_sleep - min_sleep),10)
                vitalsigns_data[date][email]['sleep'] = norm_sleep
            else:
                no_zero_total_sleep.append(average_sleep)
                norm_sleep = round((average_sleep - min_sleep)/(max_sleep - min_sleep),10)
                vitalsigns_data[date][email]['sleep'] = norm_sleep
            
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

normalized_steps = []
normalized_distance = []
normalized_runDistance = []
normalized_calories = []
normalized_heartbeat = []
normalized_sleep = []

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

            normalized_steps.append(useable_data[date][email]['steps'])
            normalized_distance.append(useable_data[date][email]['distance'])
            normalized_runDistance.append(useable_data[date][email]['runDistance'])
            normalized_calories.append(useable_data[date][email]['calories'])
            normalized_heartbeat.append(useable_data[date][email]['heartbeat'])
            normalized_sleep.append(useable_data[date][email]['sleep'])
            normalized_spo2.append(useable_data[date][email]['spo2'])

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

# # figure, axis = plt.subplots(2,4)
# # # figure.tight_layout()
# # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.7)
# # axis.flat[0].set(xlabel='Value', ylabel='Frequency')
# # axis.flat[1].set(xlabel='Value', ylabel='Frequency')
# # axis.flat[2].set(xlabel='Value', ylabel='Frequency')
# # axis.flat[3].set(xlabel='Value', ylabel='Frequency')
# # axis.flat[4].set(xlabel='Value', ylabel='Frequency')
# # axis.flat[5].set(xlabel='Value', ylabel='Frequency')
# # axis.flat[6].set(xlabel='Value', ylabel='Frequency')
# # axis[0, 0].set_title("Histogram Steps")
# # axis[0, 1].set_title("Histogram Distance")
# # axis[0, 2].set_title("Histogram Run Distance")
# # axis[0, 3].set_title("Histogram Calories")
# # axis[1, 0].set_title("Histogram Heartbeat Rate")
# # axis[1, 1].set_title("Histogram Sleep")
# # axis[1, 2].set_title("Histogram SpO2")
# # axis[0, 0].hist(node_steps_list) 
# # axis[0, 1].hist(node_distance_list) 
# # axis[0, 2].hist(node_runDistance_list) 
# # axis[0, 3].hist(node_calories_list) 
# # axis[1, 0].hist(node_heartbeat_list) 
# # # axis[1, 1].hist(node_sleep_list) 
# # # axis[1, 2].hist(node_spo2_list) 
# # plt.draw()

# # sns.kdeplot(np.array(normalized_steps))
# # sns.histplot(data=np.array(normalized_distance), kde=True, kde_kws=dict(cut=3),line_kws={'color': 'red'})
# sns.histplot(data=np.array(normalized_spo2), kde=False, stat='density')
# sns.kdeplot(data=np.array(normalized_spo2), color='crimson')

# plt.xlabel('SpO2')
# plt.ylabel('Density')
# plt.title('KDE SpO2')
# plt.show()


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

all_train_precision_fl = []
all_validation_precision_fl = []

all_train_precision_ce = []
all_validation_precision_ce = []

all_train_recall_fl = []
all_validation_recall_fl = []

all_train_recall_ce = []
all_validation_recall_ce = []

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
    epoch = 100
    focal_loss = FocalLoss()


    for e in range(epoch):
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        # if (e == 0 or e == epoch-1) and testing==True:
        #     print(f"Epoch {e}")
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             print(name, param.data)
        #     print(f'Weight 1: {model.conv1.weight.grad}')
        #     print(f'Weight 2: {model.conv2.weight.grad}')
        #     print(f'Weight 3: {model.conv3.weight.grad}')

        # Forward
        final_emb, final_emb_clone = model(g, features)
        
        labels = np.reshape(labels, (len(labels), 1))
        # if e == epoch-1:
        #     print(f"Final emb: {final_emb}")
        #     # print(f"Final emb clone: {final_emb_clone}")
        #     print(f"Labels: {labels}")

        # Compute loss
        if loss_function == 'focal_loss':
            loss = focal_loss(final_emb, labels)
        elif loss_function == 'ce':
            loss = F.binary_cross_entropy(final_emb, labels)
        # print(f"Loss {e+1}: {loss}")
        loss_arr.append(loss)
        
        
        # Compute precision, recall
        for row in range(final_emb_clone.shape[0]):
            if final_emb_clone[row] == 1.0 and labels[row] == 1.0:
                tp += 1 
            elif final_emb_clone[row] == 1.0 and labels[row] == 0.0:
                fp += 1
            elif final_emb_clone[row] == 0.0 and labels[row] == 0.0:
                tn +=1
            elif final_emb_clone[row] == 0.0 and labels[row] == 1.0:
                fn +=1
        try:
            precise = tp/(tp+fp)
            rec = tp/(tp+fn)
        except:
            precise = 0
            rec = 0
        overall_precision.append(precise)
        overall_recall.append(rec)

        # f1_score = (2 * precise * rec) / (precise + rec)
        # f1.append(f1_score)
        if validate_g != None:
            validation_loss, validation_precise, validation_rec = validate(g=validate_g, model=model, loss_function = loss_function)

            validation_loss_arr.append(validation_loss)
            validation_overall_precision.append(validation_precise)
            validation_overall_recall.append(validation_rec)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if to_print:
        print(f"Overall Train Precision: {overall_precision[-1]}")    
        print(f"Overall Train Recall: {overall_recall[-1]}")    
       
        with open('output3.txt', 'a') as f:  
            f.writelines(f"Overall Train Precision: {overall_precision[-1]}\n")    
            f.writelines(f"Overall Train Recall: {overall_recall[-1]}\n")  
            f.close() 
        if validate_g != None:
            print(f"\nOverall Validation Precision: {validation_overall_precision[-1]}")    
            print(f"Overall Validation Recall: {validation_overall_recall[-1]}\n")  
            with open('output3.txt', 'a') as f:
                f.writelines(f"\nOverall Validation Precision: {validation_overall_precision[-1]}\n")    
                f.writelines(f"Overall Validation Recall: {validation_overall_recall[-1]}\n\n")  
                f.close() 
        # print(f"Train F1 Score: {f1[-1]}")    
            # validate


    validation_loss_arr= [loss.detach().numpy() for loss in validation_loss_arr]

    if loss_function == 'focal_loss':
        all_validation_loss_fl.append(validation_loss_arr)
    else:
        all_validation_loss_ce.append(validation_loss_arr)


    # plt.plot(np.arange(0,epoch),overall_precision)
    # plt.plot(np.arange(0,epoch),overall_recall)
    # plt.legend(["Precision", "Recall"])
    
    loss_arr= [loss.detach().numpy() for loss in loss_arr]
    # precision_arr = [overall_precision]
    # recall_arr = [overall_recall]
    
    if loss_function == 'focal_loss':
        all_train_loss_fl.append(loss_arr)
        all_train_precision_fl.append(overall_precision)
        all_train_recall_fl.append(overall_recall)
        all_validation_precision_fl.append(validation_overall_precision)
        all_validation_recall_fl.append(validation_overall_recall)

    else:
        all_train_loss_ce.append(loss_arr)
        all_train_precision_ce.append(overall_precision)
        all_train_recall_ce.append(overall_recall)
        all_validation_precision_ce.append(validation_overall_precision)
        all_validation_recall_ce.append(validation_overall_recall)

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
                axis[0, 0].set_title("Fold 1 (Binary Cross Entropy)")
                axis[0, 1].set_title("Fold 2 (Binary Cross Entropy)")
                axis[1, 0].set_title("Fold 3 (Binary Cross Entropy)")
                axis[1, 1].set_title("Fold 4 (Binary Cross Entropy)")
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


            figure2, axis2 = plt.subplots(2,2)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.7)
            for ax in axis2.flat:
                ax.set(xlabel='epoch', ylabel='precision')

            figure3, axis3 = plt.subplots(2,2)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.7)
            for ax in axis3.flat:
                ax.set(xlabel='epoch', ylabel='recall')

            if loss_function == 'focal_loss':
                axis2[0, 0].set_title("Fold 1 (Focal Loss)")
                axis2[0, 1].set_title("Fold 2 (Focal Loss)")
                axis2[1, 0].set_title("Fold 3 (Focal Loss)")
                axis2[1, 1].set_title("Fold 4 (Focal Loss)")
                axis2[0, 0].plot(np.arange(0,epoch),all_train_precision_fl[0], label = "Training")
                axis2[0, 1].plot(np.arange(0,epoch),all_train_precision_fl[1], label = "Training")
                axis2[1, 0].plot(np.arange(0,epoch),all_train_precision_fl[2], label = "Training")
                axis2[1, 1].plot(np.arange(0,epoch),all_train_precision_fl[3], label = "Training")

                axis2[0, 0].plot(np.arange(0,epoch),all_validation_precision_fl[0], label = "Validation")
                axis2[0, 1].plot(np.arange(0,epoch),all_validation_precision_fl[1], label = "Validation")
                axis2[1, 0].plot(np.arange(0,epoch),all_validation_precision_fl[2], label = "Validation")
                axis2[1, 1].plot(np.arange(0,epoch),all_validation_precision_fl[3], label = "Validation")
                

                axis3[0, 0].set_title("Fold 1 (Focal Loss)")
                axis3[0, 1].set_title("Fold 2 (Focal Loss)")
                axis3[1, 0].set_title("Fold 3 (Focal Loss)")
                axis3[1, 1].set_title("Fold 4 (Focal Loss)")
                
                axis3[0, 0].plot(np.arange(0,epoch),all_train_recall_fl[0], label = "Training")
                axis3[0, 1].plot(np.arange(0,epoch),all_train_recall_fl[1], label = "Training")
                axis3[1, 0].plot(np.arange(0,epoch),all_train_recall_fl[2], label = "Training")
                axis3[1, 1].plot(np.arange(0,epoch),all_train_recall_fl[3], label = "Training")

                axis3[0, 0].plot(np.arange(0,epoch),all_validation_recall_fl[0], label = "Validation")
                axis3[0, 1].plot(np.arange(0,epoch),all_validation_recall_fl[1], label = "Validation")
                axis3[1, 0].plot(np.arange(0,epoch),all_validation_recall_fl[2], label = "Validation")
                axis3[1, 1].plot(np.arange(0,epoch),all_validation_recall_fl[3], label = "Validation")
            else:
                axis2[0, 0].set_title("Fold 1 (Binary Cross Entropy)")
                axis2[0, 1].set_title("Fold 2 (Binary Cross Entropy)")
                axis2[1, 0].set_title("Fold 3 (Binary Cross Entropy)")
                axis2[1, 1].set_title("Fold 4 (Binary Cross Entropy)")
                axis2[0, 0].plot(np.arange(0,epoch),all_train_precision_ce[0], label = "Training")
                axis2[0, 1].plot(np.arange(0,epoch),all_train_precision_ce[1], label = "Training")
                axis2[1, 0].plot(np.arange(0,epoch),all_train_precision_ce[2], label = "Training")
                axis2[1, 1].plot(np.arange(0,epoch),all_train_precision_ce[3], label = "Training")  

                axis2[0, 0].plot(np.arange(0,epoch),all_validation_precision_ce[0], label = "Validation")
                axis2[0, 1].plot(np.arange(0,epoch),all_validation_precision_ce[1], label = "Validation")
                axis2[1, 0].plot(np.arange(0,epoch),all_validation_precision_ce[2], label = "Validation")
                axis2[1, 1].plot(np.arange(0,epoch),all_validation_precision_ce[3], label = "Validation")

                axis3[0, 0].set_title("Fold 1 (Binary Cross Entropy)")
                axis3[0, 1].set_title("Fold 2 (Binary Cross Entropy)")
                axis3[1, 0].set_title("Fold 3 (Binary Cross Entropy)")
                axis3[1, 1].set_title("Fold 4 (Binary Cross Entropy)")
                axis3[0, 0].plot(np.arange(0,epoch),all_train_recall_ce[0], label = "Training")
                axis3[0, 1].plot(np.arange(0,epoch),all_train_recall_ce[1], label = "Training")
                axis3[1, 0].plot(np.arange(0,epoch),all_train_recall_ce[2], label = "Training")
                axis3[1, 1].plot(np.arange(0,epoch),all_train_recall_ce[3], label = "Training")  

                axis3[0, 0].plot(np.arange(0,epoch),all_validation_recall_ce[0], label = "Validation")
                axis3[0, 1].plot(np.arange(0,epoch),all_validation_recall_ce[1], label = "Validation")
                axis3[1, 0].plot(np.arange(0,epoch),all_validation_recall_ce[2], label = "Validation")
                axis3[1, 1].plot(np.arange(0,epoch),all_validation_recall_ce[3], label = "Validation")
            axis2[0, 0].legend(loc="upper right")
            axis2[0, 1].legend(loc="upper right")
            axis2[1, 0].legend(loc="upper right")
            axis2[1, 1].legend(loc="upper right")
            axis3[0, 0].legend(loc="upper right")
            axis3[0, 1].legend(loc="upper right")
            axis3[1, 0].legend(loc="upper right")
            axis3[1, 1].legend(loc="upper right")
            

        plt.draw()

def validate(g, model, loss_function,fold=0, fold_no=0):
    validation_loss_arr = []

    validation_tp = 0
    validation_fp = 0
    validation_tn = 0
    validation_fn = 0

    features = g.ndata['h']
    labels = g.ndata['labels']
    final_emb, final_emb_clone = model(g, features)
    
    labels = np.reshape(labels, (len(labels), 1))

    for row in range(final_emb_clone.shape[0]):
        if final_emb_clone[row] == 1.0 and labels[row] == 1.0:
            validation_tp +=1
        elif final_emb_clone[row] == 1.0 and labels[row] == 0.0:
            validation_fp +=1
        elif final_emb_clone[row] == 0.0 and labels[row] == 0.0:
            validation_tn+=1
        elif final_emb_clone[row] == 0.0 and labels[row] == 1.0:
            validation_fn+=1

    focal_loss = FocalLoss()
    if loss_function == 'focal_loss':
        loss = focal_loss(final_emb, labels)
    elif loss_function == 'ce':
        loss = F.binary_cross_entropy(final_emb, labels)

    try:
        validation_precise =validation_tp/(validation_tp+validation_fp)
        validation_rec = validation_tp/(validation_tp+validation_fn)
    except:
        validation_precise = 0
        validation_rec = 0

    # validation_f1_score = (2 * validation_precise * validation_rec) / (validation_precise + validation_rec)
    
    return loss, validation_precise, validation_rec

def test(g, model):
    test_tp = 0
    test_fp = 0
    test_tn = 0
    test_fn = 0

    features = g.ndata['h']
    labels = g.ndata['labels']
    # labels = labels[:, 5]
    final_emb, final_emb_clone = model(g, features)
    # print(f'final_emb_clone: {final_emb_clone}')
    # print(f'labels: {labels}')
    labels = np.reshape(labels, (len(labels), 1))


    for row in range(final_emb_clone.shape[0]):
        if final_emb_clone[row] == 1.0 and labels[row] == 1.0:
            test_tp +=1
        elif final_emb_clone[row] == 1.0 and labels[row] == 0.0:
            test_fp +=1
        elif final_emb_clone[row] == 0.0 and labels[row] == 0.0:
            test_tn+=1
        elif final_emb_clone[row] == 0.0 and labels[row] == 1.0:
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


    # test_f1_score = (2 * test_precise * test_rec) / (test_precise + test_rec) 
    print(f"Overall Test Precision: {test_precise}")    
    print(f"Overall Test Recall: {test_rec}\n")  
    # print(f"Test F1 Score: {test_f1_score}")  
    with open('output3.txt', 'a') as f:
        f.writelines(f"Overall Test Precision: {test_precise}\n")    
        f.writelines(f"Overall Test Recall: {test_rec}\n")  
        f.close() 

######################### train #####################################################
open("output4.txt", "w").close()

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



pos_x = np.random.uniform(low=0.5, high=1, size=(200,))
neg_x = np.random.uniform(low=0, high=0.5, size=(1000,))


x_train = np.concatenate((pos_x, neg_x))
x_train = np.reshape(x_train, (len(x_train), 1))
np.random.shuffle(x_train)
y_train = np.zeros(len(x_train))
for i in range(len(x_train)):
    if x_train[i] >= 0.5:
        y_train[i] = 1
y_train = np.reshape(y_train, (len(y_train), 1))

# need this for testing purpose (use all the nodes)
all_len = len(x_train)
node_list1 = []
node_list2 = []
for j in range(all_len):
    for k in range(all_len):
        if k == j:
            continue
        node_list1.append(j)
        node_list2.append(k)



# 4 fold
k=4
split_len = len(y_train)//4 

for i in range(k):
    print(f'Fold: {i}')
    if i == 0:
        dummy_features = x_train[split_len:]
        validation_dummy_features = x_train[:split_len]
        dummy_labels = y_train[split_len:]
        validation_dummy_labels = y_train[:split_len]
    elif i == 1:
        dummy_features = np.concatenate((x_train[0:split_len], x_train[split_len+split_len:]))
        validation_dummy_features = x_train[split_len:split_len+split_len]
        dummy_labels = np.concatenate((y_train[0:split_len], y_train[split_len+split_len:]))
        validation_dummy_labels = y_train[split_len:split_len+split_len]
    elif i == 2:
        dummy_features = np.concatenate((x_train[0:split_len+split_len], x_train[split_len+split_len+split_len:]))
        validation_dummy_features = x_train[split_len+split_len:split_len+split_len+split_len]
        dummy_labels = np.concatenate((y_train[0:split_len+split_len], y_train[split_len+split_len+split_len:]))
        validation_dummy_labels = y_train[split_len+split_len:split_len+split_len+split_len]
    elif i == 3:
        dummy_features = x_train[0:split_len+split_len+split_len]
        validation_dummy_features = x_train[split_len+split_len+split_len:]
        dummy_labels = y_train[0:split_len+split_len+split_len]
        validation_dummy_labels = y_train[split_len+split_len+split_len:]

    print(f'Total training data: {len(dummy_features)}')
    print(f'Total validation data: {len(validation_dummy_features)}')

    figure, axis = plt.subplots(1,1)
    sns.histplot(data=np.array(dummy_features), kde=False, stat='count')
    sns.kdeplot(data=np.array(dummy_features), color='crimson')
    plt.xlabel('SpO2')
    plt.ylabel('Density')
    plt.title(f'Histogram - Features (Fold {i+1} Training)')
    plt.draw()

    figure, axis = plt.subplots(1,1)
    sns.histplot(data=np.array(validation_dummy_features), kde=False, stat='count')
    sns.kdeplot(data=np.array(validation_dummy_features), color='crimson')
    plt.xlabel('SpO2')
    plt.ylabel('Density')
    plt.title(f'Histogram - Features (Fold {i+1} Validation)')
    plt.draw()

    train_len = len(dummy_features)
    current_fold_train_list1 = []
    current_fold_train_list2 = []
    for j in range(train_len):
        for k in range(train_len):
            if k == j:
                continue
            current_fold_train_list1.append(j)
            current_fold_train_list2.append(k)

    validation_len = len(validation_dummy_features)
    current_fold_validation_list1 = []
    current_fold_validation_list2 = []
    for j in range(validation_len):
        for k in range(validation_len):
            if k == j:
                continue
            current_fold_validation_list1.append(j)
            current_fold_validation_list2.append(k)

    train_graph= dgl.graph((current_fold_train_list1, current_fold_train_list2), num_nodes = len(dummy_features))
    train_graph.ndata['h'] = torch.from_numpy(dummy_features).float()
    train_graph = dgl.add_self_loop(train_graph)
    train_graph.ndata['labels'] = torch.from_numpy(dummy_labels).float()

    validation_graph= dgl.graph((current_fold_validation_list1, current_fold_validation_list2), num_nodes = len(validation_dummy_features))
    validation_graph.ndata['h'] = torch.from_numpy(validation_dummy_features).float()
    validation_graph = dgl.add_self_loop(validation_graph)
    validation_graph.ndata['labels'] = torch.from_numpy(validation_dummy_labels).float()

    model = GCN(1, 1)
    # train(g=train_graph, validate_g =validation_graph, model=model, loss_function = 'ce', fold=k, fold_no=i)

# ######################### test# #####################################################
print(f'Testing')

test_pos_x = np.random.uniform(low=0.5, high=1, size=(100,))
test_neg_x = np.random.uniform(low=0, high=0.5, size=(300,))

x_test = np.concatenate((test_pos_x, test_neg_x))
x_test = np.reshape(x_test, (len(x_test), 1))
np.random.shuffle(x_test)
y_test = np.zeros(len(x_test))
for i in range(len(x_test)):
    if x_test[i] >= 0.5:
        y_test[i] = 1
y_test = np.reshape(y_test, (len(y_test), 1))

test_all_len = len(x_test)
test_node_list1 = []
test_node_list2 = []
for j in range(test_all_len):
    for k in range(test_all_len):
        if k == j:
            continue
        test_node_list1.append(j)
        test_node_list2.append(k)
print(f'Total testing data: {len(x_test)}')
figure, axis = plt.subplots(1,1)
sns.histplot(data=np.array(dummy_features), kde=False, stat='count')
sns.kdeplot(data=np.array(dummy_features), color='crimson')
plt.xlabel('Features')
plt.ylabel('Frequency')
plt.title(f'Histogram - Features (Testing)')
plt.draw()


train_graph_2 = dgl.graph((node_list1, node_list2), num_nodes=len(x_train))
train_graph_2.ndata['h'] = torch.from_numpy(x_train).float()
train_graph_2 = dgl.add_self_loop(train_graph_2)
train_graph_2.ndata['labels'] = torch.from_numpy(y_train).float()
model2_2 = GCN(1, 1)
train(g = train_graph_2, model = model2_2, loss_function = 'ce', to_print=False, testing=True)


# test
test_graph= dgl.graph((test_node_list1, test_node_list2), num_nodes=len(x_test))
test_graph.ndata['h'] = torch.from_numpy(x_test).float()
test_graph = dgl.add_self_loop(test_graph)
test_graph.ndata['labels'] = torch.from_numpy(y_test).float()
with open('output3.txt', 'a') as f:
    f.writelines(f"\n================Testing================")    
    f.close()

print(f'----------------------Binary Cross Entropy Loss----------------------')
with open('output3.txt', 'a') as f:
    f.writelines(f'\n----------------------Binary Cross Entropy Loss----------------------\n')
    f.close() 
test(test_graph,model2_2)

plt.show()