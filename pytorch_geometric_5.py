# class 6 classification
# remove distance feature
# use pytorch geometric

import pandas as pd
import math
import numpy as np
import scipy.stats as stats
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
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
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
        self.conv1 = GCNConv(in_feats, in_feats)
        # self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.conv2 = GCNConv(in_feats, h_feats)


    def forward(self, g, in_feat):
        # h = self.conv1(g, in_feat)
        h = self.conv1(g.x, g.edge_index)
        h = torch.relu(h) 
        
        # h = self.conv2(g, h)
        # h = self.conv2(h, g.edge_index)
        # h = torch.relu(h)

        # h = self.conv3(g, h)
        h = self.conv2(h, g.edge_index)
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

stopped_epoch = [float("inf"), float("inf"), float("inf"), float("inf")]


def train(g, model, loss_function, validate_g=None, fold=0, fold_no=0, to_print=True, testing=False):
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
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

    best_validation_loss, best_validation_epoch = None, None
    max_stagnation = 10


    f1 = []

    # features = g.ndata['h']
    features = g.x
    # labels = g.ndata['labels']
    labels = g.y
    labels_shape = labels.shape
    epoch = 100
    focal_loss = FocalLoss()

    early_stop = False

    for e in range(epoch):
        if early_stop:
            break
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        # # if (e == 0 or e == epoch-1) and testing==True:
        # if (e == 0 or e == epoch-1):
        #     print(f"Epoch {e}")
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             print(name, param.data)
        #     print(f'Weight 1: {model.conv1.weight.grad}')
        #     print(f'Weight 2: {model.conv2.weight.grad}')
        #     print(f'Weight 3: {model.conv3.weight.grad}')

        # Forward
        final_emb, final_emb_clone = model(g, features)
        
        # for class 6 classification 
        labels = np.reshape(labels, (len(labels), 1))

        # if e == epoch-1:
        #     # print(f"Final emb: {final_emb}")
        #     print(f"Final emb clone: {final_emb_clone}")
        #     # print(f"Labels: {labels}")

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

            if best_validation_loss is None or best_validation_loss > validation_loss:
                best_validation_loss, best_validation_epoch = validation_loss, e
                # print(f'best_validation_loss: {best_validation_loss}')
                # print(f'best_validation_epoch: {best_validation_epoch}')
            if best_validation_epoch < e - max_stagnation:
                early_stop = True
                stopped_epoch[fold_no] = e+1

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if to_print:
        print(f"Overall Train Precision: {overall_precision[-1]}")    
        print(f"Overall Train Recall: {overall_recall[-1]}")    
       
        # with open('output3.txt', 'a') as f:  
        #     f.writelines(f"Overall Train Precision: {overall_precision[-1]}\n")    
        #     f.writelines(f"Overall Train Recall: {overall_recall[-1]}\n")  
        #     f.close() 
        if validate_g != None:
            print(f"\nOverall Validation Precision: {validation_overall_precision[-1]}")    
            print(f"Overall Validation Recall: {validation_overall_recall[-1]}\n")  
            # with open('output3.txt', 'a') as f:
            #     f.writelines(f"\nOverall Validation Precision: {validation_overall_precision[-1]}\n")    
            #     f.writelines(f"Overall Validation Recall: {validation_overall_recall[-1]}\n\n")  
            #     f.close() 
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

            for k in range(len(stopped_epoch)):
                if stopped_epoch[k] == float('inf'):
                    stopped_epoch[k] = epoch

            figure, axis = plt.subplots(2,2)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.7)

            for ax in axis.flat:
                ax.set(xlabel='epoch', ylabel='loss')
            
            if loss_function == 'focal_loss':
                axis[0, 0].set_title("Fold 1 (Focal Loss)")
                axis[0, 1].set_title("Fold 2 (Focal Loss)")
                axis[1, 0].set_title("Fold 3 (Focal Loss)")
                axis[1, 1].set_title("Fold 4 (Focal Loss)")
                axis[0, 0].plot(np.arange(0,stopped_epoch[0]),all_train_loss_fl[0], label = "Training")
                axis[0, 1].plot(np.arange(0,stopped_epoch[1]),all_train_loss_fl[1], label = "Training")
                axis[1, 0].plot(np.arange(0,stopped_epoch[2]),all_train_loss_fl[2], label = "Training")
                axis[1, 1].plot(np.arange(0,stopped_epoch[3]),all_train_loss_fl[3], label = "Training")

                axis[0, 0].plot(np.arange(0,stopped_epoch[0]),all_validation_loss_fl[0], label = "Validation")
                axis[0, 1].plot(np.arange(0,stopped_epoch[1]),all_validation_loss_fl[1], label = "Validation")
                axis[1, 0].plot(np.arange(0,stopped_epoch[2]),all_validation_loss_fl[2], label = "Validation")
                axis[1, 1].plot(np.arange(0,stopped_epoch[3]),all_validation_loss_fl[3], label = "Validation")

            else:
                axis[0, 0].set_title("Fold 1 (Binary Cross Entropy)")
                axis[0, 1].set_title("Fold 2 (Binary Cross Entropy)")
                axis[1, 0].set_title("Fold 3 (Binary Cross Entropy)")
                axis[1, 1].set_title("Fold 4 (Binary Cross Entropy)")
                axis[0, 0].plot(np.arange(0,stopped_epoch[0]),all_train_loss_ce[0], label = "Training")
                axis[0, 1].plot(np.arange(0,stopped_epoch[1]),all_train_loss_ce[1], label = "Training")
                axis[1, 0].plot(np.arange(0,stopped_epoch[2]),all_train_loss_ce[2], label = "Training")
                axis[1, 1].plot(np.arange(0,stopped_epoch[3]),all_train_loss_ce[3], label = "Training")  

                axis[0, 0].plot(np.arange(0,stopped_epoch[0]),all_validation_loss_ce[0], label = "Validation")
                axis[0, 1].plot(np.arange(0,stopped_epoch[1]),all_validation_loss_ce[1], label = "Validation")
                axis[1, 0].plot(np.arange(0,stopped_epoch[2]),all_validation_loss_ce[2], label = "Validation")
                axis[1, 1].plot(np.arange(0,stopped_epoch[3]),all_validation_loss_ce[3], label = "Validation")
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
                axis2[0, 0].plot(np.arange(0,stopped_epoch[0]),all_train_precision_fl[0], label = "Training")
                axis2[0, 1].plot(np.arange(0,stopped_epoch[1]),all_train_precision_fl[1], label = "Training")
                axis2[1, 0].plot(np.arange(0,stopped_epoch[2]),all_train_precision_fl[2], label = "Training")
                axis2[1, 1].plot(np.arange(0,stopped_epoch[3]),all_train_precision_fl[3], label = "Training")

                axis2[0, 0].plot(np.arange(0,stopped_epoch[0]),all_validation_precision_fl[0], label = "Validation")
                axis2[0, 1].plot(np.arange(0,stopped_epoch[1]),all_validation_precision_fl[1], label = "Validation")
                axis2[1, 0].plot(np.arange(0,stopped_epoch[2]),all_validation_precision_fl[2], label = "Validation")
                axis2[1, 1].plot(np.arange(0,stopped_epoch[3]),all_validation_precision_fl[3], label = "Validation")
                

                axis3[0, 0].set_title("Fold 1 (Focal Loss)")
                axis3[0, 1].set_title("Fold 2 (Focal Loss)")
                axis3[1, 0].set_title("Fold 3 (Focal Loss)")
                axis3[1, 1].set_title("Fold 4 (Focal Loss)")
                
                axis3[0, 0].plot(np.arange(0,stopped_epoch[0]),all_train_recall_fl[0], label = "Training")
                axis3[0, 1].plot(np.arange(0,stopped_epoch[1]),all_train_recall_fl[1], label = "Training")
                axis3[1, 0].plot(np.arange(0,stopped_epoch[2]),all_train_recall_fl[2], label = "Training")
                axis3[1, 1].plot(np.arange(0,stopped_epoch[3]),all_train_recall_fl[3], label = "Training")

                axis3[0, 0].plot(np.arange(0,stopped_epoch[0]),all_validation_recall_fl[0], label = "Validation")
                axis3[0, 1].plot(np.arange(0,stopped_epoch[1]),all_validation_recall_fl[1], label = "Validation")
                axis3[1, 0].plot(np.arange(0,stopped_epoch[2]),all_validation_recall_fl[2], label = "Validation")
                axis3[1, 1].plot(np.arange(0,stopped_epoch[3]),all_validation_recall_fl[3], label = "Validation")
            else:
                axis2[0, 0].set_title("Fold 1 (Binary Cross Entropy)")
                axis2[0, 1].set_title("Fold 2 (Binary Cross Entropy)")
                axis2[1, 0].set_title("Fold 3 (Binary Cross Entropy)")
                axis2[1, 1].set_title("Fold 4 (Binary Cross Entropy)")
                axis2[0, 0].plot(np.arange(0,stopped_epoch[0]),all_train_precision_ce[0], label = "Training")
                axis2[0, 1].plot(np.arange(0,stopped_epoch[1]),all_train_precision_ce[1], label = "Training")
                axis2[1, 0].plot(np.arange(0,stopped_epoch[2]),all_train_precision_ce[2], label = "Training")
                axis2[1, 1].plot(np.arange(0,stopped_epoch[3]),all_train_precision_ce[3], label = "Training")  

                axis2[0, 0].plot(np.arange(0,stopped_epoch[0]),all_validation_precision_ce[0], label = "Validation")
                axis2[0, 1].plot(np.arange(0,stopped_epoch[1]),all_validation_precision_ce[1], label = "Validation")
                axis2[1, 0].plot(np.arange(0,stopped_epoch[2]),all_validation_precision_ce[2], label = "Validation")
                axis2[1, 1].plot(np.arange(0,stopped_epoch[3]),all_validation_precision_ce[3], label = "Validation")

                axis3[0, 0].set_title("Fold 1 (Binary Cross Entropy)")
                axis3[0, 1].set_title("Fold 2 (Binary Cross Entropy)")
                axis3[1, 0].set_title("Fold 3 (Binary Cross Entropy)")
                axis3[1, 1].set_title("Fold 4 (Binary Cross Entropy)")
                axis3[0, 0].plot(np.arange(0,stopped_epoch[0]),all_train_recall_ce[0], label = "Training")
                axis3[0, 1].plot(np.arange(0,stopped_epoch[1]),all_train_recall_ce[1], label = "Training")
                axis3[1, 0].plot(np.arange(0,stopped_epoch[2]),all_train_recall_ce[2], label = "Training")
                axis3[1, 1].plot(np.arange(0,stopped_epoch[3]),all_train_recall_ce[3], label = "Training")  

                axis3[0, 0].plot(np.arange(0,stopped_epoch[0]),all_validation_recall_ce[0], label = "Validation")
                axis3[0, 1].plot(np.arange(0,stopped_epoch[1]),all_validation_recall_ce[1], label = "Validation")
                axis3[1, 0].plot(np.arange(0,stopped_epoch[2]),all_validation_recall_ce[2], label = "Validation")
                axis3[1, 1].plot(np.arange(0,stopped_epoch[3]),all_validation_recall_ce[3], label = "Validation")
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

    # features = g.ndata['h']
    features = g.x
    # labels = g.ndata['labels']
    labels = g.y
    final_emb, final_emb_clone = model(g, features)

    # for class 6 classification
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
    # test_tp = 0
    # test_fp = 0
    # test_tn = 0
    # test_fn = 0

    # get features & labels
    features = g.x
    labels = g.y
   
    final_emb, final_emb_clone = model(g, features)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    final_emb = final_emb[-1]
    final_emb_clone = final_emb_clone[-1]

    # for class 6 classification
    labels = np.reshape(labels, (len(labels), 1))
    labels = labels[-1]

    # for row in range(final_emb_clone.shape[0]):
    #     if final_emb_clone[row] == 1.0 and labels[row] == 1.0:
    #         test_tp +=1
    #     elif final_emb_clone[row] == 1.0 and labels[row] == 0.0:
    #         test_fp +=1
    #     elif final_emb_clone[row] == 0.0 and labels[row] == 0.0:
    #         test_tn+=1
    #     elif final_emb_clone[row] == 0.0 and labels[row] == 1.0:
    #         test_fn+=1
    print(f"\nFinal Embedding: {final_emb}")
    print(f"Final Embedding: {final_emb_clone}")
    print(f"Labels: {labels}")
    return final_emb_clone
    # try:
    #     test_precise =test_tp/(test_tp+test_fp)
    #     test_rec = test_tp/(test_tp+test_fn)
    # except:
    #     test_precise = 0
    #     test_rec = 0


    # # test_f1_score = (2 * test_precise * test_rec) / (test_precise + test_rec) 
    # print(f"Overall Test Precision: {test_precise}")    
    # print(f"Overall Test Recall: {test_rec}\n")  
    # # print(f"Test F1 Score: {test_f1_score}")  
    # # with open('output3.txt', 'a') as f:
    # #     f.writelines(f"Overall Test Precision: {test_precise}\n")    
    # #     f.writelines(f"Overall Test Recall: {test_rec}\n")  
    # #     f.close() 

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
print(f'no negative: {no_negative}\n')      
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


# correlation matrix
class_6_node_labels = node_labels[:, 5].reshape(len(node_labels),1)
feat_label_concat = np.concatenate((features, class_6_node_labels), axis=1)

df_features = pd.DataFrame(features, columns = ['Steps', 'Distance', 'Run Distance', 'Calories', 'Heartbeat', 'Sleep', 'SpO2'])
df_features_labels_concat = pd.DataFrame(feat_label_concat, columns = ['Steps', 'Distance', 'Run Distance', 'Calories', 'Heartbeat', 'Sleep', 'SpO2', 'Class 6 Symptoms'])

feat_corr_matrix = df_features.corr()
feat_label_corr_matrix = df_features_labels_concat.corr()
# sns.heatmap(feat_corr_matrix, annot=True)
# sns.heatmap(feat_label_corr_matrix, annot=True)
# plt.show()
# exit()

# 4 fold
k=4
split_len = len(node_labels)//4 

for i in range(k):
    if i == 0:
        validation_min_index=0
        validation_max_index=split_len-1
    elif i == k-1:
        validation_min_index=i*split_len
        validation_max_index=len(features)
    else:
        validation_min_index=i*split_len
        validation_max_index=validation_min_index+(split_len-1)
    
    current_fold_validation_list1 = []
    current_fold_validation_list2 = []
    current_fold_train_list1 = []
    current_fold_train_list2 = []

    # node list 1 & node list 2
    for j in range(len(node_list1)):
        if validation_min_index <= node_list1[j] <= validation_max_index and validation_min_index <= node_list2[j] <= validation_max_index:
            current_fold_validation_list1.append(node_list1[j])
            current_fold_validation_list2.append(node_list2[j])
        else:
            current_fold_train_list1.append(node_list1[j])
            current_fold_train_list2.append(node_list2[j])
    
    missing_train_nodes_index = []
    for j in range(len(features)):
        if j < validation_min_index or j > validation_max_index:
            if j not in current_fold_train_list1:
                missing_train_nodes_index.append(j)

    missing_validation_nodes_index = []  
    for j in range(validation_min_index, validation_max_index):
        if j not in current_fold_validation_list1:
            missing_validation_nodes_index.append(j)

    total_train_nodes_index = list(set(list(set(missing_train_nodes_index)) + list(set(current_fold_train_list1))))
    total_validation_nodes_index = list(set(list(set(missing_validation_nodes_index)) + list(set(current_fold_validation_list1))))

    current_fold_validation_features = []
    current_fold_validation_labels = []
    current_fold_train_features = []
    current_fold_train_labels = []
    
    # features
    for j in range(len(features)):
        if j in total_validation_nodes_index:
            current_fold_validation_features.append(features[j])
        if j in total_train_nodes_index:
            current_fold_train_features.append(features[j])

    current_fold_validation_features = np.delete(current_fold_validation_features, 1, 1)
    current_fold_train_features = np.delete(current_fold_train_features, 1, 1)


    # labels
    for j in range(len(node_labels)):
        if j in total_validation_nodes_index:
            current_fold_validation_labels.append(node_labels[j])
        if j in total_train_nodes_index:
            current_fold_train_labels.append(node_labels[j])
    

    # change validation index
    no_validation_nodes = len(total_validation_nodes_index)

    change_validation_index_dict = {}
    temp_index = validation_min_index
    for j in range(no_validation_nodes):
        change_validation_index_dict[temp_index] = j
        temp_index += 1
    
    for j in range(len(current_fold_validation_list1)):
        current_fold_validation_list1[j] = change_validation_index_dict[current_fold_validation_list1[j]]
        current_fold_validation_list2[j] = change_validation_index_dict[current_fold_validation_list2[j]]

    # change train index
    no_train_nodes = len(total_train_nodes_index)

    change_train_index_dict = {}
    temp_index = 0
    for j in range(no_train_nodes):
        change_train_index_dict[total_train_nodes_index[j]] = temp_index
        temp_index+=1
    
    for j in range(len(current_fold_train_list1)):
        current_fold_train_list1[j] = change_train_index_dict[current_fold_train_list1[j]]
        current_fold_train_list2[j] = change_train_index_dict[current_fold_train_list2[j]]


    # to numpy array
    current_fold_validation_features = np.array(current_fold_validation_features)
    current_fold_validation_labels = np.array(current_fold_validation_labels)
    current_fold_train_features = np.array(current_fold_train_features)
    current_fold_train_labels = np.array(current_fold_train_labels)

    current_fold_validation_labels = current_fold_validation_labels[:,5]
    current_fold_train_labels = current_fold_train_labels[:,5]
    
    current_fold_training_no_pos_label = 0
    current_fold_training_no_neg_label = 0
    current_fold_validation_no_pos_label = 0
    current_fold_validation_no_neg_label = 0

    for j in range(len(current_fold_train_labels)):
        if current_fold_train_labels[j] == 0.0:
            current_fold_training_no_neg_label+=1
        else:
            current_fold_training_no_pos_label+=1
            
    for j in range(len(current_fold_validation_labels)):
        if current_fold_validation_labels[j] == 0.0:
            current_fold_validation_no_neg_label+=1
        else:
            current_fold_validation_no_pos_label+=1
    
    # print(f'Current fold training no of positive label: {current_fold_training_no_pos_label}')
    # print(f'Current fold training no of negative label: {current_fold_training_no_neg_label}\n')
    # print(f'Current fold validation no of positive label: {current_fold_validation_no_pos_label}')
    # print(f'Current fold validation no of negative label: {current_fold_validation_no_neg_label}\n')

    # extract each fatures
    current_fold_train_steps = current_fold_train_features[:, 0]
    current_fold_train_runDistance = current_fold_train_features[:, 1]
    current_fold_train_calories = current_fold_train_features[:, 2]
    current_fold_train_heartbeat = current_fold_train_features[:, 3]
    current_fold_train_sleep = current_fold_train_features[:, 4]
    current_fold_train_spo2 = current_fold_train_features[:, 5]

    current_fold_validation_steps = current_fold_validation_features[:, 0]
    current_fold_validation_runDistance = current_fold_validation_features[:, 1]
    current_fold_validation_calories = current_fold_validation_features[:, 2]
    current_fold_validation_heartbeat = current_fold_validation_features[:, 3]
    current_fold_validation_sleep = current_fold_validation_features[:, 4]
    current_fold_validation_spo2 = current_fold_validation_features[:, 5]
    
    # find wasserstein distance
    was_steps = stats.wasserstein_distance(current_fold_train_steps, current_fold_validation_steps)
    was_runDistance = stats.wasserstein_distance(current_fold_train_runDistance, current_fold_validation_runDistance)
    was_calories = stats.wasserstein_distance(current_fold_train_calories, current_fold_validation_calories)
    was_heartbeat = stats.wasserstein_distance(current_fold_train_heartbeat, current_fold_validation_heartbeat)
    was_sleep = stats.wasserstein_distance(current_fold_train_sleep, current_fold_validation_sleep)
    was_spo2 = stats.wasserstein_distance(current_fold_train_spo2, current_fold_validation_spo2)

    print(f'Fold {i+1}')
    # print(f'Wasserstein distance - steps: {round(was_steps, 4)}')
    # print(f'Wasserstein distance - runDistance: {round(was_runDistance, 4)}')
    # print(f'Wasserstein distance - calories: {round(was_calories, 4)}')
    # print(f'Wasserstein distance - heartbeat: {round(was_heartbeat, 4)}')
    # print(f'Wasserstein distance - sleep: {round(was_sleep, 4)}')
    # print(f'Wasserstein distance - spo2: {round(was_spo2, 4)}\n')
        
    # training mean & std
    current_fold_train_steps_mean = np.mean(np.array(current_fold_train_steps))
    current_fold_train_steps_std = np.std(np.array(current_fold_train_steps))

    current_fold_train_runDistance_mean = np.mean(np.array(current_fold_train_runDistance))
    current_fold_train_runDistance_std = np.std(np.array(current_fold_train_runDistance))

    current_fold_train_calories_mean = np.mean(np.array(current_fold_train_calories))
    current_fold_train_calories_std = np.std(np.array(current_fold_train_calories))

    current_fold_train_heartbeat_mean = np.mean(np.array(current_fold_train_heartbeat))
    current_fold_train_heartbeat_std = np.std(np.array(current_fold_train_heartbeat))

    current_fold_train_sleep_mean = np.mean(np.array(current_fold_train_sleep))
    current_fold_train_sleep_std = np.std(np.array(current_fold_train_sleep))

    current_fold_train_spo2_mean = np.mean(np.array(current_fold_train_spo2))
    current_fold_train_spo2_std = np.std(np.array(current_fold_train_spo2))


    # validation mean & std
    current_fold_validation_steps_mean = np.mean(np.array(current_fold_validation_steps))
    current_fold_validation_steps_std = np.std(np.array(current_fold_validation_steps))

    current_fold_validation_runDistance_mean = np.mean(np.array(current_fold_validation_runDistance))
    current_fold_validation_runDistance_std = np.std(np.array(current_fold_validation_runDistance))

    current_fold_validation_calories_mean = np.mean(np.array(current_fold_validation_calories))
    current_fold_validation_calories_std = np.std(np.array(current_fold_validation_calories))

    current_fold_validation_heartbeat_mean = np.mean(np.array(current_fold_validation_heartbeat))
    current_fold_validation_heartbeat_std = np.std(np.array(current_fold_validation_heartbeat))

    current_fold_validation_sleep_mean = np.mean(np.array(current_fold_validation_sleep))
    current_fold_validation_sleep_std = np.std(np.array(current_fold_validation_sleep))

    current_fold_validation_spo2_mean = np.mean(np.array(current_fold_validation_spo2))
    current_fold_validation_spo2_std = np.std(np.array(current_fold_validation_spo2))

    # print pdf
    # plt.plot(x, stats.norm.pdf(x,loc=x_loc, scale=x_scale))


    # # print out mean & std
    # print(f'Mean steps (training): {round(current_fold_train_steps_mean, 4)}')
    # print(f'Std steps (training): {round(current_fold_train_steps_std, 4)}\n')

    # print(f'Mean distance (training): {round(current_fold_train_distance_mean, 4)}')
    # print(f'Std distance (training): {round(current_fold_train_distance_std, 4)}\n')

    # print(f'Mean run distance (training): {round(current_fold_train_runDistance_mean, 4)}')
    # print(f'Std run distance (training): {round(current_fold_train_runDistance_std, 4)}\n')

    # print(f'Mean run calories (training): {round(current_fold_train_calories_mean, 4)}')
    # print(f'Std run calories (training): {round(current_fold_train_calories_std, 4)}\n')

    # print(f'Mean run heartrate (training): {round(current_fold_train_heartbeat_mean, 4)}')
    # print(f'Std run heartrate (training): {round(current_fold_train_heartbeat_std, 4)}\n')

    # print(f'Mean run sleep (training): {round(current_fold_train_sleep_mean, 4)}')
    # print(f'Std run sleep (training): {round(current_fold_train_sleep_std, 4)}\n')

    # print(f'Mean run spo2 (training): {round(current_fold_train_spo2_mean, 4)}')
    # print(f'Std run spo2 (training): {round(current_fold_train_spo2_std, 4)}\n')

    # print(f'Mean steps (validation): {round(current_fold_validation_steps_mean, 4)}')
    # print(f'Std steps (validation): {round(current_fold_validation_steps_std, 4)}\n')

    # print(f'Mean distance (validation): {round(current_fold_validation_distance_mean, 4)}')
    # print(f'Std distance (validation): {round(current_fold_validation_distance_std, 4)}\n')

    # print(f'Mean run distance (validation): {round(current_fold_validation_runDistance_mean, 4)}')
    # print(f'Std run distance (validation): {round(current_fold_validation_runDistance_std, 4)}\n')

    # print(f'Mean run calories (validation): {round(current_fold_validation_calories_mean, 4)}')
    # print(f'Std run calories (validation): {round(current_fold_validation_calories_std, 4)}\n')

    # print(f'Mean run heartrate (validation): {round(current_fold_validation_heartbeat_mean, 4)}')
    # print(f'Std run heartrate (validation): {round(current_fold_validation_heartbeat_std, 4)}\n')

    # print(f'Mean run sleep (validation): {round(current_fold_validation_sleep_mean, 4)}')
    # print(f'Std run sleep (validation): {round(current_fold_validation_sleep_std, 4)}\n')

    # print(f'Mean run spo2 (validation): {round(current_fold_validation_spo2_mean, 4)}')
    # print(f'Std run spo2 (validation): {round(current_fold_validation_spo2_std, 4)}\n')
    # if i == 3:
    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_train_steps), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_train_steps), color='crimson')
    #     plt.xlabel('Steps')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Steps (Fold {i+1} - Training)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_train_distance), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_train_distance), color='crimson')
    #     plt.xlabel('Distance')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Distance (Fold {i+1} - Training)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_train_runDistance), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_train_runDistance), color='crimson')
    #     plt.xlabel('Run Distance')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Run Distance (Fold {i+1} - Training)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_train_calories), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_train_calories), color='crimson')
    #     plt.xlabel('Calories')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Calories (Fold {i+1} - Training)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_train_heartbeat), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_train_heartbeat), color='crimson')
    #     plt.xlabel('Heartbeat')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Heartbeat (Fold {i+1} - Training)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_train_sleep), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_train_sleep), color='crimson')
    #     plt.xlabel('Sleep')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Sleep (Fold {i+1} - Training)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_train_spo2), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_train_spo2), color='crimson')
    #     plt.xlabel('SpO2')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram SpO2 (Fold {i+1} - Training)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_validation_steps), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_validation_steps), color='crimson')
    #     plt.xlabel('Steps')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Steps (Fold {i+1} - Validation)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_validation_distance), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_validation_distance), color='crimson')
    #     plt.xlabel('Distance')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Distance (Fold {i+1} - Validation)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_validation_runDistance), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_validation_runDistance), color='crimson')
    #     plt.xlabel('Run Distance')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Run Distance (Fold {i+1} - Validation)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_validation_calories), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_validation_calories), color='crimson')
    #     plt.xlabel('Calories')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Calories (Fold {i+1} - Validation)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_validation_heartbeat), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_validation_heartbeat), color='crimson')
    #     plt.xlabel('Heartbeat')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Heartbeat (Fold {i+1} - Validation)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_validation_sleep), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_validation_sleep), color='crimson')
    #     plt.xlabel('Sleep')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram Sleep (Fold {i+1} - Validation)')
    #     plt.draw()

    #     figure, axis = plt.subplots(1,1)
    #     sns.histplot(data=np.array(current_fold_validation_spo2), kde=False, stat='count')
    #     # sns.kdeplot(data=np.array(current_fold_validation_spo2), color='crimson')
    #     plt.xlabel('SpO2')
    #     plt.ylabel('count')
    #     plt.title(f'Histogram SpO2 (Fold {i+1} - Validation)')
    #     plt.draw()

    # train
    train_edge_index = torch.tensor(([current_fold_train_list1, current_fold_train_list2]), dtype=torch.long)
    train_edge_index = add_self_loops(edge_index = train_edge_index, num_nodes=len(current_fold_train_labels))[0]
    train_graph = Data(x = torch.from_numpy(current_fold_train_features).float(), edge_index = train_edge_index, y = torch.from_numpy(current_fold_train_labels).float())

    validation_edge_index = torch.tensor(([current_fold_validation_list1, current_fold_validation_list2]), dtype=torch.long)
    validation_edge_index = add_self_loops(edge_index = validation_edge_index, num_nodes=len(current_fold_validation_labels))[0]
    validation_graph = Data(x = torch.from_numpy(current_fold_validation_features).float(), edge_index = validation_edge_index, y = torch.from_numpy(current_fold_validation_labels).float())

    model = GCN(6, 1)
    # train(g=train_graph, validate_g =validation_graph, model=model, loss_function = 'ce', fold=k, fold_no=i)

# ######################### test# #####################################################
print(f'Testing')
# this is for class 6 classification
node_labels = node_labels[:,5]
overall_training_features = np.delete(features, 1, 1)
train_edge_index_2 = torch.tensor(([node_list1, node_list2]), dtype=torch.long)
train_edge_index_2 = add_self_loops(edge_index = train_edge_index_2, num_nodes=len(node_labels))[0]
train_graph_2 = Data(x = torch.from_numpy(overall_training_features).float(), edge_index = train_edge_index_2, y = torch.from_numpy(node_labels).float())

model2 = GCN(6, 1)
train(g = train_graph_2, model = model2, loss_function = 'ce', to_print=True, testing=True)


features_array = np.array([
    [1028, 52, 28, 87, average_sleep, 97],
    [21, 41, 12, 81, 1.167, 100],
    [947, 142, 26, 77, 8.283, 96],
    [1457, 288, 33, 92, 9.983, 92],
    [7402, 266, 127, 94, average_sleep, 95],
])

for i in range(features_array.shape[0]):
    features_array[i,0] = round((features_array[i,0] - min_steps)/(max_steps - min_steps),10)
    features_array[i,1] = round((features_array[i,1] - min_runDistance)/(max_runDistance - min_runDistance),10)
    features_array[i,2] = round((features_array[i,2] - min_calories)/(max_calories - min_calories),10)
    features_array[i,3] = round((features_array[i,3] - min_heartbeat)/(max_heartbeat - min_heartbeat),10)
    features_array[i,4] = round((features_array[i,4] - min_sleep)/(max_sleep - min_sleep),10)
    features_array[i,5] = features_array[i,5]/100

test_label = np.array([
[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])
test_label = test_label[:,5]

num_of_nodes = len(node_labels)
test_data_index = num_of_nodes
test_tp = 0
test_fp = 0
test_tn = 0
test_fn = 0

for i in range(len(features_array)):
    test_data_features = [features_array[i]]
    test_data_labels = [test_label[i]]
    overall_features_include_test = np.concatenate((overall_training_features, test_data_features))
    overall_labels_include_test = np.concatenate((node_labels, test_data_labels))
    overall_node_list1_include_test = np.copy(node_list1)
    overall_node_list2_include_test = np.copy(node_list2)
    
    for j in range(num_of_nodes):
        if test_data_labels == node_labels[j]:
            overall_node_list1_include_test = np.concatenate((overall_node_list1_include_test,[test_data_index]))
            overall_node_list2_include_test = np.concatenate((overall_node_list2_include_test,[j]))
            overall_node_list1_include_test = np.concatenate((overall_node_list1_include_test,[j]))
            overall_node_list2_include_test = np.concatenate((overall_node_list2_include_test,[test_data_index]))
    
    overall_test_edge_index = torch.tensor(([overall_node_list1_include_test, overall_node_list2_include_test]), dtype=torch.long)
    overall_test_edge_index = add_self_loops(edge_index = overall_test_edge_index, num_nodes=len(overall_labels_include_test))[0]
    
    test_graph = Data(x = torch.from_numpy(overall_features_include_test).float(), edge_index = overall_test_edge_index, y = torch.from_numpy(overall_labels_include_test).float())
    pred = test(test_graph,model2)

    if pred == 1.0 and test_data_labels[0] == 1.0:
        test_tp +=1
    elif pred == 1.0 and test_data_labels[0] == 0.0:
        test_fp +=1
    elif pred == 0.0 and test_data_labels[0] == 0.0:
        test_tn+=1
    elif pred == 0.0 and test_data_labels[0] == 1.0:
        test_fn+=1

try:
    test_precise =test_tp/(test_tp+test_fp)
    test_rec = test_tp/(test_tp+test_fn)
except:
    test_precise = 0
    test_rec = 0


print(f"Overall Test Precision: {test_precise}")    
print(f"Overall Test Recall: {test_rec}\n")  


# # test
# test_graph= dgl.graph(([0,1,0,3,1,2], [1,0,3,0,2,1]), num_nodes=5)

# features_array = np.array([
#     [1028, 52, 28, 87, average_sleep, 97],
#     [21, 41, 12, 81, 1.167, 100],
#     [947, 142, 26, 77, 8.283, 96],
#     [1457, 288, 33, 92, 9.983, 92],
#     [7402, 266, 127, 94, average_sleep, 95],
# ])

# for i in range(features_array.shape[0]):
#     features_array[i,0] = round((features_array[i,0] - min_steps)/(max_steps - min_steps),10)
#     features_array[i,1] = round((features_array[i,1] - min_runDistance)/(max_runDistance - min_runDistance),10)
#     features_array[i,2] = round((features_array[i,2] - min_calories)/(max_calories - min_calories),10)
#     features_array[i,3] = round((features_array[i,3] - min_heartbeat)/(max_heartbeat - min_heartbeat),10)
#     features_array[i,4] = round((features_array[i,4] - min_sleep)/(max_sleep - min_sleep),10)
#     features_array[i,5] = features_array[i,5]/100
    
# test_edge_index = torch.tensor(([0,1,0,3,1,2], [1,0,3,0,2,1]), dtype=torch.long)
# test_edge_index = add_self_loops(edge_index = test_edge_index)[0]

# test_label = np.array([
# [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
# [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
# [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
# [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# ])
# test_label = test_label[:,5]
# test_graph = Data(x = torch.from_numpy(features_array).float(), edge_index = test_edge_index, y = torch.from_numpy(test_label).float())



# print(f'----------------------Binary Cross Entropy Loss----------------------')

# test(test_graph,model2)

# plt.show()