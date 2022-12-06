import pandas as pd
import math
import numpy as np
import datetime as dt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

# date{email{sp02, symptoms}}

# 'yizentan@gmail.com':[],
# 'afrinaad@gmail.com':[],
# 'darrensow.01@gmail.com':[],
# 'drsawband@gmail.com':[],
# 'hong_chiow0316@yahoo.com':[],
# 'elainemap16@gmail.com':[],
# 'hamizahhelmie@gmail.com':[],

email_list = []

questionnaire_data = {}

all_email_list = ['yizentan@gmail.com',
'afrinaad@gmail.com',
'darrensow.01@gmail.com',
'drsawband@gmail.com',
'hong_chiow0316@yahoo.com',
'elainemap16@gmail.com',
'hamizahhelmie@gmail.com',]

df2_date_list = [
    '01-07-22',
    '04-07-22',
    '06-07-22',
    '08-07-22',
    '11-07-22',
    '13-07-22',
    '15-07-22',
    '18-07-22',
    '20-07-22',
    '22-07-22',
    '25-07-22',
    '27-07-22',
    '29-07-22',
]

date_list = [
    # '2022-06-13',
    # '2022-06-15',
    # '2022-06-17',
    # '2022-06-20',
    # '2022-06-22',
    # '2022-06-25',
    # '2022-06-27',
    # '2022-06-29',
    # '2022-07-01',
    # '2022-07-04',
    # '2022-07-06',
    # '2022-07-08',
    # '2022-07-11',
    # '2022-07-13',
    # '2022-07-15',
    # '2022-07-18',
    # '2022-07-20',
    # '2022-07-22',
    # '2022-07-25',
    # '2022-07-27',
    # '2022-07-29',
    '2022-08-01',
    '2022-08-03',
    '2022-08-05',
    '2022-08-08',
    '2022-08-10',
    '2022-08-12',
    '2022-08-15',
    '2022-08-17',
    '2022-08-19',
    '2022-08-22',
    '2022-08-24',
    '2022-08-26',
    '2022-08-29',
    '2022-08-31',
   ]

# ---------------------------- symptoms --------------------------------------    
excel = [
# 'Daily Symptoms Questionnaire (13_6) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (15_6) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (17_6) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (20_6) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (22_6) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (24_6) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (27_6) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (29_6) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (1_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (4_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (6_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (8_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (11_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (13_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (15_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (18_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (20_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (22_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (25_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (27_7) (Responses).xlsx',
# 'Daily Symptoms Questionnaire (29_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (1_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (3_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (5_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (8_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (10_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (12_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (15_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (17_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (19_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (22_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (24_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (26_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (29_8) (Responses).xlsx',
'Daily Symptoms Questionnaire (31_8) (Responses).xlsx',
]
for index in range(len(excel)):
    df = pd.read_excel(excel[index])
    row = df.shape[0]
    col = df.shape[1]

    people = {} 

    for i in range(row):
        email = df.loc[i]['Email Address']
        spo2 = df.iloc[i]['Oxygen level (SpO2) (%)']
        symptoms = ''
        for j in range(4,13):
            if type(df.iloc[i,j]) == str and df.iloc[i,j] != 'Normal':
                symptoms += '1'
            else:
                symptoms += '0'

        people[email] = {
            'spo2': round(spo2/100,3),
            'symptoms': symptoms
        }
    
    questionnaire_data[date_list[index]] = people


# --------------------------------- vital signs ---------------------------------
csv = [
    # ['../../Data/Tan Yi Zen/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659266515815.csv', '../../Data/Tan Yi Zen/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659266515958.csv', '../../Data/Tan Yi Zen/1 Jun - 30 Jun/SLEEP/SLEEP_1659266515875.csv'],
    # ['../../Data/Elaine/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659340973129.csv', '../../Data/Elaine/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659340973647.csv', '../../Data/Elaine/1 Jun - 30 Jun/SLEEP/SLEEP_1659340973376.csv'],
    # ['../../Data/Chiow/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659593423752.csv', '../../Data/Chiow/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659593424267.csv', '../../Data/Chiow/1 Jun - 30 Jun/SLEEP/SLEEP_1659593423989.csv'],
    # ['../../Data/Darren/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659592313228.csv', '../../Data/Darren/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659592313371.csv', '../../Data/Darren/1 Jun - 30 Jun/SLEEP/SLEEP_1659592313301.csv'],
    # ['../../Data/Darren/1 July - 31 July/ACTIVITY/ACTIVITY_1663271695902.csv', '../../Data/Darren/1 July - 31 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1663271696050.csv', '../../Data/Darren/1 July - 31 July/SLEEP/SLEEP_1663271696009.csv'],
    # ['../../Data/Chiow/1 July - 31 July/ACTIVITY/ACTIVITY_1661786663870.csv', '../../Data/Chiow/1 July - 31 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1661786664512.csv', '../../Data/Chiow/1 July - 31 July/SLEEP/SLEEP_1661786664202.csv'],
    # ['../../Data/Elaine/1 July - 16 July/ACTIVITY/ACTIVITY_1662768816086.csv', '../../Data/Elaine/1 July - 16 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1662768816736.csv', '../../Data/Elaine/1 July - 16 July/SLEEP/SLEEP_1662768816417.csv'],
    # ['../../Data/Tan Yi Zen/1 July - 16 July/ACTIVITY/ACTIVITY_1661786026474.csv', '../../Data/Tan Yi Zen/1 July - 16 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1661786026621.csv', '../../Data/Tan Yi Zen/1 July - 16 July/SLEEP/SLEEP_1661786026539.csv'],
    # ['../../Data/Hamizah/1 July - 31 July/ACTIVITY/ACTIVITY_1662646712049.csv', '../../Data/Hamizah/1 July - 31 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1662646712481.csv', '../../Data/Hamizah/1 July - 31 July/SLEEP/SLEEP_1662646712303.csv'],
    # ['../../Data/Elaine/17 July - 31 July/ACTIVITY/ACTIVITY_1662768816087.csv', '../../Data/Elaine/17 July - 31 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1662768816736.csv', '../../Data/Elaine/17 July - 31 July/SLEEP/SLEEP_1662768816417.csv'],
    # ['../../Data/Tan Yi Zen/17 July - 31 July/ACTIVITY/ACTIVITY_1661786141648.csv', '../../Data/Tan Yi Zen/17 July - 31 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1661786141770.csv', '../../Data/Tan Yi Zen/17 July - 31 July/SLEEP/SLEEP_1661786141705.csv'],
    ['../../Data/Afrina/1 August - 31 August/ACTIVITY/ACTIVITY_1662634396271.csv', '../../Data/Afrina/1 August - 31 August/HEARTRATE_AUTO/HEARTRATE_AUTO_1662634396455.csv', '../../Data/Afrina/1 August - 31 August/SLEEP/SLEEP_1662634396373.csv'],
    ['../../Data/Chiow/1 August - 31 August/ACTIVITY/ACTIVITY_1662635449661.csv', '../../Data/Chiow/1 August - 31 August/HEARTRATE_AUTO/HEARTRATE_AUTO_1662635450144.csv', '../../Data/Chiow/1 August - 31 August/SLEEP/SLEEP_1662635449908.csv'],
    ['../../Data/Darren/1 August - 31 August/ACTIVITY/ACTIVITY_1662646831765.csv', '../../Data/Darren/1 August - 31 August/HEARTRATE_AUTO/HEARTRATE_AUTO_1662646831913.csv', '../../Data/Darren/1 August - 31 August/SLEEP/SLEEP_1662646831868.csv'],
    ['../../Data/Dr Saw/1 August - 28 August/ACTIVITY/ACTIVITY_1662647187804.csv', '../../Data/Dr Saw/1 August - 28 August/HEARTRATE_AUTO/HEARTRATE_AUTO_1662647188030.csv', '../../Data/Dr Saw/1 August - 28 August/SLEEP/SLEEP_1662647187889.csv'],
    ['../../Data/Dr Saw/29 August - 31 August/ACTIVITY/ACTIVITY_1662647249898.csv', '../../Data/Dr Saw/29 August - 31 August/HEARTRATE_AUTO/HEARTRATE_AUTO_1662647249972.csv', '../../Data/Dr Saw/29 August - 31 August/SLEEP/SLEEP_1662647249935.csv'],
    ['../../Data/Elaine/1 August - 31 August/ACTIVITY/ACTIVITY_1662768927375.csv', '../../Data/Elaine/1 August - 31 August/HEARTRATE_AUTO/HEARTRATE_AUTO_1662768927718.csv', '../../Data/Elaine/1 August - 31 August/SLEEP/SLEEP_1662768927549.csv'],
    ['../../Data/Hamizah/1 August - 31 August/ACTIVITY/ACTIVITY_1662646766347.csv', '../../Data/Hamizah/1 August - 31 August/HEARTRATE_AUTO/HEARTRATE_AUTO_1662646766876.csv', '../../Data/Hamizah/1 August - 31 August/SLEEP/SLEEP_1663297369216.csv'],
    ['../../Data/Tan Yi Zen/1 August - 31 August/ACTIVITY/ACTIVITY_1662633836007.csv', '../../Data/Tan Yi Zen/1 August - 31 August/HEARTRATE_AUTO/HEARTRATE_AUTO_1662633836380.csv', '../../Data/Tan Yi Zen/1 August - 31 August/SLEEP/SLEEP_1662633836216.csv'],
]

vitalsigns_data = {}

steps_arr = []
distance_arr = []
runDistance_arr = []
calories_arr = []
heartbeat_arr = []
sleep_arr = []

steps_mean=0
steps_std=0
distance_mean=0
distance_std=0
runDistance_mean=0
runDistance_std=0
calories_mean=0
calories_std=0
heartbeat_mean=0
heartbeat_std=0
sleep_mean=0
sleep_std=0


for i in range(len(csv)):
    if 'Tan Yi Zen' in csv[i][0]:
        email = 'yizentan@gmail.com'
    elif 'Elaine' in csv[i][0]:
        email = 'elainemap16@gmail.com'
    elif 'Afrina' in csv[i][0]:
        email = 'afrinaad@gmail.com'
    elif 'Darren' in csv[i][0]:
        email = 'darrensow.01@gmail.com'
    elif 'Chiow' in csv[i][0]:
        email = 'hong_chiow0316@yahoo.com'
    elif 'Hamizah' in csv[i][0]:
        email = 'hamizahhelmie@gmail.com'
    elif 'Dr Saw' in csv[i][0]:
        email = 'drsawband@gmail.com'
    
    df0 = pd.read_csv(csv[i][0])
    df1 = pd.read_csv(csv[i][1])
    df2 = pd.read_csv(csv[i][2],on_bad_lines='skip')

    for date in date_list:
        if 'Jun' in csv[i][0]:
            try:
                steps = df0.loc[date]['steps']
                distance = df0.loc[date]['distance']
                runDistance = df0.loc[date]['runDistance']
                calories = df0.loc[date]['calories']
                heartbeat = math.ceil(df1.loc[df1['date'] == date]['heartRate'].mean())
                sleep = round((df2.loc[date]['stop'] - df2.loc[date]['start'])/3600,3)

                steps_arr.append(steps)
                distance_arr.append(distance)
                runDistance_arr.append(runDistance)
                calories_arr.append(calories)
                heartbeat_arr.append(heartbeat)
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
        elif 'July' in csv[i][0]:
            try:
                steps = df0.loc[df0['date'] == date]['steps'].item()
                
                distance = df0.loc[df0['date'] == date]['distance'].item()
                runDistance = df0.loc[df0['date'] == date]['runDistance'].item()
                calories = df0.loc[df0['date'] == date]['calories'].item()
                heartbeat = math.ceil(df1.loc[df1['date'] == date]['heartRate'].mean())
                if 'Darren' in csv[i][0] or 'Hamizah' in csv[i][0]:
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
                    split_date = date.split('-')
                    date = split_date[2]+'-'+split_date[1]+'-22'
                    
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
                    
                steps_arr.append(steps)
                distance_arr.append(distance)
                runDistance_arr.append(runDistance)
                calories_arr.append(calories)
                heartbeat_arr.append(heartbeat)
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

        elif 'August' in csv[i][0]:
            try:
                steps = df0.loc[df0['date'] == date]['steps'].item()
                distance = df0.loc[df0['date'] == date]['distance'].item()
                runDistance = df0.loc[df0['date'] == date]['runDistance'].item()
                calories = df0.loc[df0['date'] == date]['calories'].item()
                heartbeat = math.ceil(df1.loc[df1['date'] == date]['heartRate'].mean())
                if 'Hamizah' in csv[i][0]:
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

                steps_arr.append(steps)
                distance_arr.append(distance)
                runDistance_arr.append(runDistance)
                calories_arr.append(calories)
                heartbeat_arr.append(heartbeat)
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



average_sleep = np.mean(sleep_arr)

for i in range(len(sleep_arr)):
    if sleep_arr[i] == 0.0:
        sleep_arr[i] = round(average_sleep,3)

for date in date_list:
    for email in all_email_list:
        try:
            if vitalsigns_data[date][email]['sleep'] == 0.0:
                vitalsigns_data[date][email]['sleep'] = round(average_sleep,3)            
        except:
            continue

total_steps = np.sum(steps_arr)
mean_steps = np.mean(steps_arr)
std_steps = np.std(steps_arr)

total_distance = np.sum(distance_arr)
mean_distance = np.mean(distance_arr)
std_distance = np.std(distance_arr)

total_runDistance = np.sum(runDistance_arr)
mean_runDistance = np.mean(runDistance_arr)
std_runDistance = np.std(runDistance_arr)

total_calories = np.sum(calories_arr)
mean_calories = np.mean(calories_arr)
std_calories = np.std(calories_arr)

total_heartbeat = np.sum(heartbeat_arr)
mean_heartbeat = np.mean(heartbeat_arr)
std_heartbeat = np.std(heartbeat_arr)

total_sleep = np.sum(sleep_arr)
mean_sleep = np.mean(sleep_arr)
std_sleep = np.std(sleep_arr)

for date in date_list:
    for email in all_email_list:
        try:
            vitalsigns_data[date][email]['steps'] = round((vitalsigns_data[date][email]['steps']-mean_steps)/std_steps,3)
            vitalsigns_data[date][email]['distance'] = round((vitalsigns_data[date][email]['distance']-mean_distance)/std_distance,3)
            vitalsigns_data[date][email]['runDistance'] = round((vitalsigns_data[date][email]['runDistance']-mean_runDistance)/std_runDistance,3)
            vitalsigns_data[date][email]['calories'] = round((vitalsigns_data[date][email]['calories']-mean_calories)/std_calories,3)
            vitalsigns_data[date][email]['heartbeat'] = round((vitalsigns_data[date][email]['heartbeat']-mean_heartbeat)/std_heartbeat,3)
            vitalsigns_data[date][email]['sleep'] = round((vitalsigns_data[date][email]['sleep']-mean_sleep)/std_sleep,3)
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


# --------------------------- transform data ----------------------------------
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

# # ---------------------- train ------------------------------------
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

def train(g, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tp=0
    fp=0
    tn=0
    fn=0
    loss_arr = []
    precision = []
    recall = []

    features = g.ndata['h']
    labels = g.ndata['labels']
    labels_shape = labels.shape
    epoch = 100

    for e in range(epoch):
        tp=0
        fp=0
        tn=0
        fn=0

        # Forward
        final_emb, final_emb_clone = model(g, features)

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

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Train Precision: {precision[-1]}")    
    print(f"Train Recall: {recall[-1]}")    

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

    try:
        test_precise = test_tp/(test_tp+test_fp)
        test_rec = test_tp/(test_tp+test_fn)
    except:
        test_precise = 0
        test_rec = 0

    print(f"\nTest Precision: {test_precise}")    
    print(f"Test Recall: {test_rec}")    

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
test_graph = dgl.graph(([0,0,0,1,4,2,1,1,1,4,2,3,4,4,2,3], [1,4,2,0,0,0,4,2,3,1,1,1,2,3,4,4]), num_nodes=5)

features_array = np.array([
    [9486, 6005, 458, 146, 84, 3.517, 95],
    [4284, 2916, 177, 108, 82, 5.3, 95],
    [2692, 1425, 423, 64, 81, 4.717, 97],
    [2577, 1809, 141,66,70,6.683,96],
    [4688, 2955, 390, 99, 86, 5.054, 95],
])

for i in range(features_array.shape[0]):
    features_array[i,0] = round((features_array[i,0]-mean_steps)/std_steps,3)
    features_array[i,1] = round((features_array[i,1]-mean_distance)/std_distance,3)
    features_array[i,2] = round((features_array[i,2]-mean_runDistance)/std_runDistance,3)
    features_array[i,3] = round((features_array[i,3]-mean_calories)/std_calories,3)
    features_array[i,4] = round((features_array[i,4]-mean_heartbeat)/std_heartbeat,3)
    features_array[i,5] = round((features_array[i,5]-mean_sleep)/std_sleep,3)
    features_array[i,6] = features_array[i,6]/100
    

test_graph.ndata['h'] = torch.from_numpy(features_array).float()

test_graph = dgl.add_self_loop(test_graph)

test_graph.ndata['labels'] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
])

G = test_graph.to_networkx()
options = {
    'node_color': 'black',
    'node_size': 20,
    'width': 1,
}

test(test_graph,model)
print('=================================================')






