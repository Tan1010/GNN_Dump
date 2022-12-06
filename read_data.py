import pandas as pd
import math
import numpy as np
import datetime as dt

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
    '2022-06-13',
    '2022-06-15',
    '2022-06-17',
    '2022-06-20',
    '2022-06-22',
    '2022-06-25',
    '2022-06-27',
    '2022-06-29',
    '2022-07-01',
    '2022-07-04',
    '2022-07-06',
    '2022-07-08',
    '2022-07-11',
    '2022-07-13',
    '2022-07-15',
    '2022-07-18',
    '2022-07-20',
    '2022-07-22',
    '2022-07-25',
    '2022-07-27',
    '2022-07-29',
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
'Daily Symptoms Questionnaire (13_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (15_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (17_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (20_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (22_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (24_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (27_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (29_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (1_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (4_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (6_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (8_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (11_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (13_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (15_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (18_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (20_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (22_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (25_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (27_7) (Responses).xlsx',
'Daily Symptoms Questionnaire (29_7) (Responses).xlsx',
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
    ['../../Data/Tan Yi Zen/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659266515815.csv', '../../Data/Tan Yi Zen/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659266515958.csv', '../../Data/Tan Yi Zen/1 Jun - 30 Jun/SLEEP/SLEEP_1659266515875.csv'],
    ['../../Data/Elaine/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659340973129.csv', '../../Data/Elaine/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659340973647.csv', '../../Data/Elaine/1 Jun - 30 Jun/SLEEP/SLEEP_1659340973376.csv'],
    ['../../Data/Chiow/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659593423752.csv', '../../Data/Chiow/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659593424267.csv', '../../Data/Chiow/1 Jun - 30 Jun/SLEEP/SLEEP_1659593423989.csv'],
    ['../../Data/Darren/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659592313228.csv', '../../Data/Darren/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659592313371.csv', '../../Data/Darren/1 Jun - 30 Jun/SLEEP/SLEEP_1659592313301.csv'],
    ['../../Data/Darren/1 July - 31 July/ACTIVITY/ACTIVITY_1663271695902.csv', '../../Data/Darren/1 July - 31 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1663271696050.csv', '../../Data/Darren/1 July - 31 July/SLEEP/SLEEP_1663271696009.csv'],
    ['../../Data/Chiow/1 July - 31 July/ACTIVITY/ACTIVITY_1661786663870.csv', '../../Data/Chiow/1 July - 31 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1661786664512.csv', '../../Data/Chiow/1 July - 31 July/SLEEP/SLEEP_1661786664202.csv'],
    ['../../Data/Elaine/1 July - 16 July/ACTIVITY/ACTIVITY_1662768816086.csv', '../../Data/Elaine/1 July - 16 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1662768816736.csv', '../../Data/Elaine/1 July - 16 July/SLEEP/SLEEP_1662768816417.csv'],
    ['../../Data/Tan Yi Zen/1 July - 16 July/ACTIVITY/ACTIVITY_1661786026474.csv', '../../Data/Tan Yi Zen/1 July - 16 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1661786026621.csv', '../../Data/Tan Yi Zen/1 July - 16 July/SLEEP/SLEEP_1661786026539.csv'],
    ['../../Data/Hamizah/1 July - 31 July/ACTIVITY/ACTIVITY_1662646712049.csv', '../../Data/Hamizah/1 July - 31 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1662646712481.csv', '../../Data/Hamizah/1 July - 31 July/SLEEP/SLEEP_1662646712303.csv'],
    ['../../Data/Elaine/17 July - 31 July/ACTIVITY/ACTIVITY_1662768816087.csv', '../../Data/Elaine/17 July - 31 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1662768816736.csv', '../../Data/Elaine/17 July - 31 July/SLEEP/SLEEP_1662768816417.csv'],
    ['../../Data/Tan Yi Zen/17 July - 31 July/ACTIVITY/ACTIVITY_1661786141648.csv', '../../Data/Tan Yi Zen/17 July - 31 July/HEARTRATE_AUTO/HEARTRATE_AUTO_1661786141770.csv', '../../Data/Tan Yi Zen/17 July - 31 July/SLEEP/SLEEP_1661786141705.csv'],
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
    # print(df2)
    # if email == 'hong_chiow0316@yahoo.com':
    #     print(df2)
    #     start = str(df2.loc['2022-08-01']['date']).split()[3]+' '+str(df2.loc['2022-08-01']['date']).split()[4]
    #     print(f'start: {start}')
    #     print(f"Date: {df2.loc[df2['date'] ==  '01-07-22']['stop']}")
    #     print(f"Date: {df2.loc[df2['date'] ==  '01-07-22']['start']}")
        # print(f"Date: {df2.loc[df2[date] == '01-07-22']}")
        # print(f"Date: {df2.loc['2022-07-03']['date']}")
        # print(f"Date1: {str(df2.loc['2022-07-03']['date']).split()[3]} {str(df2.loc['2022-07-01']['date']).split()[4]}")
        # print(f"Date2: {df2.loc['2022-07-03']['date'].item()}")
        # print('=========================')
    for date in date_list:
        if 'Jun' in csv[i][0]:
            try:
                steps = df0.loc[date]['steps']
                # print(f"June steps: {steps}")
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
                # print(f'email: {email}')
                # print(f'people: {people}')
                if vitalsigns_data.get(date) == None:
                    vitalsigns_data[date] = {}
                    vitalsigns_data[date][email] = people
                else:
                    # print(f'Here')
                    # print(f'Email: {email}')
                    vitalsigns_data[date][email] = people
            except:
                continue

        elif 'August' in csv[i][0]:
            try:
                steps = df0.loc[df0['date'] == date]['steps'].item()
                distance = df0.loc[df0['date'] == date]['distance'].item()
                # print(f"distance: {distance}")
                runDistance = df0.loc[df0['date'] == date]['runDistance'].item()
                # print(f"runDistance: {runDistance}")
                calories = df0.loc[df0['date'] == date]['calories'].item()
                # print(f"calories: {calories}")
                heartbeat = math.ceil(df1.loc[df1['date'] == date]['heartRate'].mean())
                # print(f"heartbeat: {heartbeat}")
                # print(f"stop: {df2.loc[date]['stop']}")
                # print(f"start: {df2.loc[date]['start']}")
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
                # print(f'email: {email}')
                # print(f'people: {people}')
                if vitalsigns_data.get(date) == None:
                    vitalsigns_data[date] = {}
                    vitalsigns_data[date][email] = people
                else:
                    # print(f'Here')
                    # print(f'Email: {email}')
                    vitalsigns_data[date][email] = people
            except:
                continue


total_steps = np.sum(steps_arr)
total_distance = np.sum(distance_arr)
total_runDistance = np.sum(runDistance_arr)
total_calories = np.sum(calories_arr)
total_heartbeat = np.sum(heartbeat_arr)
total_sleep = np.sum(sleep_arr)
average_sleep = np.mean(sleep_arr)

for date in date_list:
    for email in all_email_list:
        try:
            vitalsigns_data[date][email]['steps'] = round(vitalsigns_data[date][email]['steps']/total_steps,3)
            vitalsigns_data[date][email]['distance'] = round(vitalsigns_data[date][email]['distance']/total_distance,3)
            vitalsigns_data[date][email]['runDistance'] = round(vitalsigns_data[date][email]['runDistance']/total_runDistance,3)
            vitalsigns_data[date][email]['calories'] = round(vitalsigns_data[date][email]['calories']/total_calories,3)
            vitalsigns_data[date][email]['heartbeat'] = round(vitalsigns_data[date][email]['heartbeat']/total_heartbeat,3)
            if vitalsigns_data[date][email]['sleep'] != 0.0:
                vitalsigns_data[date][email]['sleep'] = round(vitalsigns_data[date][email]['sleep']/24,3)
            else:
                vitalsigns_data[date][email]['sleep'] = round(average_sleep,3)
            
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

# print(useable_data)

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
    # print(f'len:{len(useable_data[date].keys())}')
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
# print(total_nodes)
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


dgl.seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0.5, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)
       
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1.0,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()
# criterion = FocalLoss(2.0,  0.25)

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
    labels_shape = labels.shape
    epoch = 10

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
        print(f'CE loss: {loss}')
        number=0
        loss_2=0
        alpha=0.25
        gamma=2
        for row in range(labels_shape[0]):
            for col in range(labels_shape[1]):
                number += 1
                if labels[row][col]==0.0:
                    # pt = 1-final_emb[row][col]
                    # print(f"PT: {pt}")
                    # alpha_t = 1-alpha
                    l = -math.log(1-final_emb[row][col])
                else:
                    # pt = 1
                    # alpha_t = alpha
                    l = -math.log(final_emb[row][col])
                
                # focal_loss = (-1*alpha_t)*((1-pt)**gamma)*(math.log(pt))
                # loss += focal_loss
                loss_2 += l
        # loss /= number
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
    
    loss_arr= [loss.detach().numpy() for loss in loss_arr]
    plt.plot(np.arange(0,epoch),loss_arr)
    plt.show()

def test(g, model):
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

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
# print(f"Total nodes: {total_nodes}")
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

    
train_graph.ndata['h'] = torch.from_numpy(features).float()

train_graph = dgl.add_self_loop(train_graph)

# train_graph.ndata['labels'] = torch.tensor(node_labels)
train_graph.ndata['labels'] = torch.from_numpy(node_labels).float()

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
# test_graph = dgl.graph(([0,1,0,3,1,2], [1,0,3,0,2,1]), num_nodes=5)
test_graph= dgl.graph(([0,1,0,3,1,2,0,5,0,7,0,8,1,5,1,6,1,7,1,9,2,6,2,7,2,8,2,9,3,5,3,7,3,8,3,9], [1,0,3,0,2,1,5,0,7,0,8,0,5,1,6,1,7,1,9,1,6,2,7,2,8,2,9,2,5,3,7,3,8,3,9,3]), num_nodes=10)


# step | distance | run distance | calories | heartbeat | sleep | sp02
# test_graph.ndata['h'] = torch.tensor([
# [0.087,0.093,0.073,0.121,0.202,0.228,0.970],
# [0.018,0.017,0.057,0.052,0.188,0.049,1.000],
# [0.080,0.081,0.198,0.113,0.179,0.345,0.960],
# [0.123,0.113,0.402,0.143,0.213,0.416,0.920],
# [0.693,0.695,0.271,0.571,0.218,0.259,0.950]
# ])
test_graph.ndata['h'] = torch.tensor([
[0.087,0.093,0.073,0.121,0.202,0.228,0.970],
[0.018,0.017,0.057,0.052,0.188,0.049,1.000],
[0.080,0.081,0.198,0.113,0.179,0.345,0.960],
[0.123,0.113,0.402,0.143,0.213,0.416,0.920],
[0.693,0.695,0.271,0.571,0.218,0.259,0.950],
[0.400,0.397,0.288,0.302,0.208,0.147,0.950],
[0.181,0.193,0.111,0.224,0.203,0.221,0.950],
[0.113,0.094,0.266,0.133,0.201,0.197,0.970],
[0.109,0.120,0.089,0.137,0.174,0.278,0.960],
[0.198,0.196,0.245,0.205,0.213,0.211,0.950],
])

test_graph = dgl.add_self_loop(test_graph)

# nausea | dizziness | headache | cough | weakness | feeling very tired | fever
# test_graph.ndata['labels'] = torch.tensor([
# [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
# [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
# [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
# [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# ])
test_graph.ndata['labels'] = torch.tensor([
[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
])

test(test_graph,model)
print('=================================================')






