import pandas as pd
import math
import numpy as np

date_list = ['2022-06-13',
    '2022-06-15',
    '2022-06-17',
    '2022-06-20',
    '2022-06-22',
    '2022-06-25',
    '2022-06-27',
    '2022-06-29']

email_list = ['yizentan@gmail.com',
'afrinaad@gmail.com',
'darrensow.01@gmail.com',
'drsawband@gmail.com',
'hong_chiow0316@yahoo.com',
'elainemap16@gmail.com',
'hamizahhelmie@gmail.com',]

csv = [
    ['../../Data/Tan Yi Zen/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659266515815.csv', '../../Data/Tan Yi Zen/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659266515958.csv', '../../Data/Tan Yi Zen/1 Jun - 30 Jun/SLEEP/SLEEP_1659266515875.csv'],
    ['../../Data/Elaine/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659340973129.csv', '../../Data/Elaine/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659340973647.csv', '../../Data/Elaine/1 Jun - 30 Jun/SLEEP/SLEEP_1659340973376.csv'],
    ['../../Data/Chiow/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659593423752.csv', '../../Data/Chiow/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659593424267.csv', '../../Data/Chiow/1 Jun - 30 Jun/SLEEP/SLEEP_1659593423989.csv'],
    ['../../Data/Darren/1 Jun - 30 Jun/ACTIVITY/ACTIVITY_1659592313228.csv', '../../Data/Darren/1 Jun - 30 Jun/HEARTRATE_AUTO/HEARTRATE_AUTO_1659592313371.csv', '../../Data/Darren/1 Jun - 30 Jun/SLEEP/SLEEP_1659592313301.csv'],
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
    df2 = pd.read_csv(csv[i][2])
    for date in date_list:
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

total_steps = np.sum(steps_arr)
total_distance = np.sum(distance_arr)
total_runDistance = np.sum(runDistance_arr)
total_calories = np.sum(calories_arr)
total_heartbeat = np.sum(heartbeat_arr)
total_sleep = np.sum(sleep_arr)

for date in date_list:
    for email in email_list:
        try:
            vitalsigns_data[date][email]['steps'] = round(vitalsigns_data[date][email]['steps']/total_steps,3)
            vitalsigns_data[date][email]['distance'] = round(vitalsigns_data[date][email]['distance']/total_distance,3)
            vitalsigns_data[date][email]['runDistance'] = round(vitalsigns_data[date][email]['runDistance']/total_runDistance,3)
            vitalsigns_data[date][email]['calories'] = round(vitalsigns_data[date][email]['calories']/total_calories,3)
            vitalsigns_data[date][email]['heartbeat'] = round(vitalsigns_data[date][email]['heartbeat']/total_heartbeat,3)
            vitalsigns_data[date][email]['sleep'] = round(vitalsigns_data[date][email]['sleep']/24,3)
            
        except:
            continue

print(vitalsigns_data)
