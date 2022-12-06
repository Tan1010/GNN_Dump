import pandas as pd
import math

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

date = ['1306',
    '1506',
    '1706',
    '2006',
    '2206',
    '2506',
    '2706',
    '2906']

date_list = ['2022-06-13',
    '2022-06-15',
    '2022-06-17',
    '2022-06-20',
    '2022-06-22',
    '2022-06-25',
    '2022-06-27',
    '2022-06-29']
    
excel = ['Daily Symptoms Questionnaire (13_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (15_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (17_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (20_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (22_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (24_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (27_6) (Responses).xlsx',
'Daily Symptoms Questionnaire (29_6) (Responses).xlsx',
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
            'spo2': spo2,
            'symptoms': symptoms
        }

    questionnaire_data[date_list[index]] = people

total_nodes = 0
symptoms_list = []
for date in date_list:
    emails = questionnaire_data[date].keys()
    total_nodes += (len(questionnaire_data[date].keys()))
    for email in emails:
        email_list.append(email)
        try:
            sym = questionnaire_data[date][email]['symptoms']
            symptoms_list.append(sym)
        except:
            continue

list1=[]
list2=[]

for i in range(len(symptoms_list)):
    for j in range(i+1,len(symptoms_list)):
        for k in range(len(symptoms_list[i])):
            if symptoms_list[i][k] == '1' and symptoms_list[j][k] == '1':
                list1.append(i)
                list1.append(j)
                list2.append(j)
                list2.append(i)
                break
            

# print(list1)
# print()
# print(list2)
# print(f"total_nodes: {total_nodes}")
print(questionnaire_data)
questionnaire_data.pop('2022-06-13')
print()
print(questionnaire_data)

print(list1)
print(list2)