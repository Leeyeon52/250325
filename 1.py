import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as pit
import os
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')
train_data.info()
print(train_data.isnull().sum())
train_data.head()

data = {
    'PassengerId': [1, 2, 3],
    'Pclass': [3, 1, 1],
    'Name': ['A', 'B', 'C'],
    'Sex': ['male', 'female', 'female'],
    'Age': [22, 38, 26],
    'SibSp': [1, 1, 0],
    'Parch': [0, 0, 0],
    'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
    'Fare': [7.25, 71.2833, 7.925],
    'Cabin': ['C85', 'C123', 'E46'],
    'Embarked': ['S', 'C', 'Q'],
    'Survived': [0, 1, 1]
}
df = pd.DataFrame(data)

df = df.drop(columns=[ 'Ticket', 'Fare', 'Cabin', 'Embarked','Pclass'])

X = df[['Sex', 'Age', 'SibSp', 'Parch']]
y = df['Survived']

print(df)

data = {'Sex': ['male', 'female', 'female', 'male', 'male', 'female', 'female'],
        'Survived': [1, 1, 1, 0, 0, 1, 0]}

print(df.describe())

survial_rate = df['Survived'].mean()
print(f"Survival Rate: {survial_rate * 100:.2f}%")

data = {
    'Sex': ['male', 'female', 'female', 'male', 'male', 'female', 'female'],
    'Survived': [1, 1, 1, 0, 0, 1, 0]
}
df = pd.DataFrame(data)
sex_survival = df.groupby('Sex')['Survived'].mean()
print(sex_survival)

data = {
    'SibSp': [1, 0, 2, 1, 0, 3, 1],
    'Parch': [0, 1, 0, 1, 0, 0, 1],
    'Survived': [1, 0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)
family_survival = df.groupby(['SibSp', 'Parch'])['Survived'].mean()
print(family_survival)

data = {
    'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625],
    'Survived': [0, 1, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

bins = [0, 10, 50, 100]
labels = ['0-10', '10-50', '50+']
df['Fare_binned'] = pd.cut(df['Fare'], bins=bins, labels=labels, right=False)

fare_survival = df.groupby('Fare_binned')['Survived'].mean()


print(fare_survival)

survival_rate = df['Survived'].mean()

# 그래프 그리기
pit.figure(figsize=(6, 4))
pit.bar(['Survived','Died'], [survival_rate, 1 - survival_rate], color=['green', 'skyblue'])
pit.title('Survival Rate')
pit.ylabel('Percentage')
pit.show()

labels = ['Survived', 'Died']
sizes = [35, 65]
pit.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)


data = {
    'Title': ['Master', 'Mr', 'Miss', 'Mrs', 'Dr', 'Master', 'Ms', 'Mr'],
    'Survived': [1, 0, 1, 1, 0, 1, 1, 0]
}


df = pd.DataFrame(data)
title_to_gender = {
    'Mr': 'Male', 'Master': 'Male', 'Dr': 'Male', 'Rev': 'Male', 'Col': 'Male', 'Major': 'Male',
    'Miss': 'Female', 'Ms': 'Female', 'Mrs': 'Female', 'Mme': 'Female', 'Mlle': 'Female'
}

df['Sex'] = df['Title'].map(title_to_gender)

male_survival_rate = df[df['Sex'] == 'Male']['Survived'].mean()
female_survival_rate = df[df['Sex'] == 'Female']['Survived'].mean()

print(f"Male Survival Rate: {male_survival_rate:.2f}")
print(f"Female Survival Rate: {female_survival_rate:.2f}")

data = {'Sex':['male','female','female','male','male','female','female'],"Survived":[1,1,1,0,0,1,0]}

df = pd.DataFrame(data)

print(df.columns)

male_survived = len(df[(df['Sex'] == 'male') & (df['Survived'] == 1)])
male_died = len(df[(df['Sex'] == 'male') & (df['Survived'] == 0)])

female_survived = len(df[(df['Sex'] == 'female') & (df['Survived'] == 1)])
female_died = len(df[(df['Sex'] == 'female') & (df['Survived'] == 0)])

print(f"Male Survived: {male_survived}, Male Died: {male_died}")
print(f"Female Survived: {female_survived}, Female Died: {female_died}")
print(f"Male Survival Rate: {male_survival_rate:.2f}")
print(f"Female Survival Rate: {female_survival_rate:.2f}")


labels = ['Male Survived', 'Male Died', 'Female Survived', 'Female Died']
sizes = [male_survived, male_died, female_survived, female_died]

pit.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
pit.axis('equal')
pit.show()

data = {'Sex':['male','female','female','male','male','female','female'],
        "Survived":[1,1,1,0,0,1,0]}

df = pd.DataFrame(data)

print(df.columns)

male_survived = len(df[(df['Sex'] == 'male') & (df['Survived'] == 1)])
male_died = len(df[(df['Sex'] == 'male') & (df['Survived'] == 0)])

female_survived = len(df[(df['Sex'] == 'female') & (df['Survived'] == 1)])
female_died = len(df[(df['Sex'] == 'female') & (df['Survived'] == 0)])

labels = ['Survived', 'Died']
male_counts = [male_survived, male_died]
female_counts = [female_survived, female_died]

fig, ax = pit.subplots(figsize=(8, 6))

bar_width = 0.35  
index = range(len(labels))

ax.bar(index, male_counts, bar_width, label='Male', color='skyblue', alpha=0.7)

ax.bar([p + bar_width for p in index], female_counts, bar_width, label='Female', color='pink')

ax.set_xlabel('Survival Status')
ax.set_ylabel('Count')
ax.set_title('Survival and Death by Gender')
ax.set_xticks([p + bar_width / 2 for p in index])  
ax.set_xticklabels(labels)  
ax.legend()  


pit.show()

import pandas as pd


data = {
    'Age': [22, 25, 30, 35, 40, 50, 60, 15, 80, 28],
    'Survived': [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)


bins = [0, 18, 30, 40, 50, 60, 100] 
labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61+']  

df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)


age_group_survival_rate = df.groupby('AgeGroup')['Survived'].mean()


print(age_group_survival_rate)


data = {
    
    'Name': ['John', 'Jane', 'Mike', 'Anna', 'Paul'],
    'Sex': ['male', 'female', 'male', 'female', 'male'],
    'Age': [22, 25, 30, 35, 40],
    'SibSp': [1, 1, 0, 0, 1],
    'Parch': [0, 1, 0, 2, 0]
}

df = pd.DataFrame(data)

df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df = df.drop(columns=['Name'])

corr_df = df.corr()


pit.figure(figsize=(8, 6))
ax = sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
pit.title("Correlation Heatmap")
pit.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)