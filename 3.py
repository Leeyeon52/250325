import numpy as np
import pandas as pd
import matplotlib.pyplot as pit
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')


train_data = train_data.drop(columns=['Ticket', 'Fare', 'Cabin', 'Embarked'])
test_data = test_data.drop(columns=['Ticket', 'Fare', 'Cabin', 'Embarked'])
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())  
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())  

train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})
test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})

sex_survival = train_data.groupby('Sex')['Survived'].value_counts().unstack().fillna(0)
print("Sex-wise Survival/Death Count:")
print(sex_survival)


bins = [0, 18, 30, 40, 50, 60, 100] 
labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61+']  
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=bins, labels=labels, right=False)

age_survival = train_data.groupby('AgeGroup')['Survived'].value_counts().unstack().fillna(0)
print("\nAge-wise Survival/Death Count:")
print(age_survival)


sibsp_survival = train_data.groupby('SibSp')['Survived'].value_counts().unstack().fillna(0)
print("\nSibSp-wise Survival/Death Count:")
print(sibsp_survival)

parch_survival = train_data.groupby('Parch')['Survived'].value_counts().unstack().fillna(0)
print("\nParch-wise Survival/Death Count:")
print(parch_survival)


pclass_survival = train_data.groupby('Pclass')['Survived'].value_counts().unstack().fillna(0)
print("\nPclass-wise Survival/Death Count:")
print(pclass_survival)


X_train = train_data[['Sex', 'Age', 'SibSp', 'Parch', 'Pclass']]
y_train = train_data['Survived']
X_test_data = test_data[['Sex', 'Age', 'SibSp', 'Parch', 'Pclass']]

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200, solver='liblinear')  
model.fit(X_train_split, y_train_split)


y_pred_val = model.predict(X_val_split)
accuracy = accuracy_score(y_val_split, y_pred_val)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")



sex_survival.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], figsize=(8, 6))
pit.title('Survival/Death Count by Sex')
pit.xlabel('Sex')
pit.ylabel('Count')
pit.xticks([0, 1], ['Female', 'Male'], rotation=0)
pit.legend(['Died', 'Survived'], loc='upper right')
pit.show()

age_survival.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], figsize=(8, 6))
pit.title('Survival/Death Count by Age Group')
pit.xlabel('Age Group')
pit.ylabel('Count')
pit.legend(['Died', 'Survived'], loc='upper right')
pit.show()

sibsp_survival.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], figsize=(8, 6))
pit.title('Survival/Death Count by SibSp')
pit.xlabel('SibSp')
pit.ylabel('Count')
pit.legend(['Died', 'Survived'], loc='upper right')
pit.show()

parch_survival.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], figsize=(8, 6))
pit.title('Survival/Death Count by Parch')
pit.xlabel('Parch')
pit.ylabel('Count')
pit.legend(['Died', 'Survived'], loc='upper right')
pit.show()

pclass_survival.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], figsize=(8, 6))
pit.title('Survival/Death Count by Pclass')
pit.xlabel('Pclass')
pit.ylabel('Count')
pit.legend(['Died', 'Survived'], loc='upper right')
pit.show()


y_pred_test = model.predict(X_test_data)


submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'], 
    'Survived': y_pred_test  
})


submission.to_csv('submission.csv', index=False)

print("Submission file has been saved as 'submission.csv'.")
