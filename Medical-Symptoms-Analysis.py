import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as stat
import seaborn as sns

dataset=pd.read_csv("C:/Users/uufci/Downloads/Disease_symptom_and_patient_profile_dataset.csv")

#detect missing values in dataset#
print(dataset.isnull().sum())

#drop rows with missing values from features Smoking,chest pains#
dataset.dropna(subset=['Smoking'], inplace=True)
dataset.dropna(subset=['Chest Pains'], inplace=True)
dataset.dropna(subset=['Fever'], inplace=True)
dataset.dropna(subset=['Pregnancy'], inplace=True)
dataset.dropna(subset=['Auto-Immune Disease'], inplace=True)
dataset.dropna(subset=['Blood Pressure'], inplace=True)
dataset.dropna(subset=['Exercise Ratio'], inplace=True)

#Fill in the missing value for the 'Weight_Attribute' and the 'Height_Attribute' with the mean value#
MeanW = dataset['Weight'].mean()
print("Weight Mean is   " , MeanW)
dataset['Weight'].fillna(MeanW,inplace=True)

MeanH = dataset['Height'].mean()
print("Height Mean is   " , MeanH)
dataset['Height'].fillna(MeanH,inplace=True)

#Calculate the BMI formula for the missing weight/height#  
def BMI_Calc(height, weight):
    return (weight / (height*height))
dataset['BMI'].fillna(BMI_Calc(dataset['Height'],dataset['Weight']), inplace=True)

#check for duplicated records#
duplicated_rows = dataset[dataset.duplicated()]
print("dupliacted rows are:" , duplicated_rows)

print(dataset.info())

#Data Visualisation#
#Correlation Matrix#
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

#discovering the correlation between features Weight and BMI#
fig=plt.figure(figsize=(10,10),dpi=100,facecolor='lightblue',clear=True)
x=dataset['Weight']
y=dataset['BMI']
plt.xlabel('Weight(KG)')
plt.ylabel('BMI')
plt.scatter(x,y)
plt.show()

#A Bar figure to identify the BMI Categories#
BMI_CAT = ['Under_Weight','Normal','Overweight','Obese']

Count_UW= len(dataset[dataset['BMI']<18])
Count_NW= len(dataset[(dataset['BMI']>=18) & (dataset['BMI']<=25)])
Count_OW=len(dataset[(dataset['BMI']>25)  & (dataset['BMI']<=30)])
Count_OB=len(dataset[dataset['BMI']>30])

Count_P = [Count_UW,Count_NW,Count_OW,Count_OB]

plt.bar(BMI_CAT,Count_P,label='BMI Distribution')
plt.xlabel('BMI Categories')
plt.ylabel('Number of People')
plt.show()


#A Bar Figure to identify how many men and women are in each category 0,1,2,3#

Count_M_1 =dataset[(dataset['Gender']==0)& (dataset['Classification']==0)]
Count_M_2 =dataset[(dataset['Gender']==0)& (dataset['Classification']==1)]
Count_M_3 =dataset[(dataset['Gender']==0)& (dataset['Classification']==2)]
Count_M_4 =dataset[(dataset['Gender']==0)& (dataset['Classification']==3)]

Count_F_1 =dataset[(dataset['Gender']==0)& (dataset['Classification']==0)]
Count_F_2 =dataset[(dataset['Gender']==0)& (dataset['Classification']==1)]
Count_F_3 =dataset[(dataset['Gender']==0)& (dataset['Classification']==2)]
Count_F_4 =dataset[(dataset['Gender']==0)& (dataset['Classification']==3)]
    
Categories =['Category1','Category2','Category3','Category4']

males=[len(Count_M_1),len(Count_M_2),len(Count_M_3),len(Count_M_4)]
females=[len(Count_F_1),len(Count_F_2),len(Count_F_3),len(Count_F_4)]

x=np.arange(len(Categories))

plt.bar(x,males,label='Male')
plt.bar(x,females,bottom=males,label='Female')

plt.legend()
plt.show()

#Apply KNN Algorithm#
Y_KNN = dataset.iloc[:,16].values
X_KNN=np.array(dataset.iloc[:,:-1].values)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_KNN,Y_KNN,test_size=0.2)

from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()
Scaler.fit(x_train)
xtest=Scaler.transform(x_test)
xtrain = Scaler.transform(x_train)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
KNeighborsClassifier()
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)

print(y_pred)

#Calculate Accuracy using the confusion matrix#
from sklearn.metrics import classification_report, confusion_matrix
CM=confusion_matrix(y_test, y_pred)
print("KNN Evaluation:","\n","The Confusion Matrix is:","\n", CM)
print("The Classification Report is as the following:","\n",classification_report(y_test, y_pred))
sns.heatmap(CM)
plt.show()

#Apply Random Forest#

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()
Scaler.fit(x_train)
xtest=Scaler.transform(x_test)
xtrain = Scaler.transform(x_train)

rf_model.fit(x_train, y_train)
y_predr = rf_model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_predr)
print("Random Forest Evaluation","\n", f'Accuracy: {accuracy:.2f}')


#Calculate Accuracy using the confusion matrix#
from sklearn.metrics import classification_report, confusion_matrix
CM=confusion_matrix(y_test, y_predr)
print("The Confusion Matrix is:","\n", CM)
print("The Classification Report is as the following:","\n",classification_report(y_test, y_predr))
sns.heatmap(CM)
plt.show()







