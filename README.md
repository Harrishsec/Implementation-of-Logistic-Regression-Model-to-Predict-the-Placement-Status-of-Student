# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook / Google Colab

## Algorithm
1. Load dataset (student features + placement status).
2. Preprocess data (handle missing values, encode, normalize).
3. Split data into training and testing sets.
4. Initialize weights and bias.
5. Compute prediction using sigmoid:
   $h(x) = \frac{1}{1+e^{-(wx+b)}}$
6. Calculate cost using cross-entropy loss.
7. Update weights using gradient descent.
8. Repeat until convergence.
9. Predict placement status (≥0.5 → Placed, else Not Placed).
10. Evaluate model (accuracy, precision, recall, F1-score).


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HARRISH P
RegisterNumber: 212224230088
*/
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"], axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1 ["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression (solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
 
 
<img width="1247" height="231" alt="image" src="https://github.com/user-attachments/assets/0289e1ba-867d-4285-b8f6-c40cab5ebab4" />


<img width="1105" height="221" alt="image" src="https://github.com/user-attachments/assets/c0421ea9-d004-409f-8aab-a332501f30c1" />


<img width="222" height="552" alt="image" src="https://github.com/user-attachments/assets/ab01eac5-032b-44cc-a9e5-755bb6543ac9" />


<img width="132" height="52" alt="image" src="https://github.com/user-attachments/assets/352d92ee-c8fc-4c1e-8f62-5748a6655f9e" />


<img width="1017" height="473" alt="image" src="https://github.com/user-attachments/assets/37042a2d-0153-42d3-ba87-cb1ab10864fb" />


<img width="970" height="467" alt="image" src="https://github.com/user-attachments/assets/8d33b057-ec78-4072-8baf-2f8e8136220f" />


<img width="188" height="505" alt="image" src="https://github.com/user-attachments/assets/fd8b755c-8e25-45a4-8197-7de52c645356" />

## Logistic Regression:

<img width="662" height="67" alt="image" src="https://github.com/user-attachments/assets/49d427f0-b041-436c-8809-9d8ee74c8c2d" />

## ACCURACY VALUE:
<img width="192" height="35" alt="image" src="https://github.com/user-attachments/assets/d498d2a1-5d79-4127-bfb7-cd8d01339f5b" />

## CONFUSION ARRAY:
<img width="213" height="72" alt="image" src="https://github.com/user-attachments/assets/a330b710-05bb-4ec2-a7f6-7edf87adc44d" />

## CLASSFICATION REPORT:
<img width="537" height="192" alt="image" src="https://github.com/user-attachments/assets/b0a67a77-ca36-4ab2-a8c6-8fa9bab130b1" />

## PREDICTION:
<img width="142" height="27" alt="image" src="https://github.com/user-attachments/assets/d3554fd3-b849-47d1-bbb2-afc51e9e3f79" />

 
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
