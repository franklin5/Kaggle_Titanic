import pandas as pd
import numpy as np

ti = pd.read_csv("train.csv")
ti["Age"] = ti["Age"].fillna(ti["Age"].median())
ti.loc[ti["Sex"]=="male","Sex"] = 0
ti.loc[ti["Sex"]=="female","Sex"] = 1
ti["Embarked"] = ti["Embarked"].fillna("S")
ti.loc[ti["Embarked"]=="S","Embarked"]=0
ti.loc[ti["Embarked"]=="C","Embarked"]=1
ti.loc[ti["Embarked"]=="Q","Embarked"]=2

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
pred = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
algo=LogisticRegression(C=100)
X_train,X_test,y_train,y_test = train_test_split(ti[pred],ti["Survived"],test_size=0.3,random_state=1)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
algo.fit(X_train_std,y_train)
X_test_std = scaler.transform(X_test)
'''
predictions=algo.predict(X_test_std)
accuracy = 1-float(abs(predictions-y_test.values).sum())/len(predictions)
print(accuracy)
''' #equivalent to the following:
print algo.score(X_test_std,y_test)

ti_test = pd.read_csv("test.csv")
ti_test["Age"] = ti_test["Age"].fillna(ti["Age"].median())
ti_test.loc[ti_test["Sex"]=="male","Sex"] = 0
ti_test.loc[ti_test["Sex"]=="female","Sex"] = 1
ti_test["Embarked"] = ti_test["Embarked"].fillna("S")
ti_test.loc[ti_test["Embarked"]=="S","Embarked"]=0
ti_test.loc[ti_test["Embarked"]=="C","Embarked"]=1
ti_test.loc[ti_test["Embarked"]=="Q","Embarked"]=2
ti_test["Fare"] = ti_test["Fare"].fillna(ti_test["Fare"].median())
ti_test_std = scaler.transform(ti_test[pred])
submission = algo.predict(ti_test_std)
sub = pd.DataFrame({"PassengerId": ti_test["PassengerId"], "Survived": submission})
#sub["Survived"] = sub["Survived"].astype(int)
#print sub.dtypes
sub.to_csv("kaggle_test_titanic.csv",index=False)
