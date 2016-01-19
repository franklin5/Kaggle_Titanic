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
from sklearn.cross_validation import KFold
pred = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
algo=LogisticRegression()
kf = KFold(len(ti),n_folds=3,random_state=1)
predictions = []
for train, test in kf:
    train_pred = ti[pred].iloc[train,:]
    train_targ = ti["Survived"].iloc[train]
    algo.fit(train_pred,train_targ)
    test_pred = algo.predict(ti[pred].iloc[test,:])
    predictions.append(test_pred)

predictions = np.concatenate(predictions)
accuracy = 1-abs(predictions-ti["Survived"]).sum()/len(predictions)
print(accuracy)

ti_test = pd.read_csv("test.csv")
ti_test["Age"] = ti_test["Age"].fillna(ti["Age"].median())
ti_test.loc[ti_test["Sex"]=="male","Sex"] = 0
ti_test.loc[ti_test["Sex"]=="female","Sex"] = 1
ti_test["Embarked"] = ti_test["Embarked"].fillna("S")
ti_test.loc[ti_test["Embarked"]=="S","Embarked"]=0
ti_test.loc[ti_test["Embarked"]=="C","Embarked"]=1
ti_test.loc[ti_test["Embarked"]=="Q","Embarked"]=2
ti_test["Fare"] = ti_test["Fare"].fillna(ti_test["Fare"].median())
submission = algo.predict(ti_test[pred])
sub = pd.DataFrame({"PassengerId": ti_test["PassengerId"], "Survived": submission})
sub["Survived"] = sub["Survived"].astype(int)
print sub.dtypes
sub.to_csv("kaggle_test_titanic.csv",index=False)
