import numpy as np
import pandas as pd
#import random forest classifier from sklearn
from sklearn.ensemble import RandomForestClassifier

#download the titanic dataset here (https://www.kaggle.com/c/titanic/data)
titanic = pd.read_csv("C:\\Users\\Ankush Raut\\Downloads\\Titanic data\\train.csv")

#drop whatever u don't need
titanic = titanic.drop(labels=['PassengerId', 'Cabin', 'Ticket', 'Embarked'], axis=1)
titanic = titanic.drop(labels=['Name'], axis=1)

#fill null values
titanic['Age'] = titanic['Age'].fillna(method='pad')

#feature scaling
titanic.Age = ((titanic.Age - np.mean(titanic.Age))/np.std(titanic.Age))
titanic.Fare = ((titanic.Fare - np.mean(titanic.Fare))/np.std(titanic.Fare))
titanic.Parch = ((titanic.Parch - np.mean(titanic.Parch))/np.std(titanic.Parch))
titanic.SibSp = ((titanic.SibSp - np.mean(titanic.SibSp))/np.std(titanic.SibSp))

ordered_sex = ['male', 'female']
titanic.Sex = titanic.Sex.astype("category", ordered=True, categories=ordered_sex).cat.codes

#seperate dependent and independent features
y = titanic['Survived'].copy()
titanic.drop(labels=['Survived'], inplace=True, axis=1)

#create the model
clf = RandomForestClassifier(n_estimators = 100, min_samples_split = 50)
model1 = clf.fit(titanic, y)

#check the cross validation score
from sklearn.cross_validation import cross_val_score

score = cross_val_score(model1, titanic, y, cv = 10)
from scipy.stats import sem
def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))

#Testing

test = pd.read_csv("C:\\Users\\Ankush Raut\\Downloads\\Titanic data\\test.csv")

#drop whatever u don't need
test = test.drop(labels=['PassengerId', 'Cabin', 'Ticket', 'Embarked'], axis=1)
test = test.drop(labels=['Name'], axis=1)

#fill null values
test['Age'] = test['Age'].fillna(method='pad')
test['Fare'] = test['Fare'].fillna(method='pad')


#feature scaling
test.Age = ((test.Age - np.mean(test.Age))/np.std(test.Age))
test.Fare = ((test.Fare - np.mean(test.Fare))/np.std(test.Fare))
test.Parch = ((test.Parch - np.mean(test.Parch))/np.std(test.Parch))
test.SibSp = ((test.SibSp - np.mean(test.SibSp))/np.std(test.SibSp))

ordered_sex = ['male', 'female']
test.Sex = test.Sex.astype("category", ordered=True, categories=ordered_sex).cat.codes

#predict
y_pred = model1.predict(test)

#store in a dataframe
ID = pd.Series(range(892, 1310))

y1 = np.array(y_pred)

sol = pd.DataFrame({'PassengerId':ID, 'Survived':y1})
sol.to_csv('Solution1.csv')

#the accuracy is found to be 0.77512
