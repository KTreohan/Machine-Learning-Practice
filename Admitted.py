#Importing packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
 
#Building the Dataframe 
candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
 'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
 'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
 'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
 }
#Creating columns of dataframe
df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])
print (df)
 
#Create Logistic Regression 
X = df[['gmat', 'gpa','work_experience']] #Independent Variable - User Changes
y = df['admitted'] #Dependent Variable - What we're predicting
 
#Then, apply train_test_split. For example, you can set the test size to 0.25, and therefore the model testing will be based on 25% of the dataset, while the model training will be based on 75% of the dataset:
#Splitting the dataset - Training and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
 
#Applying Logistic Regression 
#Based off variables - will predict who will be admitted and who not
#0.05% significance tested against - If p value lower than 0.05 then Significant
#If significant they will be admitted
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train) 
y_pred=logistic_regression.predict(X_test)
 
#Creating a confusion matrix - used for valuation of results
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
 
#Printing Accuracy
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
