# Ex-07-Feature-Selection

## AIM

To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation

Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM

### STEP 1

Read the given Data

### STEP 2

Clean the Data Set using Data Cleaning Process

### STEP 3

Apply Feature selection techniques to all the features of the data set

### STEP 4

Save the data to the file


# CODE

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv('/content/titanic_dataset.csv')

df.head()

df.isnull().sum()

df.drop('Cabin',axis=1,inplace=True)

df.drop('Name',axis=1,inplace=True)

df.drop('Ticket',axis=1,inplace=True)

df.drop('PassengerId',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)

df

df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()

plt.title("Dataset with outliers")

df.boxplot()

plt.show()

cols = ['Age','SibSp','Fare']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

from sklearn.preprocessing import OrdinalEncoder

climate = ['C','S','Q']

en= OrdinalEncoder(categories = [climate])

df['Embarked']=en.fit_transform(df[["Embarked"]])

df

climate = ['male','female']

en= OrdinalEncoder(categories = [climate])

df['Sex']=en.fit_transform(df[["Sex"]])

df

from sklearn.preprocessing import RobustScaler

sc=RobustScaler()

df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])

df

import statsmodels.api as sm

import numpy as np

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()

df1["Survived"]=np.sqrt(df["Survived"])

df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])

df1["Sex"]=np.sqrt(df["Sex"])

df1["Age"]=df["Age"]

df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])

df1["Fare"],parameters=stats.yeojohnson(df["Fare"])

df1["Embarked"]=df["Embarked"]

df1.skew()

import matplotlib

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1)

y = df1["Survived"]

plt.figure(figsize=(12,10))

cor = df1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)

plt.show()

cor_target = abs(cor["Survived"])

relevant_features = cor_target[cor_target>0.5]

relevant_features

X_1 = sm.add_constant(X)

model = sm.OLS(y,X_1).fit()

model.pvalues

cols = list(X.columns)

pmax = 1

while (len(cols)>0):

p= []

X_1 = X[cols]

X_1 = sm.add_constant(X_1)

model = sm.OLS(y,X_1).fit()

p = pd.Series(model.pvalues.values[1:],index = cols)  

pmax = max(p)

feature_with_p_max = p.idxmax()

if(pmax>0.05):

    cols.remove(feature_with_p_max)
    
else:

    break
    selected_features_BE = cols

print(selected_features_BE)

model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)

nof_list=np.arange(1,6)

high_score=0

nof=0

score_list =[]

for n in range(len(nof_list)):

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

model = LinearRegression()

rfe = RFE(model,step=nof_list[n])

X_train_rfe = rfe.fit_transform(X_train,y_train)

X_test_rfe = rfe.transform(X_test)

model.fit(X_train_rfe,y_train)

score = model.score(X_test_rfe,y_test)

score_list.append(score)

if(score>high_score):

    high_score = score
    
    nof = nof_list[n]
print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)

model = LinearRegression()

rfe = RFE(model, step=2)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")

plt.show()

# OUPUT


![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/4cd847d5-ada2-4a9b-ad6d-7d4e2f347fbe)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/ac3cb49f-c230-42ad-8d01-f83078f9f568)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/f1780901-82b8-46f7-8cde-a7cd1d04a6e8)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/93746ed2-4371-4a1f-a8af-9338b2c3b4e4)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/0c101e2f-74d1-4f90-b8af-82329227a381)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/d650ef8e-3e6a-42c1-b5cf-44bca72c64ef)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/c35ca458-9138-4d42-80e3-922a1813c656)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/46a8a516-ca98-4330-b88f-6c4c799a797a)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/9a453ddc-3c6f-4732-8517-5bfc90a6154a)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/ca89050a-da13-4071-a594-0f4a814006b9)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/ed1065fe-68cc-4281-9476-498078f54340)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/d9498228-27be-4b3e-a248-b578d5e434a7)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/350fcb90-61ac-4407-af8d-16d584fd5432)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/dd5c592f-4a05-4fd6-a420-30fda775cf3a)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/2268f5b0-2c6d-4397-8055-a615e8836de2)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/d028cf0b-e024-4a7b-ad41-78f856a5a971)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/7e1d1b66-e8b5-47fd-9647-5778910e4232)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/91bd4a53-e630-4a3e-82cc-7c31478333d5)



![image](https://github.com/nivetharajaa/Ex-07-Feature-Selection/assets/120543388/f722644e-d171-4885-ba93-ba0190a11c27)


# RESULT

The various feature selection techniques are performed on a dataset and saved the data to a file.



















