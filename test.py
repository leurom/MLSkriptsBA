import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
#from dbfiller import DBManager
#from dbfiller import create_connection
from datareader import DataManager

print("--------------------------------")
print("Import packages successfull")
print("--------------------------------")

# database = r"Database/Model_Results.sqlite"
# create a database connection
# conn = create_connection(database)

#Data preprocessing
dataManager = DataManager()
data = dataManager.readData()
print(data)
print(data.dtypes)
# print(data.keys())
drop = ['Datum / Zeit','Umstellung Verbr.']
#y = data['Umstellung Verbr.']
#x = data.drop(drop, axis=1)  # 'axis' sollte 1 sein, um Spalten zu löschen
#exploreData = dataManager.exploreData(data)
#moreData = dataManager.generateDataSmote()

labelcentroid = dataManager.clusterData(data)
labels = labelcentroid[0]
centroids = labelcentroid[1]
print(labelcentroid[0])
analyzeCluster = dataManager.analyzeCluster(labels, centroids, data)
#rf = dataManager.randomForest(labelcentroid[2])

#logreg = dataManager.logRegression(data)
neuralnetwork = dataManager.neuralNetwork(labelcentroid[2])
'''
#Modeling
baseModel = LinearRegression()
baseModel.fit(x,y)
rsquare = baseModel.score(x,y)

#The value of 𝑏₀ describes when 𝑥 is zero
intercept = baseModel.intercept_

#The value 𝑏₁ means that the predicted response rises by 0.54 when 𝑥 is increased by one
slope = baseModel.coef_

print(f"coefficient of determinant: {rsquare}")
print(f"intercept(𝑏₀): {intercept}")
print(f"slope(𝑏₁): {slope}")

#Prediction
y_pred = baseModel.predict(x)
# print(f"predicted response: {y_pred}")

#---------------------------------------------
##Advanced Linear Regression with statsmodels
x = sm.add_constant(x)
Statsmodel = sm.OLS(y,x)
results = Statsmodel.fit()
print(results.summary())
print(f"coefficient of determination: {results.rsquared}")
# print(f"predicted response:\n{results.fittedvalues}")
# print(f"predicted response:\n{results.predict(x)}")

#dbManager = DBManager()
#dbManager.Insert(conn, results.rsquared)
'''


print("--------------------------------")
print("End")
print("--------------------------------")