import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

print("--------------------------------")
print("Import packages successfull")
print("--------------------------------")
##Multiple Linear Regression
#Data preprocessing
""" x = [
   [0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]
]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x,y = np.array(x), np.array(y) """

data = pd.read_csv('Advertising.csv')
# print(data.keys())
drop = ['Unnamed: 0', 'sales', 'newspaper']
y = data['sales']
x = data.drop(drop, axis=1)  # 'axis' sollte 1 sein, um Spalten zu löschen

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