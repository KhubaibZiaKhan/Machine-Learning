import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd .read_csv('petrol_consumption.csv')

dataset.head()

dataset.describe()

x = dataset[['Petrol_Tax', 'Average_Income', 'Paved_Highways',
             'Population_Driver_Licence(%)']]
y = dataset['Petrol_Consumtion']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor . fit(x_train, y_train)

coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns=['Coeffecient'])
coeff_df

y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))