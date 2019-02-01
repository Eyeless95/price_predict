import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('kc_house_data.csv')

# data.head()

described_data = data.describe(include=[np.number])

# Values, that used in analys from input file
names = ['Price', 'Bedrooms', 'Bathrooms', 'Living Sqft', 'Lot Sqft', 'Floors', 'Waterfront', 'View', 'Condition',
         'Grade', 'Above Sqft', 'Basement Sqft', 'ZIPCode', 'Latitude', 'Altitude']
df = data[names]
correlations = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 15, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

data['Waterfront'] = data['Waterfront'].astype('category', ordered=True)
data['View'] = data['View'].astype('category', ordered=True)
data['Condition'] = data['Condition'].astype('category', ordered=True)
data['Grade'] = data['Grade'].astype('category', ordered=False)
data['ZIPCode'] = data['ZIPCode'].astype('category', ordered=False)

# Show plots for most interesting data
sns.regplot(x='Living Sqft', y='Price', data=data)
plt.show()

sns.regplot(x='Basement Sqft', y='Price', data=data)
plt.show()

sns.regplot(x='Above Sqft', y='Price', data=data)
plt.show()

sns.stripplot(x='Bedrooms', y='Price', data=data)
plt.show()

sns.stripplot(x='Bathrooms', y='Price', data=data, size=5)
plt.show()

sns.stripplot(x='Grade', y='Price', data=data, size=5)
plt.show()

# Skip houses with more than 10 bedrooms and more than 8 bathrooms
data = data[data['Bedrooms'] < 10]
data = data[data['Bathrooms'] < 8]
# print(data.head())

# 5 top parameters that have influence on price
top_params = ['Bedrooms', 'Bathrooms', 'Living Sqft', 'Above Sqft', 'Grade']
df = data[top_params]

df = pd.get_dummies(df, columns=['Grade'], drop_first=True)
y = data['Price']
x_train, x_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=42)
x_train.head()
reg = LinearRegression()
reg.fit(x_train, y_train)
print('Coefficients: \n', reg.coef_)
print('Mean squared error = {}'.format(metrics.mean_squared_error(y_test, reg.predict(x_test))))
print('R square = {}'.format(reg.score(x_test, y_test)))


df = pd.get_dummies(data, columns=['Waterfront', 'View', 'Condition', 'Grade', 'ZIPCode'], drop_first=True)
y = data['Price']
df = df.drop(['Date', 'ID', 'Price'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=42)
reg.fit(x_train, y_train)
print('Coefficients: \n', reg.coef_)
print('Mean squared error = {}'.format(metrics.mean_squared_error(y_test, reg.predict(x_test))))
print('R square = {}'.format(reg.score(x_test, y_test)))
