from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

boston = load_boston()

print('--- %s ---' % 'boston type')
print(type(boston))
print('--- %s ---' % 'boston keys')
print(boston.keys())
print('--- %s ---' % 'boston data')
print(type(boston.data))

print('--- %s ---' % 'boston target')
print(type(boston.target))
print('--- %s ---' % 'boston data shape')
print(boston.data.shape)

print('--- %s ---' % 'boston feature names')
print(boston.feature_names);


X = boston.data
y = boston.target
df = pd.DataFrame(X, columns= boston.feature_names)

print('--- %s ---' % 'df.head')
print(df.head())
print('--- %s ---' % 'df.info')
print(df.info())
print('--- %s ---' % 'df.describe')
print(df.describe())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

regressor = LinearRegression()

regressor.fit(X_train, y_train) #training the algorithm
#我们前面曾讨论过，线性回归模型可以大致找到截距和斜率的最佳值，从而确定最符合相关数据的线的位置。如要查看线性回归算法为我们的数据集计算的截距和斜率的值，请执行以下代码。
#To retrieve the intercept:

print(regressor.intercept_)

#For retrieving the slope:
#print("regressor.coef_")
print(regressor.coef_)

#coeff_df= pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df= pd.DataFrame(regressor.coef_, df.columns, columns=['Coefficient'])

coeff_df
print(coeff_df)


y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted':y_pred.flatten()})
print(df)
#还可以使用以下脚本，用条形图的方式展示这些比较：
#注意：由于记录数量庞大，为了更好的展示效果，我们只使用了其中25条记录。
df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
#plt.scatter(X_test, y_test, color='gray')
#plt.plot(X_test, y_pred, color='red', linewidth=2)
#plt.show()
print('MeanAbsolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('MeanSquared Error:', metrics.mean_squared_error(y_test, y_pred))
print('RootMean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))