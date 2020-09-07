import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def generateData():
    X = []
    y = []
    for i in range(0, 100):
        tem_x = []
        tem_x.append(i)
        X.append(tem_x)
        tem_y = []
        tem_y.append(i + 2.128 + np.random.uniform(-15,15))
        y.append(tem_y)
    plt.scatter(X, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    #plt.show()
    return X,y
if __name__ == '__main__':
    np.random.seed(0)
    X,y = generateData()
    print(len(X))
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_result = regressor.predict(X_train)
    plt.plot(X_train, y_result, color='red',alpha=0.6, linewidth=3, label='Predicted Line')  # plotting
    plt.show()