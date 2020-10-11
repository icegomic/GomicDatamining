from sklearn import datasets
import numpy as np
from sklearn.neural_network import MLPClassifier
np.random.seed(0)
iris=datasets.load_iris()
iris_x=iris.data
iris_y=iris.target
indices = np.random.permutation(len(iris_x))
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_x_test  = iris_x[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(10,10,10), random_state=1)
clf.fit(iris_x_train,iris_y_train) #拟合
#result = clf.predict()
iris_y_predict = clf.predict(iris_x_test)
 #调用该对象的测试方法，主要接收一个参数：测试数据集
#probility=clf.predict_proba(iris_x_test)
 #计算各测试样本基于概率的预测
#neighborpoint=clf.kneighbors([iris_x_test[-1]],5)
#计算与最后一个测试样本距离在最近的5个点，返回的是这些样本的序号组成的数组
score=clf.score(iris_x_test,iris_y_test,sample_weight=None)
#调用该对象的打分方法，计算出准确率

print('iris_y_predict = ')
print(iris_y_predict)

print('iris_y_test = ')
print(iris_y_test)
print('Accuracy:',score)
print('layers nums :',clf.n_layers_)