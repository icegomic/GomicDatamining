#这个需要安装graphviz
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
np.random.seed(0)
iris=datasets.load_iris()
iris_x=iris.data
iris_y=iris.target
indices = np.random.permutation(len(iris_x))
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_x_test  = iris_x[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(iris_x_train, iris_y_train)

from IPython.display import Image
from sklearn import tree
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=iris.feature_names,
class_names=iris.target_names,
filled=True, rounded=True,
special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
#把图片存下来
graph.write_png('tree.png')
iris_y_predict = clf.predict(iris_x_test)
 #调用该对象的测试方法，主要接收一个参数：测试数据集

score=clf.score(iris_x_test,iris_y_test,sample_weight=None)
#调用该对象的打分方法，计算出准确率

print('iris_y_predict = ')
print(iris_y_predict)
#输出测试的结果

print('iris_y_test = ')
print(iris_y_test)
#输出原始测试数据集的正确标签，以方便对比
print('Accuracy:',score)