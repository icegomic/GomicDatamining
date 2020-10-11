from sklearn import datasets

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

""" 画出聚类后的图像
labels: 聚类后的label, 从0开始的数字
cents: 质心坐标
n_cluster: 聚类后簇的数量
color: 每一簇的颜色
"""
def draw_result(train_x, labels, cents, title):
    n_clusters = np.unique(labels).shape[0]
    color = ["red", "orange", "yellow"]
    plt.figure()
    plt.title(title)
    for i in range(n_clusters):
        current_data = train_x[labels == i]
        plt.scatter(current_data[:, 0], current_data[:,1], c=color[i])
        plt.scatter(cents[i, 0], cents[i, 1], c="blue", marker="*", s=100)
    return plt


if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_x = iris.data
    clf = KMeans(n_clusters=3, max_iter=10,n_init=10, init="k-means++", algorithm="full", tol=1e-4,n_jobs= -1,random_state=1)
    clf.fit(iris_x)
    print("SSE = {0}".format(clf.inertia_))
    draw_result(iris_x, clf.labels_, clf.cluster_centers_, "kmeans").show()