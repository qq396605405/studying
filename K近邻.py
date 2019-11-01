
#K近邻处理二分类
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data=make_blobs(n_samples=200,centers=2,random_state=8)    #二分类
X,y=data
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolors='k')  #c根据y来画图
#上述已经生成好了训练数据

import numpy as np
clf = KNeighborsClassifier()
clf.fit(X,y)
#画图
x_min , x_max=X[:,0].min()-1,X[:,0].max()+1
y_min , y_max=X[:,1].min()-1,X[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))  #meshgrid生成网格点坐标矩阵，间隔0.02
a=np.c_[xx.ravel(),yy.ravel()]
Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])   #  np.c_将数组合起来  ravel 扁平化操作
Z=Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Pastel1)    #plt.pcolormesh可以画出分类图
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolors='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xlim(xx.min(),xx.max())
plt.scatter(6.75,4.82,marker='*',c='red', s=200)
clf.predict([[6.75,4.82]])
plt.show()


#K近邻处理多分类
data2=make_blobs(n_samples=500,centers=5,random_state=8)    #二分类
X2,y2=data2
plt.scatter(X2[:,0],X2[:,1],c=y2,cmap=plt.cm.spring,edgecolors='k')  #c根据y来画图
plt.show()
clf = KNeighborsClassifier()
clf.fit(X2,y2)
#画图
x_min , x_max=X2[:,0].min()-1,X2[:,0].max()+1
y_min , y_max=X2[:,1].min()-1,X2[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))  #meshgrid生成网格点坐标矩阵，间隔0.02
a=np.c_[xx.ravel(),yy.ravel()]
Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])   #  np.c_将数组合起来  ravel 扁平化操作
Z=Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Pastel1)
plt.scatter(X2[:,0],X2[:,1],c=y2,cmap=plt.cm.spring,edgecolors='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xlim(xx.min(),xx.max())
plt.show()

#K近邻用于回归分析
from sklearn.datasets import make_regression
X,y=make_regression(n_features=1,n_informative=1,noise=50,random_state=8)
plt.scatter(X,y,c='green',edgecolors='k')
from sklearn.neighbors import KNeighborsRegressor
reg=KNeighborsRegressor()
reg.fit(X,y)
z=np.linspace(-3,3,200).reshape(-1,1)
plt.scatter(X,y,c='orange',edgecolors='k')
plt.scatter(z,reg.predict(z),c='k',linewidth=3)

#K邻近---酒的分类
from sklearn.datasets import load_wine
wine_dataset=load_wine()

#拆分数据，分为数据集与训练集

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(wine_dataset['data'],wine_dataset['target'],random_state=0)
#random_stata=0时，产生不同的伪随机数。
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
#knn得分
knn.score(X_test,y_test)
#输入新的数据
import numpy as np
X_new=np.array([[13.2,2.77,2.51,18.5,96.6,1.04,2.55,0.57,1.47,6.2,1.05,3.33,820]])
prediction=knn.predict(X_new)
print("预测新红酒的分类为:{}".format(wine_dataset['target_names'][prediction]))











