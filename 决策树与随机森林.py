import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import tree,datasets
from sklearn.model_selection import train_test_split
wine=datasets.load_wine()
X=wine.data[:,:2]
y=wine.target
X_train,X_test,y_train,y_test=train_test_split(X,y)
clf=tree.DecisionTreeClassifier(max_depth=1)
clf.fit(X_train,y_train)
#画图
cmap_light=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold=ListedColormap(['#FF0000','#00FF00','#0000FF'])
x_min,x_max=X_train[:,0].min()-1,X_train[:,0].max()+1
y_min,y_max=X_train[:,1].min()-1,X_train[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
Z=clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
#画出样本散点图
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolors='k',s=20)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.title('Classifier')
plt.show()


#max_depth=5
clf3=tree.DecisionTreeClassifier(max_depth=5)
clf3.fit(X_train,y_train)
#画图
cmap_light=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold=ListedColormap(['#FF0000','#00FF00','#0000FF'])
x_min,x_max=X_train[:,0].min()-1,X_train[:,0].max()+1
y_min,y_max=X_train[:,1].min()-1,X_train[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
Z=clf3.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolors='k',s=20)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.title('Classifier')
plt.show()
#展示决策树工作过程


#随机森林的构建
from sklearn.ensemble import RandomForestClassifier
wine=datasets.load_wine()
X=wine.data[:,:2]
y=wine.target
X_train,X_test,y_train,y_test=train_test_split(X,y)
forest=RandomForestClassifier(n_estimators=6,random_state=3)
forest.fit(X_train,y_train)
#画图
cmap_light=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])  #颜色设置 可以设为camp_light-ListedColormap(['r','b','k'])
cmap_bold=ListedColormap(['#FF0000','#00FF00','#0000FF'])
x_min,x_max=X_train[:,0].min()-1,X_train[:,0].max()+1
y_min,y_max=X_train[:,1].min()-1,X_train[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
Z=forest.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolors='k',s=20)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.title('Classifier')
plt.show()


#随机森林实战分析
import pandas as pd
data=pd.read_csv('adult.csv',header=None,index_col=False,names=['年龄','单位性质','权重','学历','受教育时长',
                                                                '婚姻状况','职业','家庭情况','种族','性别',
                                                                '资产所得','资产损失','周工作时长','原籍','收入'])
data_lite=data[['年龄','单位性质','学历','性别','周工作时长','职业','收入']]
data_dummies=pd.get_dummies(data_lite)
features=data_dummies.loc[:,'年龄':'职业_ Transport-moving']  #x特征
X=features.values
y=data_dummies["收入_ >50K"].values
from sklearn.model_selection import train_test_split
from sklearn import tree
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
forests=tree.DecisionTreeClassifier(max_depth=5)
forests.fit(X_train,y_train)
forests.score(X_test,y_test)


#XGB
import xgboost
from xgboost import XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=load_wine()
X=data.data[:,:2]
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=8)
XGB=XGBClassifier()
XGB.fit(X_train,y_train)
y_pred=XGB.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)

