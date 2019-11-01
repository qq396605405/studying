import numpy as np
X=np.array([[0,1,0,1],[1,1,1,0],[0,1,1,0],[0,0,0,1],[0,1,1,0],[0,1,0,1],[1,0,0,1]])
y=np.array([0,1,1,0,1,0,0])
counts={}
for label in np.unique(y):
    counts[label]=X[y==label].sum(axis=0)
print(counts)

#朴素贝叶斯实现
from sklearn.naive_bayes import  BernoulliNB
clf=BernoulliNB()
clf.fit(X,y)
Next_day=[[0,0,1,0]]
pre=clf.predict(Next_day)
print('准确率:{}'.format(clf.predict_proba(Next_day)))

#贝努利朴素贝叶斯的局限
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
X,y=make_blobs(n_samples=500,centers=5,random_state=8)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=8)
nb=BernoulliNB()
nb.fit(X_train,y_train)
nb.score(X_test,y_test)
#得分效果很差，画图查看原理
import matplotlib.pyplot as plt
x_min,x_max=X[:,0].min()-0.5,X[:,0].max()+0.5
y_min,y_max=X[:,1].min()-0.5,X[:,1].max()+0.5
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
z=nb.predict(np.c_[(xx.ravel(),yy.ravel())]).reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.Pastel1)


#高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
gnb.score(X_test,y_test)
z=gnb.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.Pastel1)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.cool,edgecolors='k')
plt.scatter(X_train[:,0],X_train[:,1],c=y_test,cmap=plt.cm.cool,marker='*',edgecolors='k')



#朴素贝叶斯判断肿瘤是否为恶性
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
cancer=load_breast_cancer()
from sklearn.model_selection import train_test_split
X,y=cancer['data'],cancer['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=38)
gnb=GaussianNB()
gnb.fit(X_train,y_train)
gnb.score(X_test,y_test)

