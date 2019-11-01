交叉验证法
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
wine=load_wine()
svc=SVC(kernel='linear')
scores=cross_val_score(svc,wine.data,wine.target,cv=6)


#分类模型中的准确率
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
X,y=make_blobs(n_samples=200,random_state=1,centers=2,cluster_std=5)
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.cool,edgecolors='k')
from sklearn.naive_bayes import  GaussianNB
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=68)
gnb=GaussianNB()
gnb.fit(X_train,y_train)
predict_proba=gnb.predict_proba(X_test)
print(predict_proba)

#画图,用predict_proba确定模型的准确率
import matplotlib.pyplot as plt
x_min,x_max=X[:,0].min()-0.5,X[:,0].max()+0.5
y_min,y_max=X[:,1].min()-0.5,X[:,1].max()+0.5
import numpy as np
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
Z=gnb.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1].reshape(xx.shape)
plt.contourf(xx,yy,Z,cmap=plt.cm.summer,alpha=.8)  #alpha透明度
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.cool,edgecolors='k')
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=plt.cm.winter,edgecolors='k',alpha=0.9)
plt.xlim(xx.min(),xx.max())
plt.xlim(yy.min(),yy.max())











