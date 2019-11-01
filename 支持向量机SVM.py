import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=50,centers=2,random_state=6)
clf=svm.SVC(kernel='rbf',C=1000)
clf.fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=30,cmap=plt.cm.Paired)
ax=plt.gca()         #画矩形画布
xlim=ax.get_xlim()  #返回当前画布中x的上下限
ylim=ax.get_ylim()
xx=np.linspace(xlim[0],xlim[1],30)
yy=np.linspace(ylim[0],ylim[1],30)
YY,XX=np.meshgrid(yy,xx)
xy=np.vstack([XX.ravel(),YY.ravel()]).T   # np.vstack将数组垂直叠加,.T为转置
Z=clf.decision_function(xy).reshape(XX.shape)  #decision_function代表的是参数实例到各个类所代表的超平面的距离
ax.contour(XX,YY,Z,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])  #countour画等高线
ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=100,linewidths=1,facecolors='none')
plt.show()


#SVM实例
from sklearn.datasets import load_boston
boston=load_boston()
from sklearn.model_selection import train_test_split
X,y=boston.data,boston.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=8)












