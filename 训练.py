from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
data=make_blobs(n_features=2,centers=2,random_state=8)
X,y=data[0],data[1]
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolors='k')
KNN=KNeighborsClassifier()
KNN.fit(X,y)
x_min,x_max=X[:,0].min(),X[:,0].max()
y_min,y_max=X[:,1].min(),X[:,1].max()
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
Z=KNN.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Pastel1)
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolors='k')
plt.scatter(6.75,4.82,marker='X')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.show()

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
data=make_blobs(n_samples=500,centers=5,random_state=8)
X,y=data[0],data[1]
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolors='k')
import numpy as np
x_min,x_max=X[:,0].min(),X[:,0].max()
y_min,y_max=X[:,1].min(),X[:,1].max()
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
KNN=KNeighborsClassifier()
KNN.fit(X,y)
Z=KNN.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Pastel1)
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolors='k')












