from sklearn.datasets import make_blobs
blobs=make_blobs(random_state=1,centers=1)
X_blobs=blobs[0]
y_blobs=blobs[1]
import matplotlib.pyplot as plt
plt.scatter(X_blobs[:,0],X_blobs[:,1],c=y_blobs,edgecolors='k')

#K均值
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3)
kmeans.fit(X_blobs)
x_min,x_max=X_blobs[:,0].min()-0.5,X_blobs[:,0].max()+0.5
y_min,y_max=X_blobs[:,1].min()-0.5,X_blobs[:,1].max()+0.5
import numpy as np
a=np.arange(x_min,x_max,.02)
b=np.arange(y_min,y_max,.02)
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
t1=xx.ravel()
t2=yy.ravel()
t3=np.c_[t1,t2]
Z=kmeans.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z,interpolation='nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),cmap=plt.cm.summer,
           aspect='auto',origin='lower')  #nearest表示某块地方显示一个颜色
plt.plot(X_blobs[:,0],X_blobs[:,1],'r.',markersize=5)
centroids=kmeans.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidths=3,color='b',zorder=10)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks(())
plt.xticks(())
plt.show()










