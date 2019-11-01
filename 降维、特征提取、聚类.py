#数据预处理
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=40,centers=2,random_state=50,cluster_std=2)
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.cool)
#用StandardScaler进行数据预处理
from sklearn.preprocessing import StandardScaler
X_1=StandardScaler().fit_transform(X)
plt.scatter(X_1[:,0],X[:,1],c=y,cmap=plt.cm.cool)
#用MinMaxScaler进行数据预处理
from sklearn.preprocessing import MinMaxScaler
X_2=MinMaxScaler().fit_transform(X)
plt.scatter(X_2[:,0],X_2[:,1],c=y,cmap=plt.cm.cool)
#用Normalizer进行数据预处理
from sklearn.preprocessing import Normalizer
X_4=Normalizer().fit_transform(X)
plt.scatter(X_4[:,0],X_4[:,1],c=y,cmap=plt.cm.cool)


#数据降维
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
wine=load_wine()
scaler=StandardScaler()
X,y=load_wine().data,load_wine().target
X_scaled=scaler.fit_transform(X)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(X_scaled)
X_pca=pca.fit_transform(X_scaled)
X0=X_pca[wine.target==0]
X1=X_pca[wine.target==1]
X2=X_pca[wine.target==2]
plt.scatter(X0[:,0],X0[:,1],c='b',s=60,edgecolors='k')
plt.scatter(X1[:,0],X1[:,1],c='b',s=60,edgecolors='k')
plt.scatter(X2[:,0],X2[:,1],c='b',s=60,edgecolors='k')
plt.legend(wine.target_names,loc='best')
plt.xlabel('成分1')
plt.ylabel('成分2')
plt.show()


#人脸识别
from sklearn.datasets import fetch_lfw_people
faces=fetch_lfw_people(min_faces_per_person=20,resize=0.8)
image_shape=faces.images[0].shape


