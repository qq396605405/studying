from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
wine=load_wine()
X=wine.data[:,:2]
y=wine.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
mlp=MLPClassifier(solver='lbfgs',hidden_layer_sizes=[10,10],activation='tanh',alpha=0.001)
mlp.fit(X_train,y_train)
#画图
import matplotlib.pyplot as plt
from matplotlib.colors import  ListedColormap
import numpy as np
cmap_light=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])  #颜色设置 可以设为camp_light-ListedColormap(['r','b','k'])
cmap_bold=ListedColormap(['#FF0000','#00FF00','#0000FF'])
x_min,x_max=X_train[:,0].min()-1,X_train[:,0].max()+1
y_min,y_max=X_train[:,1].min()-1,X_train[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
Z=mlp.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)  #这里cmap可以等于系统自带的 如plt.cm.winter
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolors='k',s=20)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.title('Classifier')
plt.show()
mlp.score(X_test,y_test)


#神经网络实例

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
mnist=fetch_openml('MNIST original',data_home='C:/Users/hk/scikit_learn_data/openml/openml.org/api/v1/json/data/list/data_name/mnist-original/limit/2/status/active/')
X=mnist.data/255
y=mnist.target
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=5000,test_size=1000,random_state=62)
mlp_hw=MLPClassifier(solver='lbfgs',hidden_layer_sizes=[100,100],activation='relu',alpha=0.001,random_state=62)
mlp_hw.fit(X_train,y_train)
















