#对数据进行哑变量处理
import pandas as pd
fruits=pd.DataFrame({'数值特征':[5,6,7,8,9],'类型特征':['西瓜','香蕉','橘子','苹果','葡萄']})
fruits_dum=pd.get_dummies(fruits)

#比较KNN与MLP
#1.生成随机数据
import numpy as np
import matplotlib.pyplot as plt
rnd=np.random.RandomState(38)
x=rnd.uniform(-5,5,size=50)   #生成-5至5之间的随机数
y_no_noise=(np.cos(6*x)+x)
X=x.reshape(-1,1)
t=rnd.normal(size=len(x))  #生成长为50的正态分布数据
y=y_no_noise+t/2
plt.scatter(x,y,c='r')
plt.show()
#模拟神经网络
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
line=np.linspace(-5,5,1000,endpoint=False).reshape(-1,1)
mlpr=MLPRegressor().fit(X,y)
knr=KNeighborsRegressor().fit(X,y)
plt.plot(line,mlpr.predict(line),c='r',label='MLP')
plt.plot(line,knr.predict(line),c='b',label='KN')
plt.legend(loc='best')
plt.plot(X,y,'o',c='r')

#装箱处理
bins=np.linspace(-5,5,11)
target_bin=np.digitize(X,bins=bins)  #查看X数据在哪个箱子中
#使用独热编码
from sklearn.preprocessing import OneHotEncoder
onehot=OneHotEncoder(sparse=False,categories='auto')
onehot.fit(target_bin)
X_in_bin=onehot.transform(target_bin)







