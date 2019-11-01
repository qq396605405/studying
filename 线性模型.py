#画直线图
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-5,5,100)
y=0.5*x+3
plt.plot(x,y,c='k')
plt.title('Straight Line')
plt.show()

#线性模型的图形表示
from sklearn.linear_model import LinearRegression
X=[[1],[4]]
y=[3,5]
lr=LinearRegression().fit(X,y)
z=np.linspace(0,5,20)
plt.scatter(X,y,s=100,c='b')
plt.plot(z,lr.predict(z.reshape(-1,1)),c='k')
print('这条直线是y={:.3f}'.format(lr.coef_[0]),'x+','{:0.3f}'.format(lr.intercept_))


#多数据线性回归拟合
from sklearn.datasets import make_regression
X,y=make_regression(n_samples=50,n_features=1,n_informative=1,noise=50,random_state=1)
reg=LinearRegression()
reg.fit(X,y)
#z是我们生成的等差数列，用来画出线性模型的图形
z=np.linspace(-3,3,200).reshape(-1,1)
plt.scatter(X,y,c='b',s=60)
plt.plot(z,reg.predict(z),c='k')
plt.title('linear regression')
plt.show()
print('这条直线是y={:.3f}'.format(reg.coef_[0]),'x+','{:.3f}'.format(reg.intercept_))

#线性回归
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X,y=make_regression(n_samples=100,n_features=2,n_informative=2,random_state=38)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=8)
lr=LinearRegression().fit(X_train,y_train)
lr.score(X_test,y_test)

#糖尿病数据集
from sklearn.datasets import load_diabetes
X,y=load_diabetes().data,load_diabetes().target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=8)
lr=LinearRegression().fit(X_train,y_train)
lr.score(X_test,y_test)






