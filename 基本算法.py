#冒泡排序
data=list(range(30))
import random
random.shuffle(data)
for i in range(len(data)-1):
    for j in range(len(data)-1-i):
        if data[j]>data[j+1]:
            data[j],data[j+1]=data[j+1],data[j]

#生成斐波拉契数列
#1 1 2 3 5 8 13

def f(x):
    if x>=2:
        return f(x-1)+f(x-2)
    if x<2:
        return  1

def feibo(x):
    for i in range(x):
        print(f(i),end='  ')

#阶乘
def jiecheng(n):
    sum=1
    for i in range(1,n+1):
        sum=sum*i
    return sum

def jiecheng1(n):
    if n>1:
        return jiecheng(n-1)*n
    if n<=1:
        return 1




