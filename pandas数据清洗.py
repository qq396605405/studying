import pandas as pd
import numpy as np
df=pd.read_csv('BL-Flickr-Images-Book.csv',encoding="ISO-8859-1")
to_drop = ['Edition Statement',
           'Corporate Author',
           'Corporate Contributors',
           'Former owner',
           'Engraver',
           'Contributors',
          'Issuance type',
          'Shelfmarks']

df.drop(to_drop, inplace=True, axis=1)  #列是axis=1
df.drop(columns=to_drop, inplace=True)
#查找某列值是否唯一
df['Identifier'].is_unique
df['Place of Publication'].is_unique
df = df.set_index('Identifier')     #这列值没有重复 可以作为索引
extr = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
df['Date of Publication'] = pd.to_numeric(extr)
df['Date of Publication'].isnull().sum() / len(df)     #处理缺失
pub = df['Place of Publication']
london = pub.str.contains('London') #模糊筛选，筛选出是否是London的布尔值
oxford = pub.str.contains('Oxford')
df['Place of Publication'] = np.where(london, 'London',
                                      np.where(oxford, 'Oxford',
                                               pub.str.replace('-', ' ')))

df.isnull.any()  #查找各列是否存在缺失
df[df.isnull().values==True] #查找缺失值位置
df.drop_duplicates(keep=False,inplace=False)  #keep=False表示删除所有重复值


import pandas as pd
df=pd.read_csv('a.csv',encoding='ANSI',names=['name','age','sex','phone'])
df[df.isnull().values==True]   #找缺失值的位置
df.isnull().any()
df.dropna()  #删除缺失值
df.fillna(0) #填补缺失值，也可以通过value=  的方法实现每列不同的替换
df.isna()
df['is_duplicated'] = df.duplicated(['name'],keep=False)
df_dup = df.loc[df['is_duplicated'] == True]    #找重复位置
df3=pd.DataFrame([['徐扬','','','']],columns=['name','age','sex','phone'])  #colums=['name','age','sex','phone']
df4=pd.concat([df,df3],ignore_index=True,axis=0)   #行合并

#书上例子
import pandas as pd
df=pd.read_csv('C:/Users/hk/Desktop/《Python3爬虫、数据清洗与可视化》配套代码和数据集/data/pandas data/taobao_data.csv')
df['销售额']=df['价格']*df['成交量']  #生成新的列
df[(df['价格']<100)]   #找出满足条件的数据
df1=df.set_index('位置') #将位置作为标签
df1=df1.sort_index()
df2=df.set_index(['位置','卖家']).sort_index()
df.info()  #查看表的一些基本信息
df.describe() #查看表的描述性统计信息
grouped=df.groupby(df['位置']).mean()  #分组
grouped1=df.groupby(df['卖家']).mean()
df1=df[30:40][['位置','卖家']]
df2=df[80:90][['卖家','销售额']]
df3=pd.merge(df1,df2,on='卖家')
df4=pd.merge(df1,df2,how='outer')
df1.join(df2) #按照索引合并
#轴向连接
s1=df[:5]['宝贝']
s2=df[:5]['价格']
s3=df[:5]['成交量']
df5=pd.concat([s1,s2,s3],axis=1)


import pandas as pd
data=pd.read_csv('C:/Users/hk/Desktop/《Python3爬虫、数据清洗与可视化》配套代码和数据集/data/pandas data/hz_weather.csv')
data=data.stack()   #转化为series
data=data.unstack()
df=data.set_index('日期')

#旅游数据的分析与变形
df=pd.read_csv('C:/Users/hk/Desktop/《Python3爬虫、数据清洗与可视化》配套代码和数据集/data/pandas data/qunar_free_trip.csv')
df1=df['价格'].groupby([df['出发地'],df['目的地']]).mean()
df_=pd.read_csv('C:/Users/hk/Desktop/《Python3爬虫、数据清洗与可视化》配套代码和数据集/data/pandas data/qunar_route_cnt.csv')
df1=df.groupby([df['出发地'],df['目的地']],as_index=False).mean()
df2=pd.merge(df,df_)
df3=pd.pivot_table(df,values=['价格'],index=['出发地'],columns=['目的地'])
df4=pd.pivot_table(df[df['出发地']=='杭州'],values=['价格'],index=["出发地","目的地"],columns=['去程方式'])
#缺失值处理
df1.fillna(method='')  #pad是利用前一个数据代替NAN,bfill用后一个数据代替NAN,或者可以填入均值
df4=pd.pivot_table(data,values=['最高气温'],index=["天气"],columns=['风向'])


#旅游数据的值检查与处理
import pandas as pd
df=pd.read_csv('C:/Users/hk/Desktop/《Python3爬虫、数据清洗与可视化》配套代码和数据集/data/pandas data/qunar_free_trip.csv')
df.info()  #查看是否有缺失
df.isnull().any()   #查看列是否有缺失
df.duplicated().value_counts()  #查找有多少个重复
df['重复']=df.duplicated(['出发地'],keep=False)
df=df.drop_duplicates(['出发地'],keep=False)

from datetime import datetime
dt=datetime.now()
print(dt.strftime('%Y-%m-%d %H:%M:%S'))

import time
df=time.localtime()
print(time.strftime('%Y-%m-%d %H:%M:%S'))

import pandas as pd
df=pd.read_csv('C:/Users/hk/Desktop/《Python3爬虫、数据清洗与可视化》配套代码和数据集/data/pandas data/getlinks.csv')
df.link.str.extract('(.*/\d+)')



import numpy as np
b=np.empty(11)
b.fill(1.11)
b.shape()
a=[1,2,3]
b=a.copy()


import matplotlib.pyplot as plt
img=plt.imread('C:/Users/hk/Desktop/DSC_9058.jpg')
plt.imshow(img)
arr_lin=np.linspace(1,10,10)




import pandas as pd
df=pd.read_csv('a.csv',encoding='gb2312')
df.drop(columns='sex',inplace=True)  #删除列
chongfu=df.duplicated(keep=False)   #找重复
df1=df.drop_duplicates(subset='age')
df['is_duplicated']=chongfu
xuanze=df[df['age']>24]     #选择


class orange:
    def __init__(self,w,c):
        self.weight=w
        self.color=c
        print('创建成功')
or1=orange(90,'red')


class dog():
    def __init__(self,name,breed,owner):
        self.name=name
        self.breed=breed
        self.owner=owner
class person():
    def __init__(self,name):
        self.name=name

xy=person('XY')
stan=dog('st','bu',xy)

print(stan.owner.name)





