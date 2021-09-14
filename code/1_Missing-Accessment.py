import pandas as pd
import numpy as np

import matplotlib as mp 
mp.use('tkagg')
import matplotlib.pyplot as plt


data = pd.read_excel(r"D:\Courses\additional\mathematical_modeling\题目\C\附件1 近5年402家供应商的相关数据.xlsx",
                sheet_name='供应商的供货量（m³）')

data=data.iloc[0:,1:]
data=np.array(data)

ratio = 1                       #原料转换比
data_ref=np.zeros((402,240))    #将ABC的供货量转化为产品的生产量

production = np.zeros(240)      #每周总原料量
production_deal = np.zeros(240) #每周去掉特定生产商的原料量

stock = 0                       #库存
difference = 0                  #差值 中间变量
result = np.zeros(402)

for i in range(402):
    if data[i][0]=='A':
        ratio = 0.6
    elif data[i][0]=='B':
        ratio = 0.66
    else:
        ratio = 0.72
    
    for j in range(240):
        data_ref[i][j] = data[i][j+1]/ratio

for i in range(240):
    for j in range(402):
        production[i]+= data_ref[j][i]

for j in range(402):
    stock = 0
    difference = 0
    for i in range(240):
        production_deal[i]=production[i]-data_ref[j][i]
    for i in range(240):
        if stock+production_deal[i]-28200 > 0:
            stock=stock+production_deal[i]-28200
            difference = 0
        else:
            difference = 28200 - stock - production_deal[i]
            stock = 0
        result[j]+=difference**2
    result[j]=(result[j]/240)**0.5

num = np.arange(1,403)

num_result = zip(result,num)
list_result = list(num_result)

list_result=sorted(list_result,key=(lambda x:x[0]),reverse=True)
print(result)


'''
x_list = list(np.arange(240))
 
name_list_1 = []
name_list_2 = []

plt.title('Supplement-Week curve for 50 suppliers selected by Missing-Accessment model')
#缺失评估
plt.xlabel('Week')
plt.ylabel('Supplement')


for i in range(50):
    #print(list_result[i])
    plt.plot(x_list, data[list_result[i][1]-1][1:], marker = 'o', markersize = 3)
    if i in range(25):
        name_list_1.append('S' + str(list_result[i][1]))
    else :
        name_list_2.append('S' + str(list_result[i][1]))
'''

#first_legend = plt.legend(name_list_1, loc=3)
#ax = plt.gca().add_artist(first_legend)
#plt.legend(name_list_2, loc = 4)

#plt.show()

