import pandas as pd
import numpy as np

import matplotlib as mp 
mp.use('tkagg')
import matplotlib.pyplot as plt


data = pd.read_excel(r"D:\Courses\additional\mathematical_modeling\题目\C\附件1 近5年402家供应商的相关数据.xlsx",
                sheet_name='供应商的供货量（m³）')
data_1=pd.read_excel(r"D:\Courses\additional\mathematical_modeling\题目\C\附件1 近5年402家供应商的相关数据.xlsx",
                sheet_name='企业的订货量（m³）')
data_2 = pd.read_excel(r"D:\Courses\additional\mathematical_modeling\题目\C\附件2 近5年8家转运商的相关数据.xlsx",
                sheet_name = '运输损耗率（%）')

data=data.iloc[0:,1:]
data_1=data_1.iloc[0:,1:]
data_2=data_2.iloc[0:,1:]

data=np.array(data)
data_1=np.array(data_1)
data_2=np.array(data_2)

x_list = list(np.arange(240))
name_list = list(np.arange(8))

print(data_2.shape)


for i in range(8):

    plt.subplot(2, 4, i+1)
    plt.title('loss rate for shipper'+str(i))
    if (i in range(4, 8)):
        plt.xlabel('week')
    plt.ylabel('loss rate')


    plt.plot(x_list,data_2[i],marker='o', markersize=3)

plt.show()

'''

mean_list = list(np.arange(8))

for i in range(8):
    mean_list[i] = np.mean(data_2[i])

print(mean_list)

'''