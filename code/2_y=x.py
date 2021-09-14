import pandas as pd
import numpy as np
import pulp

import matplotlib as mp 
mp.use('tkagg')
import matplotlib.pyplot as plt



data = pd.read_excel(r"D:\Courses\additional\mathematical_modeling\题目\C\附件1 近5年402家供应商的相关数据.xlsx",
                sheet_name='供应商的供货量（m³）')
data_1=pd.read_excel(r"D:\Courses\additional\mathematical_modeling\题目\C\k附件1 近5年402家供应商的相关数据.xlsx",
                sheet_name='企业的订货量（m³）')

data=data.iloc[0:,1:]
data_1=data_1.iloc[0:,1:]

data=np.array(data)
data_1=np.array(data_1)

ratio = 1                       #原料转换比
data_ref=np.zeros((402,240))                  #将ABC的供货量转化为产品的生产量

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

num = np.arange(402)

num_result = zip(result,num)
list_result = list(num_result)


list_result=sorted(list_result,key=(lambda x:x[0]),reverse=True)


first_50 = []

for i in range(50):
    first_50.append(list_result[i][1])               #此处供应商序号从0开始，非从1开始

production_for_window=np.zeros(240)

count=0
data_for_window=np.zeros((50,240))

for i in range(402):
    if i in first_50:
        data_for_window[count]=data_ref[i]
        count+=1
        for j in range(240):
            production_for_window[j]+=data_ref[i][j]


satisfy=0

window_width=24     #滑动窗口宽度
start_week_set=[]      #满足条件的窗口 起始月份集合


for i in range(240-window_width):
    index=i
    stock=0
    satisfy=1
    while index < i+window_width :
        if stock + production[index] > 3*28200:
            stock=stock + production[index]-28200
            index+=1
        else:
            satisfy=0
            break
    if satisfy==1:
        start_week_set.append(i)

print(start_week_set)
#print(production_for_window[82])
'''
plt.title('supply')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('week')  # x轴标题
plt.ylabel('supply')  # y轴标题

'''
sol_list=[]
'''
for start_week in start_week_set:

    sum_matrix = np.zeros((50,24))

    for i in range(50):
        sum_matrix[i][0]=data_for_window[i][start_week]

    for i in range(50):
        for j in range(start_week+1,start_week+24):
            sum_matrix[i][j-start_week]=sum_matrix[i][j-1-start_week]+data_for_window[i][j]

    InvestLP = pulp.LpProblem("problem_for_window", sense=pulp.LpMinimize)

    types=[str(i) for i in range(1,51)]
    status = pulp.LpVariable.dicts("supplier",types,cat='Binary')

    InvestLP += pulp.lpSum([(status[i]) for i in types])


    for i in range(24):
        InvestLP += ((pulp.lpSum([(status[types[j]]*sum_matrix[j][i]) for j in range(50)])-i*28200)>=0)

                    
    InvestLP.solve()

    print(InvestLP.name)
    print("Status :", pulp.LpStatus[InvestLP.status])  # 输出求解状态
    for v in InvestLP.variables():
        print(v.name, "=", v.varValue)  # 输出每个变量的最优值
    
    print("Min f(x) =", pulp.value(InvestLP.objective))  # 输出最优解的目标函数值

    #sol_list.append(pulp.value(InvestLP.objective))
    #if pulp.value(InvestLP.objective) == 21:

#print(sol_list)
#print(len(start_week_set))
'''
start_week=start_week_set[0]

sum_matrix = np.zeros((50,24))

for i in range(50):
    sum_matrix[i][0]=data_for_window[i][start_week]

for i in range(50):
    for j in range(start_week+1,start_week+24):
        sum_matrix[i][j-start_week]=sum_matrix[i][j-1-start_week]+data_for_window[i][j]

InvestLP = pulp.LpProblem("problem_for_window", sense=pulp.LpMinimize)

types=['01','02','03','04','05','06','07','08','09']
for i in range(10,51):
    types.append(str(i))


status = pulp.LpVariable.dicts("supplier",types,cat='Binary')

InvestLP += pulp.lpSum([(status[i]) for i in types])


for i in range(24):
    InvestLP += ((pulp.lpSum([(status[types[j]]*sum_matrix[j][i]) for j in range(50)])-i*28200)>=2*28200)

                
InvestLP.solve()
print(InvestLP.name)
print("Status :", pulp.LpStatus[InvestLP.status])  # 输出求解状态
for v in InvestLP.variables():
    print(v.name, "=", v.varValue)  # 输出每个变量的最优值
print("Min f(x) =", pulp.value(InvestLP.objective))  # 输出最优解的目标函数值

count=0

least_suppliers=[]

for v in InvestLP.variables():
    if v.varValue==1:
        least_suppliers.append(count)
    count+=1

num_suppliers=len(least_suppliers)

material_type=[]
ratio_type={}
price_type={}

ratio_type['A']=0.6
ratio_type['B']=0.66
ratio_type['C']=0.72
price_type['A']=1.2
price_type['B']=1.1
price_type['C']=1

for i in least_suppliers:
    material_type.append(data[first_50[i]][0])


real_suppliers=[]

for i in least_suppliers:
    real_suppliers.append(first_50[i])

data_for_supply= np.zeros((num_suppliers,240)) #选定的21家供货商240周的供货情况 未转换为产品量
data_for_order = np.zeros((num_suppliers,240)) #选定的21家供货商240周的订货情况 未转换为产品量

count=0
for i in real_suppliers:
        data_for_order[count]=data_1[i][1:]
        data_for_supply[count]=data[i][1:]
        count+=1

hist_max=np.zeros(num_suppliers) # 选定的供货商历史供货峰值(转化为产品)

data_for_xxx= np.zeros((num_suppliers,240))

count=0
for i in real_suppliers:
    data_for_xxx[count]=data_ref[i]
    count+=1

for i in range(num_suppliers):
        for k in range(240):
            hist_max[i]=max(hist_max[i],data_for_xxx[i][k])

#print(hist_max)

s=np.zeros(24)
for i in range(num_suppliers):
    for j in range(24):
        s[j]+=hist_max[j]



one_list=np.zeros(24)

stock=np.zeros(24)
for i in range(1,24):
    stock[i]=stock[i-1]+s[i-1]-28200
    if stock[i] >=28200:
        one_list[i]=1

print(real_suppliers) #the index for each supplier
#print(stock)
#print(one_list)
#print(len(start_week_set))
#print(start_week_set)


flag = 1
#flag 0 order-supplement
#flag 1 supplement-order

if flag == 0 :
    for i in range(len(real_suppliers)):
        plt.title('order-supplement curve for suppler'+str(i))
        plt.xlabel('supplement')
        plt.ylabel('order')

        x_list = data[real_suppliers[i]][1:] #supplement
        y_list = data_1[real_suppliers[i]][1:] #order

        none_zero_cnt = 0
        for i in range(len(x_list)):
            if x_list[i] > 0:
                none_zero_cnt = none_zero_cnt + 1

        start = min(x_list)
        end = max(x_list)
        print(start, end, none_zero_cnt)

        ref_x = np.arange(start, end, 1)
        ref_y = ref_x

        plt.plot(ref_x, ref_y, marker = None)
        plt.scatter(x_list, y_list, marker='o')


        plt.show()
elif flag == 1 :
    for i in range(len(real_suppliers)):
        plt.title('supplement-order curve for suppler'+str(i))
        plt.ylabel('supplement')
        plt.xlabel('order')

        y_list = data[real_suppliers[i]][1:]
        x_list = data_1[real_suppliers[i]][1:]

        start = min(x_list)
        end = max(x_list)
        print(start, end, len(x_list))

        ref_x = np.arange(start, end, 1)
        ref_y = ref_x

        ref_y1 = 0.8 * ref_x

        plt.plot(ref_x, ref_y, marker = None)
        plt.plot(ref_x, ref_y1, marker = None)
        plt.scatter(x_list, y_list, marker='o')


        plt.show()
else :
    pass




plt.title('order-supplement curve for supplier0')
plt.xlabel('supplement')
plt.ylabel('order')

index = 0

x_list = data[real_suppliers[index]][1:]
y_list = data_1[real_suppliers[index]][1:]

start = min(x_list)
end = max(x_list)
print(start, end, len(x_list))

ref_x = np.arange(start, end, 1)
ref_y = ref_x

plt.plot(ref_x, ref_y, marker = None)
plt.scatter(x_list, y_list, marker='o')


plt.show()

#print(data)