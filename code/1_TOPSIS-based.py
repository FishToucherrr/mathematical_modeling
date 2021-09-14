import numpy as np
import pandas as pd

import matplotlib as mp 
mp.use('tkagg')
import matplotlib.pyplot as plt


M = np.matrix(np.array([[1.,3],[4,2]]))
idxType = np.array([2,2])


def MatrixNormalization(M: np.matrix):
    N = np.multiply(M, M)
    M /= np.power(np.sum(N, 0), 1/2)


def ScoreCalculation(M: np.matrix) -> np.matrix:
    n = M.shape[0]
    Max_m = np.repeat(np.max(M,0), n, 0)
    Min_m = np.repeat(np.min(M,0), n, 0)
    D_P = np.power(np.sum(np.power(M - Max_m, 2), 1), 1/2)
    D_N = np.power(np.sum(np.power(M - Min_m, 2), 1), 1/2)
    return D_N / (D_N+D_P)


def Min2Max(Col: np.matrix) -> np.matrix:
    return max(Col) - Col


def Mid2Max(Col: np.matrix, best: float) -> np.matrix:
    M = max(abs(Col-best))
    return 1 - abs(Col-best)/M
    

def Inter2Max(Col: np.matrix, a: float, b: float) -> np.matrix:
    M = max(max(Col)-a, b-min(Col))
    n = Col.shape[0]
    tmp = np.matrix(np.zeros(Col.shape))
    for i in range(n):
        num = Col[i,0]
        if(num > a):
            tmp[i,0] = 1-(num-a) / M
        elif(num < b):
            tmp[i,0] = 1-(b-num) / M
        else: 
            tmp[i,0] = 1
    return tmp


def MatrixPositivization(M: np.matrix, idxType: np.array):
    print("Processing matrix positivization")
    n = M.shape[1]
    if idxType.shape[0] != n: 
        print("inputs are not aligned")
        exit(1)
    for i,idx in enumerate(idxType):
        print("processing column:",i,end=":" )
        if(idx == 0):     #极大型
            print("max")
        elif(idx == 1):   #极小型指标转化成极大型指标
            print("min")
            M[:,i] = Min2Max(M[:,i])
        elif(idx == 2):   #中间型指标
            print("mid")
            best = float(input("please input the ideal MIDDLE measure:"))
            M[:,i] = Mid2Max(M[:,i], best)
        elif(idx == 3):   #区间型指标
            print("Inter")
            Up = float(input("please assign the UPPER bound of the range"))
            Down = float(input("please assign the LOWER bound of the range"))
            M[:,i] = Inter2Max(M[:,i], Up, Down)
            #print(N[:,i])
        else: 
            print("Invalid idx type")
            exit(1)


supplierFile = r'D:\Courses\additional\mathematical_modeling\题目\C\附件1 近5年402家供应商的相关数据.xlsx'
bookingTable = pd.read_excel(supplierFile, sheet_name='企业的订货量（m³）')
supplyingTable = pd.read_excel(supplierFile, sheet_name='供应商的供货量（m³）')

#idxType = np.array([0,2,1,3])
bookingTable.head()

tableS = supplyingTable.values[:,2:]
judge = [0] * tableS.shape[0]
TotalWeeks = 240

for i,t in enumerate(supplyingTable['材料分类']):
    if t == 'A': judge[i] = 0.6
    elif t == 'B': judge[i] = 0.66
    else: judge[i] = 0.72
judge = np.matrix(judge)
tableSW = tableS / judge.T 
dailyProduction = np.sum(tableSW, axis=0)
SumS = np.sum(tableSW, axis=1)
AverageS = SumS / TotalWeeks

tableB = bookingTable.values[:,2:]
tableD = tableB - tableS
Diff = np.power((np.square(np.sum(tableS-tableB, axis=1)) / TotalWeeks), 1/2)
DiffW = (np.matrix(Diff).T) / AverageS
em = np.hstack((AverageS, DiffW, judge.T))
idxType = np.array([0,1,1])
MatrixNormalization(em)
MatrixPositivization(em, idxType)

result = ScoreCalculation(em)


rslt_list = result.argsort(axis=0)[-50:]
x_list = list(np.arange(240))


'''
select_list = []
for i in range(50):
    cur = str(rslt_list[i]+1)
    cur = cur[2:]
    cur = cur[:-2]
    cur = 'S' + cur
    select_list.append(cur)

name_list_1 = []
name_list_2 = []


plt.figure(figsize=(20,10))

plt.title('50 suppliers selected by TOPSIS-based model')
plt.xlabel('Week')
plt.ylabel('Supplement')


for i in range(50):
    plt.plot(x_list, tableS[rslt_list[i][0]][0][0], marker = 'o', markersize = 3)
    if i in range(25):
        name_list_1.append(select_list[i])
    else :
        name_list_2.append(select_list[i])

plt.show()
'''