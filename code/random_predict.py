import pandas as pd
import numpy as np
import tensorflow as tf
df = open('d:\WorkSpace\\git\\hs300_stock_predict\\data\\bank_test_mix.csv')
data = pd.read_csv(df)
print(type(data))
len=2484
a = [0,0,0,0,0,0]
for index, row in data.iterrows():
    p_change=row['p_change']
    if p_change > 2:
        a[5]+=1
    elif 1 < p_change <= 2:
        a[4]+=1
    elif 0 < p_change <= 1:
        a[3]+=1
    elif -1 < p_change <= 0:
        a[2]+=1
    elif -2 < p_change <= -1:
        a[1]+=1
    else:
        a[0]+=1
b=[]
for i in a:
    b.append(i/len)
print(b)
for i in range(1,6):
    b[i]+=b[i-1]

real_6=[]
real_3=[]
real_2=[]
for index, row in data.iterrows():
    p_change=row['p_change']
    if p_change > 2:
        real_6.append(5)
        real_3.append(2)
        real_2.append(1)
    elif 1 < p_change <= 2:
        real_6.append(4)
        real_3.append(2)
        real_2.append(1)
    elif 0 < p_change <= 1:
        real_6.append(3)
        real_3.append(1)
        real_2.append(1)
    elif -1 < p_change <= 0:
        real_6.append(2)
        real_3.append(1)
        real_2.append(0)
    elif -2 < p_change <= -1:
        real_6.append(1)
        real_3.append(0)
        real_2.append(0)
    else:
        real_6.append(0)
        real_3.append(0)
        real_2.append(0)


ran_6=[]
ran_3=[]
ran_2=[]
for i in range(0,2484):
    tmp=np.random.rand()
    for j in range(0,6):
        if(tmp<=b[j]):
            ran_6.append(j)
            break
    if ran_6[i] == 5:
        ran_3.append(2)
        ran_2.append(1)
    elif ran_6[i] == 4:
        ran_3.append(2)
        ran_2.append(1)
    elif ran_6[i] == 3:
        ran_3.append(1)
        ran_2.append(1)
    elif ran_6[i] == 2:
        ran_3.append(1)
        ran_2.append(0)
    elif ran_6[i] == 1:
        ran_3.append(0)
        ran_2.append(0)
    else:
        ran_3.append(0)
        ran_2.append(0)

acc_6=0
acc_3=0
acc_2=0
for i in range(0,2484):
    if(ran_6[i]==real_6[i]): acc_6+=1
    if (ran_3[i] == real_3[i]): acc_3 += 1
    if (ran_2[i] == real_2[i]): acc_2 += 1
print('随机六分类 '+str(acc_6/len)+' '+str(acc_6))
print('随机三分类 '+str(acc_3/len)+' '+str(acc_3))
print('随机二分类 '+str(acc_2/len)+' '+str(acc_2))

