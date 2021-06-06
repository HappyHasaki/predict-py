import warnings
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier  # 分类决策树模型

warnings.filterwarnings("ignore")
from dataprocess import *

def get_bank_list():
    f = open('d:\WorkSpace\\git\\hs300_stock_predict\\data\\list.csv')
    list = pd.read_csv(f)
    bank_list=[]
    for index,row in list.iterrows():
        if(row['industry']=='银行'):
            bank_list.append(row['ts_code'])
    return bank_list

def run(ts_code):
    # pd.set_option()就是pycharm输出控制显示的设置
    pd.set_option('expand_frame_repr', False)  # True就是可以换行显示。设置成False的时候不允许换行
    pd.set_option('display.max_columns', None)  # 显示所有列
    # pd.set_option('display.max_rows', None)# 显示所有行
    pd.set_option('colheader_justify', 'centre')  # 显示居中

    ts.set_token('1bf2b34c8f1b11ea031f7b95d12ca7e9f04142fa7e4e1e9353fff854')
    pro = ts.pro_api('1bf2b34c8f1b11ea031f7b95d12ca7e9f04142fa7e4e1e9353fff854')

    #1.数据准备
    # df = pro.daily(ts_code='000004.SZ', start_date='20190101', end_date='20210413')
    df = ts.pro_bar(ts_code=ts_code, start_date='20190101', end_date='20210413', factors=['tor'])
    # df=ts.get_hist_data('000001', start='2021-01-01', end='2021-04-13')
    # print(df.head(5))
    df.set_index('trade_date', inplace=True)  #设置date列为索引，覆盖原来索引,这个时候索引还是 object 类型，就是字符串类型。
    df.index = pd.DatetimeIndex(df.index)  #将object类型转化成 DateIndex 类型，pd.DatetimeIndex 是把某一列进行转换，同时把该列的数据设置为索引 index。
    df = df.sort_index(ascending=True)  #将时间顺序升序，符合时间序列
    df.dropna(inplace=True)

    #2.提取特征变量和目标变量，用当天收盘后获取完整的数据为特征变量，下一天的涨跌情况为目标变量这样来训练分类决策树模型
    X = df[['open', 'high', 'close', 'low', 'vol', 'pct_chg','turnover_rate']]
    y= df[['pct_chg']]
    for index,row in y.iterrows():
        change = row['pct_chg']
        if change > 2:
            row['pct_chg']=5
        elif 1 < change <= 2:
            row['pct_chg']=4
        elif 0 < change <= 1:
            row['pct_chg']=3
        elif -1 < change <= 0:
            row['pct_chg']=2
        elif -2 < change <= -1:
            row['pct_chg']=1
        else:
            row['pct_chg']=0
    # y = np.where(df['pct_chg'].shift(-1)>0, 1, -1)  #下一天股价涨，赋值1，下跌或平，赋值-1
    y=y['pct_chg'].tolist()
    del(y[0])
    # 获取X的行数和列数，shape[0]为行数
    X=X.drop(X.index[X.shape[0]-1])
    #3设置训练集跟测试集
    X_length = X.shape[0]
    split = int(X_length * 0.87)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    #4设置模型
    model = RandomForestClassifier(max_depth=4, n_estimators=10, min_samples_leaf=5, random_state=1)
    model.fit(X_train, y_train)

    #5预测股价涨跌，根据X_test给出的'close', 'vol', 'close-open', 'MA5'等数据进行预测第二天股价的涨跌情况
    y_pred = model.predict(X_test)
    #print(y_pred)
    a_6 = list(y_pred)
    b_6 = list(y_test)
    a_2 = []
    b_2 = []
    a_3 = []
    b_3 = []
    # print(a)

    # #6预测属于各个分类的概率
    # y_pred_proba = model.predict_proba(X_test)
    # b = pd.DataFrame(y_pred_proba, columns=['分类为-1的概率', '分类为1的概率'])
    # # print(b)

    # 7整体模型的预测准确度
    from sklearn.metrics import accuracy_score
    score2 = accuracy_score(y_pred, y_test)
    print(ts_code+' 准确率： ' + str(round(score2*100, 2)) + '%')

    acc2=0
    acc3=0
    acc6=0
    for i in range(len(a_6)-1):
        if a_6[i] == 5:
            a_3.append(2)
            a_2.append(1)
        elif a_6[i] == 4:
            a_3.append(2)
            a_2.append(1)
        elif a_6[i] == 3:
            a_3.append(1)
            a_2.append(1)
        elif a_6[i] == 2:
            a_3.append(1)
            a_2.append(0)
        elif a_6[i] == 1:
            a_3.append(0)
            a_2.append(0)
        else:
            a_3.append(0)
            a_2.append(0)

        if b_6[i] == 5:
            b_3.append(2)
            b_2.append(1)
        elif b_6[i] == 4:
            b_3.append(2)
            b_2.append(1)
        elif b_6[i] == 3:
            b_3.append(1)
            b_2.append(1)
        elif b_6[i] == 2:
            b_3.append(1)
            b_2.append(0)
        elif b_6[i] == 1:
            b_3.append(0)
            b_2.append(0)
        else:
            b_3.append(0)
            b_2.append(0)
        acc2=acc2+(a_2[i]==b_2[i])
        acc3 = acc3 + (a_3[i] == b_3[i])
        acc6 = acc6 + (a_6[i] == b_6[i])
    acc2=acc2/len(a_2)
    acc3 = acc3 / len(a_2)
    acc6 = acc6 / len(a_2)
    print(ts_code)
    print('二分类 ' + str(acc2))
    print('三分类 ' + str(acc3))
    print('六分类 ' + str(acc6))

    return acc6,acc3,acc2

# #8分析特征变量的特征重要性
# features = X.columns
# importances = model.feature_importances_
# b = pd.DataFrame()
# b['特征'] = features
# b['特征重要性'] = importances
# b = b.sort_values('特征重要性', ascending=False)
# print(b)
#
# #参数调优
# from sklearn.model_selection import GridSearchCV
# parameters = {'n_estimators':[5, 10, 20], 'max_depth':[2, 3, 4, 5], 'min_samples_leaf':[5, 10, 20, 30]}
# new_model = RandomForestClassifier(random_state=1)
# grid_search = GridSearchCV(new_model, parameters, cv=6, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# grid_search.best_params_
# print('最优模型参数： ' + str(grid_search.best_params_))

def get_train_data2():
    train_dir = args.train_dir
    df = open(train_dir)
    data_otrain = pd.read_csv(df)
    data_train = data_otrain.iloc[:, 1:]
    # print('训练集长度 ' + str(len(data_train)))
    train_x = data_train[['open', 'high', 'close', 'low', 'volume', 'turnover']]
    train_y = []
    # print(data_train.head(5))
    for index,row in data_train.iterrows():
        change = row['p_change']
        if change > 2:
            train_y.append(5)
        elif 1 < change <= 2:
            train_y.append(4)
        elif 0 < change <= 1:
            train_y.append(3)
        elif -1 < change <= 0:
            train_y.append(2)
        elif -2 < change <= -1:
            train_y.append(1)
        else:
            train_y.append(0)
    return train_x, train_y

def get_test_data2():
    test_dir = args.test_dir
    df = open(test_dir)
    data_otest = pd.read_csv(df)
    data_test = data_otest.iloc[:, 1:]
    print('测试集长度 ' + str(len(data_test)))
    test_x = data_test[['open', 'high', 'close', 'low', 'volume', 'turnover']]
    test_y = []
    for index,row in data_test.iterrows():
        change = row['p_change']
        if change > 2:
            test_y.append(5)
        elif 1 < change <= 2:
            test_y.append(4)
        elif 0 < change <= 1:
            test_y.append(3)
        elif -1 < change <= 0:
            test_y.append(2)
        elif -2 < change <= -1:
            test_y.append(1)
        else:
            test_y.append(0)
    return test_x, test_y

# 4设置模型
def train(train_x, train_y):
    model = RandomForestClassifier(max_depth=4, n_estimators=10, min_samples_leaf=5, random_state=1)
    model.fit(train_x, train_y)
    return model


# 5预测股价涨跌，根据X_test给出的'close', 'vol', 'close-open', 'MA5'等数据进行预测第二天股价的涨跌情况
def predict(model, test_x, test_y):
    pred = model.predict(test_x)
    # print(y_pred)
    a = pd.DataFrame()
    a['预测值'] = list(pred)
    a['实际值'] = list(test_y)
    print(a)
    from sklearn.metrics import accuracy_score
    score = accuracy_score(pred, test_y)
    print('准确率： ' + str(round(score * 100, 2)) + '%')


if __name__ == '__main__':
    bank_list=get_bank_list()
    sum2=0
    sum3=0
    sum6=0
    for i in bank_list:
        acc6, acc3, acc2=run(i)
        sum6=sum6+acc6
        sum3 = sum3 + acc3
        sum2 = sum2 + acc2
    print('6 avg: '+str(sum6/len(bank_list)))
    print('3 avg: ' + str(sum3 / len(bank_list)))
    print('2 avg: ' + str(sum2 / len(bank_list)))
    # train_x,train_y=get_train_data2()
    # test_x,test_y=get_test_data2()
    # model=train(train_x,train_y)
    # predict(model,test_x,test_y)
