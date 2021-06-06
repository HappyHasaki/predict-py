import tensorflow as tf
import datetime
from sklearn.metrics import classification_report

from dataprocess import *

# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置


weights = {
    'in': tf.Variable(tf.random_normal([args.input_size, args.rnn_unit])),
    'out': tf.Variable(tf.random_normal([args.rnn_unit, args.output_size]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[args.rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(X):
   #with tf.name_scope('rnn'), tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
    #X.shape [len,time_steps,input_size]
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    # [7,128]
    b_in = biases['in']
    # [128]
    input = tf.reshape(X, [-1, args.input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    # [？,7]
    input_rnn = tf.matmul(input, w_in) + b_in
    # [？,128]
    input_rnn = tf.reshape(input_rnn, [-1, time_step, args.rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    print(input_rnn.shape)
    # [？,20,128]
    cell = tf.nn.rnn_cell.BasicRNNCell(args.rnn_unit)
    mlstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell)  # 设置dropout
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([mlstm_cell,mlstm_cell,mlstm_cell], state_is_tuple=True)
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.nn.dynamic_rnn(mlstm_cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = output_rnn[:, -1, :]
    output = tf.reshape(output, [-1, args.rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# -----------------------训练模型------------------------------------ #
# 用于训练模型，val（验证集）默认为开启
def train_lstm(time_step=args.time_step, val=True):
    X = tf.placeholder(tf.float32, shape=[None, time_step, args.input_size])
    Y = tf.placeholder(tf.float32, shape=[None, 1, args.output_size])
    batch_index, val_index, train_x, train_y, val_x, val_y = get_train_data()
    print('trian_y:{}, val_y:{}'.format(np.shape(train_y), np.shape(val_y)))
    # [len,1,6] 6分类标签
    pred, _ = lstm(X)
    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    train_op = tf.train.AdamOptimizer(args.lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.scalar('loss', loss)
        merged_summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(args.train_graph_dir, sess.graph)
        valid_writer = tf.summary.FileWriter(args.val_graph_dir)
        min_loss = 50
        # 训练.次
        print('开始训练...')
        for i in range(args.epoch):
            for j in range(len(batch_index) - 1):
                summary_str, _, loss_ = sess.run([merged_summary_op, train_op, loss],
                                                 feed_dict={X: train_x[batch_index[j]:batch_index[j + 1]],
                                                            Y: train_y[batch_index[j]:batch_index[j + 1]]})
            if val:
                for j in range(len(val_index) - 1):
                    valid_str, loss_val = sess.run([merged_summary_op, loss],
                                                   feed_dict={X: val_x[val_index[j]:val_index[j + 1]],
                                                              Y: val_y[val_index[j]:val_index[j + 1]]})
            # if i % 10 == 0:
            print("------------------------------------------------------")
            print('epoch: {}, train_loss: {:.4f}, Val_Loss: {:.4f}'.format(i + 1, loss_, loss_val))
            train_writer.add_summary(summary_str, i)
            valid_writer.add_summary(valid_str, i)
            if loss_val < min_loss:
                min_loss = loss_val
                print('loss '+str(min_loss))
                print("保存模型：", saver.save(sess, args.train_model_dir + args.model_name))


# -----------------------fine-tuning训练模型------------------------- #
# 用于微调模型，如新增数据在原模型继续训练，迁移学习等
# 其迭代次数不应过大，避免过拟合
def fining_tune_train(time_step=args.time_step):
    X = tf.placeholder(tf.float32, shape=[None, time_step, args.input_size])
    Y = tf.placeholder(tf.float32, shape=[None, 1, args.output_size])
    train_x, train_y = get_update_data()
    pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    train_op = tf.train.AdamOptimizer(args.lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    min_loss = 1000
    with tf.Session() as sess:
        # 参数恢复
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.train_model_dir + args.model_name)
        print("模型加载完毕")
        for i in range(args.epoch_fining):
            if len(train_x) < 10000:
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
                if i % 10 == 0:
                    print("------------------------------------------------------")
                    print('epoch: {}, train_loss: {:.4f}'.format(i + 1, loss_))
                if loss_ < min_loss:
                    min_loss = loss_
                    print("保存模型：", saver.save(sess, args.fining_turn_model_dir + args.model_name_ft))
            else:
                b_z = args.batch_size
                for j in range(len(train_x) // b_z + 1):
                    _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[b_z * j:b_z * (j + 1)],
                                                                     Y: train_y[b_z * j:b_z * (j + 1)]})
                if i % 10 == 0:
                    print("------------------------------------------------------")
                    print('epoch: {}, train_loss: {:.4f}'.format(i + 1, loss_))
                if loss_ < min_loss:
                    min_loss = loss_
                    print("保存模型：", saver.save(sess, args.fining_turn_model_dir + args.model_name_ft))


# -----------------------------测试模型------------------------------ #
# 用于测量测试集的准确率和F1
def test(time_step=args.time_step):
    X = tf.placeholder(tf.float32, shape=[None, time_step, args.input_size])
    test_x, test_y = get_test_data()
    print("---------数据加载完毕--------")
    pred, _ = lstm(X)
    pre_dict = []
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        saver.restore(sess, args.train_model_dir + args.model_name)
        print("----------模型加载完毕----------")
        if len(test_x) < 15000:
            prob = sess.run(pred, feed_dict={X: test_x})
            pre_dict.extend(prob)
        else:
            for i in range(len(test_x) // args.batch_size + 1):
                prob = sess.run(pred, feed_dict={X: test_x[args.batch_size * i:args.batch_size * (i + 1)]})
                pre_dict.extend(prob)
        pre_dict = np.array(pre_dict)
        test_label = np.array(test_y)
        a1 = list(np.argmax(pre_dict, 1))
        print(a1)
        a2 = list(np.argmax(test_label, 1))
        print(a2)
        a1_2=[]
        a1_3= []
        a2_2=[]
        a2_3 = []
        # 2分类编码
        for i in a1:
            if i >= 4:
                a1_3.append(2)
            elif i >=2:
                a1_3.append(1)
            else:
                a1_3.append(0)

            if i >= 3:
                a1_2.append(1)
            else:
                a1_2.append(0)
        for i in a2:
            if i >= 4:
                a2_3.append(2)
            elif i >= 2:
                a2_3.append(1)
            else:
                a2_3.append(0)

            if i >= 3:
                a2_2.append(1)
            else:
                a2_2.append(0)
        # 二分类准确率
        cp_2 = tf.equal(a1_2, a2_2)
        # 三分类准确率
        cp_3 = tf.equal(a1_3, a2_3)
        # 六分类准确率
        cp_6 = tf.equal(a1, a2)
        acc_2 = tf.reduce_mean(tf.cast(cp_2, tf.float32)).eval()
        acc_3 = tf.reduce_mean(tf.cast(cp_3, tf.float32)).eval()
        acc_6 = tf.reduce_mean(tf.cast(cp_6, tf.float32)).eval()
        # print(classification_report(a1, a2))
        print('二分类 ' + str(acc_2))
        print('三分类 ' + str(acc_3))
        print('六分类 ' + str(acc_6))


# -----------------------------预测模型------------------------------ #
# 用于预测第二天的收盘价
def predict(time_step=args.time_step):
    X = tf.placeholder(tf.float32, shape=[None, time_step, args.input_size])
    pre_x, code = get_predict_data(args.predict_dir)
    print("---------数据加载完毕--------")
    pred, _ = lstm(X)
    pre_y = []
    saver = tf.train.Saver(tf.variable_scope('lstm'))
    with tf.Session() as sess:
        # 参数恢复
        saver.restore(sess, args.train_model_dir + args.model_name)
        print("----------模型加载完毕----------")
        if len(pre_x) < 15000:
            prob = sess.run(pred, feed_dict={X: pre_x})
            pre_y.extend(prob)
        else:
            for i in range(len(pre_x) // args.batch_size + 1):
                prob = sess.run(pred, feed_dict={X: pre_x[args.batch_size * i:args.batch_size * (i + 1)]})
                pre_y.extend(prob)
        pre_y = np.array(pre_y)
        a = list(np.argmax(pre_y, 1))
        if a[0] == 0:
            print("模型{}预测股票{}明天的收盘价跌2%及以上".format(args.model_name, code))
        elif a[0] == 1:
            print("模型{}预测股票{}明天的收盘价跌1%-2%".format(args.model_name, code))
        elif a[0] == 2:
            print("模型{}预测股票{}明天的收盘价跌1%以内".format(args.model_name, code))
        elif a[0] == 3:
            print("模型{}预测股票{}明天的收盘价涨1%以内".format(args.model_name, code))
        elif a[0] == 4:
            print("模型{}预测股票{}明天的收盘价涨1%-2%".format(args.model_name, code))
        else:
            print("模型{}预测股票{}明天的收盘价涨2%以上".format(args.model_name, code))
    return a[0]

def predictAll(time_step=args.time_step):
    stock_list_dir = args.stock_list_dir
    df = open(stock_list_dir)
    stock_list = pd.read_csv(df)
    stock_list.drop([0])
    result_list=[]
    code_list=[]

    X = tf.placeholder(tf.float32, shape=[None, time_step, args.input_size])
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        saver.restore(sess, args.train_model_dir + args.model_name)
        for index,row in stock_list.iterrows():
            print(index)
            code=row['ts_code'][:6]
            end_time=datetime.datetime.today()-datetime.timedelta(days=1)
            start_time=end_time-datetime.timedelta(days=30)
            get_stock_data(code, str(start_time), str(end_time), 'd:\data\\everyday')

            pre_x, code = get_predict_data('d:\data\\everyday\\'+code+'.csv')
            if len(pre_x[0]) < 20 :
                continue
            pre_y = []
            if len(pre_x) < 15000:
                prob = sess.run(pred, feed_dict={X: pre_x})
                pre_y.extend(prob)
            else:
                for i in range(len(pre_x) // args.batch_size + 1):
                    prob = sess.run(pred, feed_dict={X: pre_x[args.batch_size * i:args.batch_size * (i + 1)]})
                    pre_y.extend(prob)
            pre_y = np.array(pre_y)
            a = list(np.argmax(pre_y, 1))
            result_list.append(a[0])
            code_list.append(code)
            if(index%10==0):
                print('write to mysql...')
                addToMysql(code_list=code_list, predict_list=result_list)
                code_list=[]
                result_list=[]
    print('success!')
