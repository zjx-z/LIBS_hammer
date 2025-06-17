
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
# 加载数据
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[1:, :-1].values
    y = data.iloc[1:, -1:].values
    return X, y

# 构建模型

from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, GlobalMaxPooling1D,LSTM,Reshape
from tensorflow.keras.layers import Layer, Dense, Activation, Permute, Lambda
import tensorflow.keras.backend as K

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                  shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                  shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.exp(e)
        a = e / K.sum(e, axis=1, keepdims=True)
        output = x * a
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

import random as python_random
import tensorflow as tf
# 设置随机数种子
def set_random_seeds(seed_value):
    np.random.seed(seed_value)
    python_random.seed(seed_value)
    tf.random.set_seed(seed_value)


def build_model(input_shape, num_labels):
    model = Sequential()
    # Input layer
    model.add(Conv1D(filters=8, kernel_size=80, strides=20, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    # 添加自注意力层
    model.add(SelfAttention())
    model.add(Flatten())  # 或者使用 GlobalMaxPooling1D()
    model.add(Dense(num_labels, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model



def compute_rank_k(predictions, ground_truths, k=5):
    correct_count = 0
    total_count = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        top_k_predictions = pred[:k]  # 获取前 k 个预测结果
        if gt in top_k_predictions:
            correct_count += 1

    rank_k = correct_count / total_count
    return rank_k

# 主函数
def main():
    # 设置随机数种子
    seed_value = 42
    set_random_seeds(seed_value)


    csv_file = 'E:\libs\data1245-50.csv'
    X, y = load_data(csv_file)
    guiyihua="False"
    if guiyihua=="False":
        X_train, X_val, y_train_, y_val_ = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_val, y_train_, y_val_ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(y_train_[1],y_train_[2],y_train_[3])



    xuhao={'10':0,'100':1, '103':2, '104':3, '105':4, '106':5, '107':6, '110':7, '111':8, '116':9, '117':10, '12':11, '13':12, '14':13, '16':14, '2':15, '20':16, '24':17, '29':18, '3':19, '30':20, '32':21, '35':22, '36':23, '38':24, '39':25, '41':26, '42':27, '46':28, '48':29, '51':30, '53':31, '54':32, '55':33, '56':34, '63':35, '68':36, '69':37, '70':38, '71':39, '72':40, '73':41, '77':42, '8':43, '81':44, '84':45, '85':46, '9':47, '97':48, '98':49}
    #print(X_train)
    #print(y_train)
    y_train=[[0 for j in range(50)] for i in range(len(y_train_))]#标签是one-hot向量形式
    for i in range(len(y_train_)):
        #print(str(y_train_[i][0]))
        y_train[i][xuhao[str(y_train_[i][0])]]=1
    y_train = np.array(y_train, dtype=float)
    y_val = [[0 for j in range(50)] for i in range(len(y_val_))]
    for i in range(len(y_val_)):
        y_val[i][xuhao[str(y_val_[i][0])]] = 1
    y_val = np.array(y_val, dtype=float)

    #训练模型
    model = build_model((7062,1), 50)
    epoch=100
    model.fit(X_train, y_train, epochs=epoch, batch_size=8, validation_data=(X_val, y_val), verbose=1)


    #model = load_model("conv1d+sf-att 100epoch.h5")
    #获取
    #middle1 = Model(inputs=model.get_layer('conv1d').input, outputs=model.get_layer('conv1d').output)
    #middle2=Model(inputs=model.get_layer('conv1d').input, outputs=model.get_layer('self_attention').output)
    middle = Model(inputs=model.get_layer('conv1d').input, outputs=model.get_layer('flatten').output)



    # result=middle1.predict(X1)
    # for sp in range(len(result)):
    #     samp=result[sp]
    #     l=[]
    #     for sequence in samp:
    #         l.extend(sequence)
    #     plt.figure(figsize=(40,20),dpi=100)
    #     x=[h for h in range(2800)]
    #     plt.plot(x,l)
    #     plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    #     save_path="E:\libs\code\mycode\model-pic\\"+str(y1[sp][0])+"-"+str(sp)+"conv"+".jpg"
    #     plt.savefig(save_path)
    #
    # result = middle2.predict(X1)
    # for sp in range(len(result)):
    #     samp = result[sp]
    #     l = []
    #     for sequence in samp:
    #         l.extend(sequence)
    #     plt.figure(figsize=(40, 20), dpi=100)
    #     x = [h for h in range(2800)]
    #     plt.plot(x, l)
    #     plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    #     save_path = "E:\libs\code\mycode\model-pic\\"+str(y1[sp][0]) +"-"+ str(sp) + "self_att" + ".jpg"
    #     plt.savefig(save_path)


    #my_dict是用于给图片命名的
    my_dict = {
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "8": "4",
        "9": "5",
        "10": "6",
        "12": "7",
        "13": "8",
        "14": "9",
        "16": "10",
        "20": "11",
        "24": "12",
        "29": "13",
        "30": "14",
        "32": "15",
        "35": "16",
        "36": "17",
        "38": "18",
        "39": "19",
        "41": "20",
        "42": "21",
        "43": "22",
        "46": "23",
        "48": "24",
        "51": "25",
        "53": "26",
        "54": "27",
        "55": "28",
        "56": "29",
        "117": "30",
        "63": "31",
        "68": "32",
        "69": "33",
        "70": "34",
        "71": "35",
        "72": "36",
        "73": "37",
        "77": "38",
        "81": "39",
        "84": "40",
        "85": "41",
        "97": "42",
        "98": "43",
        "100": "44",
        "103": "45",
        "104": "46",
        "105": "47",
        "106": "48",
        "107": "49",
        "110": "50",
        "111": "51",
        "116": "52"
    }
    get_middle_data="False"
    if get_middle_data=="True":
        csv_file = 'E:\\libs\\data1-10-20.csv'
        X1, y1= load_data(csv_file)
        result = middle.predict(X1)
        for sp in range(len(result)):
            l = result[sp]
            plt.figure(figsize=(50, 20), dpi=100)
            x = [h for h in range(2800)]
            if sp%7==1:
                plt.plot(x, l,"bo-",linewidth=1,markersize=0.1)
                plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.15)
                plt.tick_params(axis='both', labelsize=60)
                plt.text(90, 0.95 * max(l), "#" + str(my_dict[str(y1[sp][0])])+"-"+str((sp%7)*3), fontsize=60, color="black")
                plt.xlabel('Dim1/arb.units', fontsize=60)  # 横坐标标签
                plt.ylabel('Dim2/arb.units', fontsize=60)  # 纵坐标标签
                save_path = "E:\libs\code\mycode\model_pic2\\"+str(y1[sp][0]) +"-"+ str(sp) + "flatten" + ".jpg"
                plt.savefig(save_path)


    # # 评估模型
    X_test,y_test_=load_data("E:\libs\data36-50.csv")

    y_test = [[0 for j in range(50)] for i in range(len(y_test_))]
    for i in range(len(y_test_)):
        y_test[i][xuhao[str(y_test_[i][0])]] = 1
    if guiyihua=="False":
        pass
    else:
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)
    y_pred = model.predict(X_test)
    acc = 0
    for i in range(len(y_test)):
        if np.argmax(y_test[i]) == np.argmax(y_pred[i]):
            acc += 1
    print("acc:", round(acc / len(y_test), 4))


    data = pd.read_csv("E:\libs\data143.csv")
    X_rank = data.iloc[:, :].values
    if guiyihua=="False":
        pass
    else:
        scaler = StandardScaler()
        X_rank = scaler.fit_transform(X_rank)
    y_pred = model.predict(X_rank)
    num=["1-1","103-1","1-20","43-1"]
    t=0
    for y_pred1 in y_pred:
        print(num[t])
        t+=1
        # 对列表进行降序排序，并获取索引
        sorted_enumerate = sorted(enumerate(y_pred1), key=lambda x: x[1], reverse=True)
        # 按照索引：值的格式输出
        for index, value in sorted_enumerate:
            print(f"{index}: {value}")





    #pd.DataFrame(y_test).to_excel('CARS_300sample_23wei -5label-actual_values.xlsx', index=False)
    #pd.DataFrame(y_pred).to_excel('CARS_300sample_23wei -5label-predicted_values.xlsx', index=False)
    #model.save('conv1d+sf-att 100epoch.h5')  # 保存模型到文件

if __name__ == '__main__':
    main()