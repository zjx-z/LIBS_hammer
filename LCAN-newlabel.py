import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Layer, Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow.keras.backend as K
import random as python_random
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
# 创建回调函数
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
# 加载多个数据文件
def load_data_from_files(file_list):
    data_frames = []
    for file in file_list:
        df = pd.read_csv(file)
        data_frames.append(df)
    combined_data = pd.concat(data_frames, ignore_index=True)
    X = combined_data.iloc[:, :-1].values
    y = combined_data.iloc[:, -1:].values
    return X, y

# 加载数据
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1:].values
    return X, y

# 自定义自注意力层
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

# 构建模型
def build_model(input_shape, num_labels):
    model = Sequential()
    # Input layer
    model.add(Conv1D(filters=8, kernel_size=80, strides=20, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    # 添加自注意力层
    #model.add(SelfAttention())
    model.add(Flatten())  # 或者使用 GlobalMaxPooling1D()
    #model.add(Dense(128, activation='relu'))  # 特征层
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    #model.summary()
    return model

# 计算 Rank 指标
def calculate_rank_metrics(y_true, y_pred):
    ranks = [1, 3, 5, 10,20]
    results = {}
    for rank in ranks:
        correct = 0
        for i in range(len(y_true)):
            top_k_indices = np.argsort(y_pred[i])[-rank:][::-1]  # 获取预测值最高的前 rank 个索引
            if np.argmax(y_true[i]) in top_k_indices:
                correct += 1
        results[f'Rank-{rank}'] = correct / len(y_true)
    return results

def set_random_seeds(seed_value):
    np.random.seed(seed_value)
    python_random.seed(seed_value)
    tf.random.set_seed(seed_value)

# 自定义对每个样本进行归一化
def normalize_samples(X):
    """
    对每个样本进行归一化，使得每个样本的均值为 0，标准差为 1。
    """
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        mean = np.mean(X[i, :])
        std = np.std(X[i, :])
        X_normalized[i, :] = (X[i, :] - mean) / std
    return X_normalized

from sklearn.metrics import average_precision_score

def calculate_per_class_ap(y_true, y_pred):
    """
    计算每个类别的 Average Precision (AP)
    :param y_true: 真实标签 (one-hot 编码)
    :param y_pred: 预测概率
    :return: 每个类别的 AP 列表
    """
    n_classes = y_true.shape[1]
    aps = []
    for c in range(n_classes):
        ap = average_precision_score(y_true[:, c], y_pred[:, c])
        if not np.isnan(ap):  # 如果某类没有正样本，AP 会是 NaN，跳过
            aps.append(ap)
    return aps

def calculate_map(aps):
    """
    计算 mAP
    :param aps: 每个类别的 AP 列表
    :return: mAP
    """
    return np.mean(aps) if aps else 0.0


# 主函数
def main():
    # 设置随机数种子
    seed_value = 40
    set_random_seeds(seed_value)

    csv_file = [
    #     'E:\\libs\\20250311.csv',
    #     'E:\\libs\\0626-2-51.csv',
    #     'E:\\libs\\20250702-1.csv',
    #     'E:\\libs\\20250702-2.csv',
    #     'E:\\libs\\20250703.csv',
    #     'E:\\libs\\20250704-1.csv',
        #'E:\\libs\\20250704-2.csv',
                #'E:\libs\code\mycode\select_data\inrangedata.csv',
                #"E:\\libs\\sample_mix_lunci_train.csv",
        #'E:\libs\iron-hammer - newlabel2\\20250311.csv',
        # 'E:\libs\iron-hammer - newlabel2\\0626-2.csv',
        # 'E:\libs\iron-hammer - newlabel2\\20250702-1.csv',
        # 'E:\libs\iron-hammer - newlabel2\\20250702-2.csv',
        # 'E:\libs\iron-hammer - newlabel2\\20250703.csv',
        # 'E:\libs\iron-hammer - newlabel2\\20250704-1.csv',
        # 'E:\libs\iron-hammer - newlabel2\\20250704-2.csv'
        # 'E:\libs\hammer - newlabel3\\20250311.csv',
        # 'E:\libs\hammer - newlabel3\\0626-2.csv',
        # 'E:\libs\hammer - newlabel3\\20250702-1.csv',
        # 'E:\libs\hammer - newlabel3\\20250702-2.csv',
        # 'E:\libs\hammer - newlabel3\\20250703.csv',
        # #'E:\libs\hammer - newlabel3\\20250704-1.csv',
        # 'E:\libs\hammer - newlabel3\\20250704-2.csv',
        #'E:\libs\hammer-newlabel4\\20250311.csv',
        #'E:\libs\hammer-newlabel4\\0626-2.csv',
        'E:\libs\hammer-newlabel4\\20250702-1.csv',
        'E:\libs\hammer-newlabel4\\20250702-2.csv',
        'E:\libs\hammer-newlabel4\\20250703.csv',
        'E:\libs\hammer-newlabel4\\20250704-1.csv',
        #'E:\libs\hammer-newlabel4\\20250704-2.csv',
        #'E:\libs\hammer-newlabel4\\20250702-2 - COPY.csv',
                ]
    X, y = load_data_from_files(csv_file)

    guiyihua = "False"

    if guiyihua == "False":
        X_train, X_val, y_train_, y_val_ = train_test_split(X, y, test_size=0.2, random_state=seed_value)
    else:
        #scaler = StandardScaler()
        #X_scaled = scaler.fit_transform(X)
        X_scaled=normalize_samples(X)
        X_train, X_val, y_train_, y_val_ = train_test_split(X_scaled, y, test_size=0.2, random_state=50)

    import numpy as np
    y_train = [[0 for j in range(17)] for i in range(len(y_train_))]
    for i in range(len(y_train_)):
        y_train[i][int(y_train_[i])] = 1
    y_val = [[0 for j in range(17)] for i in range(len(y_val_))]
    y_train = np.array(y_train, dtype=float)
    for i in range(len(y_val_)):
        y_val[i][int(y_val_[i])] = 1
    y_val = np.array(y_val, dtype=float)

    # xuhao = {'1': 1, '10': 2, '100': 3, '103': 4, '104': 5, '105': 6, '106': 7, '107': 8, '110': 9, '111': 10,
    #          '116': 11, '117': 12, '12': 13, '13': 14, '14': 15, '16': 16, '2': 17, '20': 18, '24': 19, '29': 20,
    #          '3': 21, '30': 22, '32': 23, '35': 24, '36': 25, '38': 26, '39': 27, '41': 28, '42': 29, '43': 30,
    #          '46': 31, '48': 32, '51': 33, '53': 34, '54': 35, '55': 36, '56': 37, '63': 38, '68': 39, '69': 40,
    #          '70': 41, '71': 42, '72': 43, '73': 44, '77': 45, '8': 46, '81': 47, '84': 48, '85': 49, '9': 50, '97': 51,
    #          '98': 52}
    #
    # y_train = [[0 for j in range(52)] for i in range(len(y_train_))]
    # for i in range(len(y_train_)):
    #     y_train[i][xuhao[str(y_train_[i][0])]-1] = 1
    # y_train = np.array(y_train, dtype=float)
    #
    # y_val = [[0 for j in range(52)] for i in range(len(y_val_))]
    # for i in range(len(y_val_)):
    #     y_val[i][xuhao[str(y_val_[i][0])]-1] = 1
    # y_val = np.array(y_val, dtype=float)



    #model = build_model((7062, 1), 52)
    model = build_model((7062, 1), 17)
    #model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_val, y_val), verbose=1)
    model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_val, y_val), callbacks=[reduce_lr])
    model.summary()
    # 测试集评估
    X_test, y_test_ = load_data('E:\libs\hammer-newlabel4\\20250704-2.csv')

    if guiyihua == "False":
        pass
    else:
        # scaler = StandardScaler()
        # X_test = scaler.fit_transform(X_test)
        X_test = normalize_samples(X_test)

    # y_test = [[0 for j in range(52)] for i in range(len(y_test_))]
    # for i in range(len(y_test_)):
    #     y_test[i][xuhao[str(y_test_[i][0])]-1] = 1
    # y_test = np.array(y_test, dtype=float)

    y_test = [[0 for j in range(17)] for i in range(len(y_test_))]
    for i in range(len(y_test_)):
        y_test[i][int(y_test_[i])] = 1
    y_test = np.array(y_test, dtype=float)

    y_pred = model.predict(X_test)
    # for i in range(len(y_test)):
    #     print(np.argmax(y_test[i]),np.argmax(y_pred[i]))

    acc = 0
    for i in range(len(y_test)):
        if np.argmax(y_test[i]) == np.argmax(y_pred[i]):
            acc += 1
    print("acc:", round(acc / len(y_test), 4))

    #model.save_weights("model1.h5")

    #
    # # 计算 Rank 指标
    # rank_metrics = calculate_rank_metrics(y_test, y_pred)
    # for rank, value in rank_metrics.items():
    #     #print(f"{rank}: {value:.4f}")
    #     print(f"{value:.4f}")
    #     pass
    #
    # # 计算 per-class AP
    # aps = calculate_per_class_ap(y_test, y_pred)
    # mAP = calculate_map(aps)
    #
    # # 打印每个类别的 AP 和 mAP
    # for c, ap in enumerate(aps):
    #     #print(f"Class {c + 1} AP: {ap:.4f}")
    #     pass
    # print(f"mAP: {mAP:.4f}")

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 获取真实标签和预测标签
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(conf_matrix)

    plt.rcParams['font.family'] = 'Times New Roman'
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=False, yticklabels=False)
    plt.xlabel('Predicted Labels', labelpad=40)
    plt.ylabel('True Labels', labelpad=20)
    plt.title('Confusion Matrix-LCAN-Dataset7')
    #plt.savefig('Confusion Matrix-LCAN-Dataset7.png', dpi=1200)  # 保存图像，设置分辨率为 600 dpi
    #plt.show()
    #



if __name__ == '__main__':
    main()
    # 程序运行结束，发出声音
    import winsound
    winsound.Beep(2500, 1000)  # 频率 2500 Hz，持续时间 1000 毫秒

