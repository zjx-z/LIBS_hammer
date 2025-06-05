import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, GlobalMaxPooling1D,LSTM,Reshape
from tensorflow.keras.layers import Layer, Dense, Activation, Permute, Lambda
import tensorflow.keras.backend as K
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
# # 读取训练数据
# train_data = pd.read_csv('E:\libs\data1245.csv')
# train_features = train_data.iloc[:, :-1].values  # 假设特征在前7062列
# train_labels = train_data.iloc[:, -1].values  # 假设标签在最后一列
#
# # 读取测试数据
# test_data = pd.read_csv('E:\libs\data36.csv')
# test_features = test_data.iloc[:, :-1].values
# test_labels = test_data.iloc[:, -1].values
#
# # 转换为 NumPy 数组
# train_features = train_features.astype(np.float32)
# train_labels = train_labels.astype(np.int32)
# test_features = test_features.astype(np.float32)
# test_labels = test_labels.astype(np.int32)
#
# # 调整数据形状以适应模型输入
# train_features = train_features.reshape(-1, 7062, 1)
# test_features = test_features.reshape(-1, 7062, 1)
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[1:, :-1].values
    y = data.iloc[1:, -1:].values
    return X, y


csv_file = 'E:\libs\data1245-50.csv'
X, y = load_data(csv_file)


import numpy as np

def min_max_scale(X):
    # 对每一行数据进行 Min-Max Scaling
    X_min = np.min(X, axis=1, keepdims=True)  # 计算每一行的最小值
    X_max = np.max(X, axis=1, keepdims=True)  # 计算每一行的最大值
    X_scaled = (X - X_min) / (X_max - X_min)  # 应用 Min-Max Scaling 公式
    return X_scaled

# 应用 Min-Max Scaling
X_scaled = min_max_scale(X)


guiyihua="False"
if guiyihua=="False":
    X_train, X_val, y_train_, y_val_ = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_scaled = min_max_scale(X)
    X_train, X_val, y_train_, y_val_ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(y_train_[:100])


xuhao={'10':0,'100':1, '103':2, '104':3, '105':4, '106':5, '107':6, '110':7, '111':8, '116':9, '117':10, '12':11, '13':12, '14':13, '16':14, '2':15, '20':16, '24':17, '29':18, '3':19, '30':20, '32':21, '35':22, '36':23, '38':24, '39':25, '41':26, '42':27, '46':28, '48':29, '51':30, '53':31, '54':32, '55':33, '56':34, '63':35, '68':36, '69':37, '70':38, '71':39, '72':40, '73':41, '77':42, '8':43, '81':44, '84':45, '85':46, '9':47, '97':48, '98':49}
#xuhao={'0':0,'1':1,'10':2,'100':3, '103':4, '104':5, '105':6, '106':7, '107':8, '110':9, '111':10, '116':11, '117':12, '12':13, '13':14, '14':15, '16':16, '2':17, '20':18, '24':19, '29':20, '3':21, '30':22, '32':23, '35':24, '36':25, '38':26, '39':27, '41':28, '42':29, '43':30, '46':31, '48':32, '51':33, '53':34, '54':35, '55':36, '56':37, '63':38, '68':39, '69':40, '70':41, '71':42, '72':43, '73':44, '77':45, '8':46, '81':47, '84':48, '85':49, '9':50, '97':51, '98':52}
#print(X_train)
#print(y_train)
y_train=[[0 for j in range(52)] for i in range(len(y_train_))]
for i in range(len(y_train_)):
    y_train[i][xuhao[str(y_train_[i][0])]-1]=1
y_train = np.array(y_train, dtype=float)
y_val = [[0 for j in range(52)] for i in range(len(y_val_))]
for i in range(len(y_val_)):
    y_val[i][xuhao[str(y_val_[i][0])]-1] = 1
y_val = np.array(y_val, dtype=float)


import random as python_random
import tensorflow as tf
def set_random_seeds(seed_value):
    np.random.seed(seed_value)
    python_random.seed(seed_value)
    tf.random.set_seed(seed_value)

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


# 定义模型
def create_model(input_shape, num_labels):
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

seed_value = 42
set_random_seeds(seed_value)


model = create_model((7062,1), 52)
#model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
#model.fit(train_features, train_labels, epochs=10, batch_size=32, validation_data=(test_features, test_labels))
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_val, y_val), verbose=1)

# 使用 tf.GradientTape 计算梯度
with tf.GradientTape() as tape:
    # 将输入数据转换为 TensorFlow 张量
    input_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    tape.watch(input_tensor)  # 确保输入张量被跟踪
    predictions = model(input_tensor)
    loss = tf.keras.losses.categorical_crossentropy(y_train, predictions)
    #print("loss",loss)

# 计算输入数据的梯度
input_gradients = tape.gradient(loss, input_tensor)

# 检查梯度是否正确
#print(input_gradients[0])
# 将张量转换为 NumPy 数组
#input_gradients_numpy = input_gradients[0].numpy()
# 打印具体的数值
#print("具体的梯度数值：")
#for i in input_gradients_numpy:
#    print(i)

#print(list(y_train[1]).index(1)+1)

import matplotlib.pyplot as plt
# 绘制梯度值
# plt.figure(figsize=(15, 5))  # 设置图形大小
# plt.plot(input_gradients_numpy, label='Input Gradients')  # 绘制梯度值
# plt.title('Input Gradients for the 21th Sample')  # 设置标题
# plt.xlabel('Feature Index')  # 设置x轴标签
# plt.ylabel('Gradient Value')  # 设置y轴标签
# plt.legend()  # 添加图例
# plt.grid(True)  # 添加网格
# plt.show()


# 假设 X_train 和 input_gradients 已经准备好
sample_data = X_train[20].flatten()  # 获取第一个样本数据
sample_gradients = input_gradients[20].numpy().flatten()  # 获取第一个样本的梯度

# 缩放 input_gradients 数据，使其最大值与样本数据的最大值相等
max_sample_data = np.max(np.abs(sample_data))
max_gradients = np.max(np.abs(sample_gradients))
scaled_gradients = sample_gradients * (max_sample_data / max_gradients)
#scaled_sample_data=sample_data*(max_gradients/max_sample_data)
# 创建一个新的图形
plt.figure(figsize=(20, 5))

# 绘制样本数据（使用线条）
plt.plot(sample_data, label='Sample Data', color='blue', linewidth=1.5,alpha=0.3)
#plt.plot(scaled_sample_data, label='Scaled Sample Data', color='blue', linewidth=1.5)
# 绘制缩放后的 input_gradients（使用半透明背景）
plt.fill_between(range(len(scaled_gradients)), scaled_gradients, color='red', alpha=0.3, label='Scaled Input Gradients')
#plt.fill_between(range(len(sample_gradients)), sample_gradients, color='red', alpha=0.3, label='Input Gradients')

# 添加标题和标签
#plt.title('Sample Data and Scaled Input Gradients for the 1st Sample')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()

#
# # 假设这是你的原始列表
# original_list = input_gradients[20]
# # 获取排序后的索引
# sorted_indices = sorted(range(len(original_list)), key=lambda i: original_list[i], reverse=True)
# # 输出排序后的索引
# for i in range(len(sorted_indices)):
#     sorted_indices[i]+=1
# print("排序后的原始索引：", sorted_indices)


#print(tf.shape(input_gradients[0]))


# 计算每个输入变量的 SSD
ssd_values = tf.reduce_sum(tf.square(input_gradients), axis=0)

# 将 SSD 值转换为 NumPy 数组
ssd_values_numpy = ssd_values.numpy()

# 选择一个样本进行可视化
sample_index = 20  # 选择第21个样本
sample_data = X_train[sample_index].flatten()  # 获取样本数据
sample_ssds = ssd_values_numpy.flatten()  # 获取SSD值

# 缩放SSD值，使其最大值与样本数据的最大值相等
max_sample_data = np.max(np.abs(sample_data))
max_ssds = np.max(np.abs(sample_ssds))
scaled_ssds = sample_ssds * (max_sample_data / max_ssds)

# 创建一个新的图形
plt.figure(figsize=(20, 5))

# 绘制样本数据（使用线条）
plt.plot(sample_data, label='Sample Data', color='blue', linewidth=1.5, alpha=0.3)

# 绘制缩放后的SSD值作为背景
plt.fill_between(range(len(scaled_ssds)), 0, scaled_ssds, color='red', alpha=0.3, label='Scaled SSD Values')

# 添加标题和标签
#plt.title('Sample Data and Scaled SSD Values for the st Sample')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()


# 假设这是你的原始列表
original_list = scaled_ssds
# 获取排序后的索引
sorted_indices = sorted(range(len(original_list)), key=lambda i: original_list[i], reverse=True)
# 输出排序后的索引
for i in range(len(sorted_indices)):
    sorted_indices[i]+=1
print("排序后的原始索引：", sorted_indices)





# 定义中间模型
middle = Model(inputs=model.get_layer('conv1d').input, outputs=model.get_layer('flatten').output)

# 获取训练集第21个样本的原始数据和中间层输出
sample_index = 20  # 第21个样本
sample_data = X_train[sample_index].reshape(1, -1, 1)  # 调整形状以匹配模型输入
middle_output = middle.predict(sample_data)

# 获取原始数据和中间层输出的最大值
max_original = np.max(sample_data)
max_middle = np.max(middle_output)

# # 将中间层输出的最大值缩放到与原始数据的最大值相等
# scaled_middle_output = middle_output * (max_original / max_middle)
#
# 使用插值方法将中间层输出扩展到与原始数据相同的维度
original_length = sample_data.shape[1]  # 原始数据的长度
middle_length = middle_output.shape[1]  # 中间层输出的长度
#
# original_length = sample_data.shape[1]  # 原始数据的长度
# middle_length = scaled_middle_output.shape[1]  # 中间层输出的长度
# # 创建插值函数
# f = interp1d(np.linspace(0, original_length - 1, middle_length), scaled_middle_output.flatten(), kind='linear', fill_value="extrapolate")
#
# # 插值扩展到原始数据的长度
# interpolated_middle_output = f(np.arange(original_length))

f = interp1d(np.linspace(0, original_length - 1, middle_length), middle_output.flatten(), kind='linear', fill_value="extrapolate")
interpolated_middle_output = f(np.arange(original_length))

# 绘制原始数据
plt.figure(figsize=(20, 5))
plt.plot(sample_data.flatten(), label='Original Data', color='blue', linewidth=1.5, alpha=0.5)
#plt.title('Original Data for the 21st Sample')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 绘制插值后的中间层输出
plt.figure(figsize=(20, 5))
plt.plot(interpolated_middle_output, label='Interpolated Flatten Layer Output', color='red', linewidth=1.5, alpha=0.5)
#plt.title('Interpolated Flatten Layer Output for the 21st Sample')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# 定义中间模型
middle = Model(inputs=model.get_layer('conv1d').input, outputs=model.get_layer('conv1d').output)

# 获取训练集第21个样本的原始数据和中间层输出
sample_index = 20  # 第21个样本
sample_data = X_train[sample_index].reshape(1, -1, 1)  # 调整形状以匹配模型输入
middle_output = middle.predict(sample_data)

# 将中间层输出按照通道展平
flattened_middle_output = middle_output.flatten()

# 绘制原始数据
plt.figure(figsize=(20, 5))
plt.plot(sample_data.flatten(), label='Original Data', color='blue', linewidth=1.5, alpha=0.5)
plt.title('Original Data for the 21st Sample')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 绘制展平后的中间层输出
plt.figure(figsize=(20, 5))
plt.plot(flattened_middle_output, label='Flattened Conv1D Output', color='red', linewidth=1.5, alpha=0.5)
plt.title('Flattened Conv1D Output for the 21st Sample')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()