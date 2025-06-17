import pandas as pd
from keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, GlobalMaxPooling1D,LSTM,Reshape
from tensorflow.keras.layers import Layer, Dense, Activation, Permute, Lambda
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


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
    model.add(Flatten())  # 或者使用 GlobalMaxPooling1D()，但效果不好

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







# 定义中间模型
middle = Model(inputs=model.get_layer('conv1d').input, outputs=model.get_layer('flatten').output)
# 获取训练集第21个样本的原始数据和中间层输出
sample_index = 20  # 第21个样本
sample_data = X_train[sample_index].reshape(1, -1, 1)  # 调整形状以匹配模型输入
middle_output = middle.predict(sample_data)
# 绘制原始数据
plt.figure(figsize=(10, 5))
plt.plot(sample_data.flatten(), label='Original Data', color='blue', linewidth=0.5, alpha=0.5)
plt.tick_params(direction='in',which='both')
#plt.title('Original Data for the 21st Sample')
plt.xlabel('Wavelength/nm', fontproperties='Times New Roman',fontsize=15)
plt.ylabel('Intensity/arb.units', fontproperties='Times New Roman',fontsize=15)
#plt.legend()
#plt.grid(True)
# 设置刻度标签的字体和字号
plt.xticks(fontname='Times New Roman', fontsize=15)
plt.yticks(fontname='Times New Roman', fontsize=15)
# 在左上角添加文本 (a)
plt.text(0.05, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=15, fontname='Times New Roman', va='top', ha='left')
plt.show()
# 绘制插值后的中间层输出
plt.figure(figsize=(10, 5))
#plt.plot(interpolated_middle_output, label='Interpolated Flatten Layer Output', color='red', linewidth=1.5, alpha=0.5)
plt.tick_params(direction='in',which='both')
plt.plot(middle_output.flatten(),label='Flatten Layer Output', color='red', linewidth=0.5, alpha=0.5)
#plt.title('Interpolated Flatten Layer Output for the 21st Sample')
plt.xlabel('Feature index/arb.units', fontproperties='Times New Roman',fontsize=15)
plt.ylabel('Value/arb.units', fontproperties='Times New Roman',fontsize=15)
#plt.legend()
#plt.grid(True)
# 设置刻度标签的字体和字号
plt.xticks(fontname='Times New Roman', fontsize=15)
plt.yticks(fontname='Times New Roman', fontsize=15)
# 在左上角添加文本 (a)
plt.text(0.05, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=15, fontname='Times New Roman', va='top', ha='left')
plt.show()



# 定义中间模型
middle = Model(inputs=model.get_layer('conv1d').input, outputs=model.get_layer('conv1d').output)
# 获取训练集第21个样本的原始数据和中间层输出
sample_index = 20  # 第21个样本
sample_data = X_train[sample_index].reshape(1, -1, 1)  # 调整形状以匹配模型输入
middle_output = middle.predict(sample_data)
# 将中间层输出按照通道展平
flattened_middle_output = middle_output.flatten()

# 绘制原始数据
plt.figure(figsize=(10, 5))
plt.plot(sample_data.flatten(), label='Original Data', color='blue', linewidth=0.5, alpha=0.5)
#plt.title('Original Data for the 21st Sample')
plt.tick_params(direction='in',which='both')
plt.xlabel('Wavelength/nm', fontproperties='Times New Roman',fontsize=15)
plt.ylabel('Intensity/arb.units', fontproperties='Times New Roman',fontsize=15)
#plt.legend()
#plt.grid(True)
# 设置刻度标签的字体和字号
plt.xticks(fontname='Times New Roman', fontsize=15)
plt.yticks(fontname='Times New Roman', fontsize=15)
# 在左上角添加文本 (a)
plt.text(0.05, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=15, fontname='Times New Roman', va='top', ha='left')
plt.show()
# 绘制展平后的中间层输出
plt.figure(figsize=(10, 5))
plt.plot(flattened_middle_output, label='Flattened Conv1D Output', color='red', linewidth=0.5, alpha=0.5)
#plt.title('Flattened Conv1D Output for the 21st Sample')
plt.tick_params(direction='in',which='both')
plt.xlabel('Feature index/arb.units', fontproperties='Times New Roman',fontsize=15)
plt.ylabel('Value/arb.units', fontproperties='Times New Roman',fontsize=15)
#plt.legend()
#plt.grid(True)
# 设置刻度标签的字体和字号
plt.xticks(fontname='Times New Roman', fontsize=15)
plt.yticks(fontname='Times New Roman', fontsize=15)
# 在左上角添加文本 (a)
plt.text(0.05, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=15, fontname='Times New Roman', va='top', ha='left')
plt.show()