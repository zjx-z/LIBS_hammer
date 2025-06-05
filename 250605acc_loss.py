import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取 Excel 文件
file_path = 'model1.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path, header=None).values

# 提取数据
train_accuracy = data[0, :100]  # 第一行，前100个数据
train_loss = data[1, :100]      # 第二行，前100个数据
val_accuracy = data[2, :100]    # 第三行，前100个数据
val_loss = data[3, :100]        # 第四行，前100个数据

# 创建图形和轴对象
fig, ax1 = plt.subplots()

# 设置字体和字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10





# 绘制准确率曲线
ax1.set_xlabel('Epoch/arb.units', fontname='Times New Roman')#, fontsize=10

#ax1.set_ylabel('Accuracy', fontname='Times New Roman', fontsize=6, color='tab:blue')
ax1.set_ylabel('Accuracy/arb.units', fontname='Times New Roman')#, fontsize=10
ax1.plot(range(1, 101), train_accuracy, linestyle="-",label='Train Accuracy', color='tab:blue', linewidth=0.5)
ax1.plot(range(1, 101), val_accuracy, linestyle="--",label='Validation Accuracy', color='tab:orange', linewidth=0.5)
ax1.tick_params(axis='x', direction='in',labelsize=6, labelcolor='black')
#ax1.tick_params(axis='y', direction='in', labelcolor='tab:blue')
ax1.tick_params(axis='y', direction='in', labelsize=6, labelcolor='black')
# 获取当前的刻度位置
xticks = ax1.get_xticks()
ax1.set_xticklabels([f'{tick:.1f}' for tick in xticks], fontname='Times New Roman', fontsize=10)#, fontsize=10
yticks1 = ax1.get_yticks()
ax1.set_yticklabels([f'{tick:.1f}' for tick in yticks1], fontname='Times New Roman', fontsize=10)#, fontsize=10

# 创建第二个 y 轴
ax2 = ax1.twinx()
#ax2.set_ylabel('Loss', fontname='Times New Roman', fontsize=6, color='tab:green')
ax2.set_ylabel('Loss/arb.units', fontname='Times New Roman')#, fontsize=10
ax2.plot(range(1, 101), train_loss, linestyle="-.",label='Train Loss', color='tab:green', linewidth=0.5)
ax2.plot(range(1, 101), val_loss, linestyle=":",label='Validation Loss', color='tab:red', linewidth=0.5)
#ax2.tick_params(axis='y', direction='in', labelcolor='tab:green')
ax2.tick_params(axis='y', direction='in',labelsize=6, labelcolor='black')

yticks2 = ax2.get_yticks()
ax2.set_yticklabels([f'{tick:.1f}' for tick in yticks2], fontname='Times New Roman', fontsize=10)#, fontsize=10




# 添加图例
ax1.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.15, 0.5))#fontsize=10,
ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.95, 0.5))#fontsize=10,

# 添加边框
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)

# 设置背景为透明
fig.patch.set_facecolor('none')
ax1.set_facecolor('none')
ax2.set_facecolor('none')


# 显示图形
plt.show()