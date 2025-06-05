import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = 'data.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的工作表名称
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设要读取的两列数据的列名分别为 'Column1' 和 'Column2'
# 如果列名不同，请根据实际情况修改
column1 = 'Column1'
column2 = 'Column2'

# 检查数据是否正确读取
print(data[[column1, column2]].head())

# 绘制密度图
plt.figure(figsize=(10, 6))  # 设置图像大小
sns.kdeplot(data[column1], label="Inter Distance", fill=True)  # 绘制第一列的密度图
sns.kdeplot(data[column2], label="Intra Distance", fill=True)  # 绘制第二列的密度图

# 添加图例
#plt.legend()

# 添加标题和坐标轴标签
#plt.title('Density Plot of Inter Distance and Intra Distrance',fontsize=20)
plt.tick_params(direction='in',which='both')
plt.xlabel('Distance/arb.units', fontproperties='Times New Roman',fontsize=15)
plt.ylabel('Density/arb.units', fontproperties='Times New Roman',fontsize=15)
plt.tick_params(axis='both', labelsize=15)
#plt.savefig("E:\libs\code\mycode\midutu\\midutu.JPEG",dpi=600, bbox_inches='tight')  # 保存图像

# 设置刻度标签的字体和字号
plt.xticks(fontname='Times New Roman', fontsize=15)
plt.yticks(fontname='Times New Roman', fontsize=15)
# 显示图像
plt.show()
