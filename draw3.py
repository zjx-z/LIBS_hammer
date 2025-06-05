import matplotlib.pyplot as plt
import numpy as np
# 读取数据文件
# 假设数据文件名为 'data.txt'，第一列为波长，第二列为强度
data = np.loadtxt('E:/2025-03-11/2025-03-11/20/23/160922U4_2.txt')

# 分离波长和强度数据
wavelength = data[:, 0]  # 第一列是波长
intensity = data[:, 1]   # 第二列是强度

# 创建图形和轴对象
fig, ax = plt.subplots()

# 绘制LIBS谱图
ax.plot(wavelength, intensity, label='LIBS Spectrum', color='blue',linewidth=0.3)  # 设置谱图颜色为蓝色

# 在指定波长处添加竖线
#specified_wavelengths = [239.441,241.608,259.905,260.701,261.19,263.14,273.952,274.663,274.899,275.55,324.692,327.333,334.393,351.397,352.348,356.514,356.57,356.9,358.001,361.768,361.822,394.298,396.052,404.601,407.183,427.182,430.807,432.55,432.601,438.352,440.485,455.402,472.183,472.224,481.011,493.348,510.508,515.248,521.765]  # 指定波长位置
specified_wavelengths=[324.692,327.333,510.508,515.248, 521.765,275.55,358.001,407.183,427.182,430.807,432.55,438.352,263.14,394.298,396.052,351.397,352.348,356.514, 361.822,239.441,241.608,334.393]

labels=["Cu","Cu","Cu","Cu","Cu","Fe","Fe","Fe","Fe","Fe","Fe","Fe","Fe","Al","Al","Ni","Ni","Ni","Ni","Ni","Ni","Zn"]
for i in range(len(specified_wavelengths)):
    wl=specified_wavelengths[i]
    word=labels[i]
    # 找到指定波长对应的强度值
    index = np.argmin(np.abs(wavelength - wl))
    intensity_value = intensity[index]

    # 绘制竖线，从波长对应的点往上延伸，长度适中
    if intensity_value>500:
        ax.plot([wl, wl], [intensity_value, intensity_value + 1000], color='red', linestyle='-', linewidth=1)
            #label=f'Wavelength {wl} nm' if wl == specified_wavelengths[0] else "")
        ax.text(wl, intensity_value + 1000, word, color='red', ha='center', va='bottom', fontproperties='Times New Roman',fontsize=9)
# 添加图例
#ax.legend()

# 添加标题和轴标签
#ax.set_title('LIBS Spectrum with Specified Wavelengths')
ax.tick_params(direction='in',which='both')
ax.set_xlabel('Wavelength/nm',fontproperties='Times New Roman',fontsize=9)
ax.set_ylabel('Intensity/arb.units',fontproperties='Times New Roman',fontsize=9)
ax.tick_params(axis='both',labelsize=9)


# 设置刻度标签的字体和字号
for label in ax.get_xticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(9)

for label in ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(9)



# 显示图形
plt.show()