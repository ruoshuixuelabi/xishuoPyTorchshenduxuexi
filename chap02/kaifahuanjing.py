# 例2 1 查看Pytorch版本,查看Pytorch是否支持GPU运算。
import torch

print(torch.__version__)
print(torch.cuda.is_available())

# 【例2 2】  检查NumPy版本。
import numpy as np

print(np.__version__)

# 【例2 3】numpy.array应用举例。
import numpy as np

a = np.array([4, 5, 6, 7])
print(a)
print('*' * 20)
# 多于一个维度
b = np.array([[7, 2], [9, 4]])
print(b)
print('*' * 20)
# 最小维度
c = np.array([91, 72, 63, 74, 5], ndmin=2)
print(c)
print('*' * 20)
# dtype 参数
d = np.array([15, 26, 38], dtype=complex)
print(d)

# 【例2 4】numpy数据类型应用举例。
import numpy as np

# 使用标量类型
dt = np.dtype(np.int32)
print(dt)
print('*' * 10)
# int8, int16, int32, int64 四种数据类型可以使用字符串 'i1', 'i2','i4','i8' 代替
s2 = np.dtype('i4')
print(s2)
print('*' * 10)
# 字节顺序标注
s3 = np.dtype('<i4')
print(s3)
print('*' * 10)
# 首先创建结构化数据类型
s4 = np.dtype([('age', np.int8)])
print(s4)
print('*' * 10)
# 将数据类型应用于ndarray对象
w = np.dtype([('age', np.int8)])
s5 = np.array([(10,), (20,), (30,)], dtype=w)
print(s5)
print('*' * 10)
# 类型字段名可以用于存取实际的 age 列
y = np.dtype([('age', np.int8)])
s6 = np.array([(10,), (20,), (30,)], dtype=y)
print(s6['age'])
print('*' * 10)
student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
print(student)
print('*' * 10)
student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
s8 = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
print(s8)

# 【例2 5】numpy.empty创建空数组。
import numpy as np
s = np.empty([4, 6], dtype = int)
print(s)


# 【例2 6】numpy.zeros应用举例说明。
import numpy as np
# 默认为浮点数
x = np.zeros(5)
print(x)
# 设置类型为整数
y = np.zeros((5,), dtype=np.int)
print(y)
# 自定义类型
z = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
print(z)


# # 【例2 7】numpy.ones应用举例说明。
# import numpy as np
# # 默认为浮点数
# x = np.ones(5)
# print(x)
# print('*'*10)
# # 自定义类型
# x = np.ones([3, 3], dtype=int)
# print(x)

# # 【例2 8】将列表转换为ndarray。
# import numpy as np
# x = [4, 5, 6, 10000]
# a = np.asarray(x)
# print(a)


# # 【例2 9】  将元组转换为ndarray。
# import numpy as np
#
# x = (100, 2000, 300000)
# a = np.asarray(x)
# print(a)


# # 【例2 10】元组列表转换为ndarray。
# import numpy as np
# x = [(1, 2, 3), (4, 5)]
# a = np.asarray(x)
# print(a)
# print('*'*10)
# # 设置了dtype参数
# y =  [1,2,3]
# b = np.asarray(y, dtype =  float)
# print(b)


# # 【例2 11】numpy.frombuffer应用实例。
# import numpy as np
# s = b'Hello World'
# a = np.frombuffer(s, dtype='S1')
# print(a)


# # 【例2 12】numpy. fromiter应用实例。
# import numpy as np
# # 使用 range 函数创建列表对象
# list = range(10)
# it = iter(list)
# # 使用迭代器创建 ndarray
# x = np.fromiter(it, dtype=float)
# print(x)

# # 【例2 13】numpy加减乘除应用举例。
# import numpy as np
# a = np.arange(0, 27, 3, dtype=np.float_).reshape(3, 3)
# print('第一个数组：')
# print(a)
# print('*'*20)
# print('第二个数组：')
# b = np.array([3, 6, 9])
# print(b)
# print('*'*20)
# print('两个数组相加：')
# print(np.add(a, b))
# print('*'*20)
# print('两个数组相减：')
# print(np.subtract(a, b))
# print('*'*20)
# print('两个数组相乘：')
# print(np.multiply(a, b))
# print('*'*20)
# print('两个数组相除：')
# print(np.divide(a, b))


# # 【例2 14】numpy.reciprocal()应用举例。
# import numpy as np
# s = np.array([888, 1000, 20, 0.1])
# print('原数组是：')
# print(s)
# print('*'*20)
# print('调用reciprocal函数：')
# print(np.reciprocal(s))


# # 【例2 15】numpy.power()函数应用举例。
# import numpy as np
# s = np.array([2, 4, 8])
# print('原数组是；')
# print(s)
# print('*'*20)
# print('调用power函数：')
# print(np.power(s, 2))
# print('*'*20)
# print('power之后数组：')
# w = np.array([1, 2, 3])
# print(w)
# print('*'*20)
# print('再次调用power函数：')
# print(np.power(s, w))


# # 【例2 16】numpy.mod()应用举例。
# import numpy as np
# s = np.array([3, 6, 9])
# w = np.array([2, 4, 8])
# print('第一个数组：')
# print(s)
# print('*'*20)
# print('第二个数组：')
# print(w)
# print('*'*20)
# print('调用mod()函数：')
# print(np.mod(s, w))
# print('*'*20)
# print('调用remainder()函数：')
# print(np.remainder(s, w))


# # 【例2 17】numpy三角函数应用举例。
# import numpy as np
# a = np.array([0, 30, 45, 60, 90])
# print('不同角度的正弦值：')
# # 通过乘 pi/180 转化为弧度
# print(np.sin(a * np.pi / 180))
# print('*'*20)
# print('数组中角度的余弦值：')
# print(np.cos(a * np.pi / 180))
# print('*'*20)
# print('数组中角度的正切值：')
# print(np.tan(a * np.pi / 180))


# # 【例2 18】arcsin，arccos，和arctan函数应用举例。
# import numpy as np
# a = np.array([0, 30, 45, 60, 90])
# print('含有正弦值的数组：')
# sin = np.sin(a * np.pi / 180)
# print(sin)
# print('*'*20)
# print('计算角度的反正弦，返回值以弧度为单位：')
# inv = np.arcsin(sin)
# print(inv)
# print('*'*20)
# print('通过转化为角度制来检查结果：')
# print(np.degrees(inv))
# print('*'*20)
# print('arccos 和 arctan 函数行为类似：')
# cos = np.cos(a * np.pi / 180)
# print(cos)
# print('*'*20)
# print('反余弦：')
# inv = np.arccos(cos)
# print(inv)
# print('*'*20)
# print('角度制单位：')
# print(np.degrees(inv))
# print('*'*20)
# print('tan 函数：')
# tan = np.tan(a * np.pi / 180)
# print(tan)
# print('*'*20)
# print('反正切：')
# inv = np.arctan(tan)
# print(inv)
# print('*'*20)
# print('角度制单位：')
# print(np.degrees(inv))


# # 【例2 19】numpy.around()函数应用举例。
# import numpy as np
# a = np.array([100.0, 100.5, 123, 0.876, 76.998])
# print('原数组：')
# print(a)
# print('*'*20)
# print('舍入后：')
# print(np.around(a))
# print(np.around(a, decimals=1))
# print(np.around(a, decimals=-1))

# # 【例2 20】numpy.floor()应用举例。
# import numpy as np
# s = np.array([-9999.7, 100333.5, -23340.2, 0.987, 10.88888])
# print('提供的数组：')
# print(s)
# print('*'*20)
# print('修改后的数组：')
# print(np.floor(s))

# # 【例2 21】numpy.ceil()应用举例。
# import numpy as np
# s = np.array([-100.3, 18.98, -0.49999, 0.563, 10])
# print('提供的数组：')
# print(s)
# print('*'*20)
# print('修改后的数组：')
# print(np.ceil(s))


# # 【例2 73】  绘制简单图形对象。
# from matplotlib import pyplot as plt
# import numpy as np
# import math
# x = np.arange(0, math.pi*2, 0.05)
# y = np.sin(x)
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.plot(x,y)
# ax.set_title("sine wave")
# ax.set_xlabel('angle')
# ax.set_ylabel('sine')
# plt.show()


# # 【例2 74】  直线图展示销量关系。
# import matplotlib.pyplot as plt
# y = [1, 4, 9, 16, 25,36,49, 64]
# x1 = [1, 16, 30, 42,55, 68, 77,88]
# x2 = [1,6,12,18,28, 40, 52, 65]
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# #使用简写的形式color/标记符/线型
# l1 = ax.plot(x1,y,'ys-')
# l2 = ax.plot(x2,y,'go--')
# ax.legend(labels = ('tv', 'Smartphone'), loc = 'lower right') # legend placed at lower right
# ax.set_title("Advertisement effect on sales")
# ax.set_xlabel('medium')
# ax.set_ylabel('sales')
# plt.show()


# # 【例2 75】  新建的子图与现有的子图重叠。
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot([1,2,3])
# #现在创建一个子图，它表示一个有2行1列的网格的顶部图。
# #因为这个子图将与第一个重叠，所以之前创建的图将被删除
# plt.subplot(211)
# plt.plot(range(12))
# #创建带有黄色背景的第二个子图
# plt.subplot(212, facecolor='y')
# plt.plot(range(12))
# plt.show()


# # 【例2 76】  add_subplot()函数使用。
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot([1,2,3])
# #现在创建一个子图，它表示一个有2行1列的网格的顶部图。
# #因为这个子图将与第一个重叠，所以之前创建的图将被删除
# plt.subplot(211)
# plt.plot(range(12))
# #创建带有黄色背景的第二个子图
# plt.subplot(212, facecolor='y')
# plt.plot(range(12))
# plt.show()


# # 【例2 77】  通过给画布添加axes对象可以实现在同一画布中插入另外的图像。
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# x = np.arange(0, math.pi*2, 0.05)
# fig=plt.figure()
# axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
# axes2 = fig.add_axes([0.55, 0.55, 0.3, 0.3]) # inset axes
# y = np.sin(x)
# axes1.plot(x, y, 'b')
# axes2.plot(x,np.cos(x),'r')
# axes1.set_title('sine')
# axes2.set_title("cosine")
# plt.show()


# # 【例2 78】  创建了一个2行2列的子图，并在每个子图中显示4个不同的图像。
# import matplotlib.pyplot as plt
# fig,a =  plt.subplots(2,2)
# import numpy as np
# x = np.arange(1,5)
# #绘制平方函数
# a[0][0].plot(x,x*x)
# a[0][0].set_title('square')
# #绘制平方根图像
# a[0][1].plot(x,np.sqrt(x))
# a[0][1].set_title('square root')
# #绘制指数函数
# a[1][0].plot(x,np.exp(x))
# a[1][0].set_title('exp')
# #绘制对数函数
# a[1][1].plot(x,np.log10(x))
# a[1][1].set_title('log')
# plt.show()


# # 【例2 79】  显示坐标轴刻度。
# import matplotlib.pyplot as plt
# import numpy as np
# fig, axes = plt.subplots(1, 2, figsize=(10,4))
# x = np.arange(1,5)
# axes[0].plot( x, np.exp(x))
# axes[0].plot(x,x**2)
# axes[0].set_title("Normal scale")
# axes[1].plot (x, np.exp(x))
# axes[1].plot(x, x**2)
# #设置y轴
# axes[1].set_yscale("log")
# axes[1].set_title("Logarithmic scale (y)")
# axes[0].set_xlabel("x axis")
# axes[0].set_ylabel("y axis")
# axes[0].xaxis.labelpad = 10
# #设置x、y轴标签
# axes[1].set_xlabel("x axis")
# axes[1].set_ylabel("y axis")
# plt.show()

# # 【例2 80】  坐标轴颜色显示。
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# #为左侧轴，底部轴添加颜色
# ax.spines['bottom'].set_color('blue')
# ax.spines['left'].set_color('red')
# ax.spines['left'].set_linewidth(2)
# #将侧轴、顶部轴设置为None
# ax.spines['right'].set_color(None)
# ax.spines['top'].set_color(None)
# ax.plot([1,2,3,4,5])
# plt.show()


# # 【例2 81】  Matplotlib设置坐标轴。
# # 生成信号
# fs = 1000
# f = 10
# t = list(range(0, 1000))
# t = [x / fs for x in t]
# a = [math.sin(2 * math.pi * f * x) for x in t]
#
# # 作图
# plt.figure()
# plt.subplot(2, 2, 1)
# plt.plot(a)
# plt.title('Figure-1')
# plt.subplot(2, 2, 2)
# plt.plot(a)
# plt.xticks([])
# plt.title('Figure-2')
# plt.subplot(2, 2, 3)
# plt.plot(a)
# plt.yticks([])
# plt.title('Figure-3')
# plt.subplot(2, 2, 4)
# plt.plot(a)
# plt.axis('off')
# plt.title('Figure-4')
# plt.show()


# # 【例2 82】  刻度和标签的使用。
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# x = np.arange(0, math.pi*2, 0.05)
# #生成画布对象
# fig = plt.figure()
# #添加绘图区域
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# y = np.sin(x)
# ax.plot(x, y)
# #设置x轴标签
# ax.set_xlabel('angle')
# ax.set_title('sine')
# ax.set_xticks([0,2,4,6])
# #设置x轴刻度标签
# ax.set_xticklabels(['zero','two','four','six'])
# #设置y轴刻度
# ax.set_yticks([-1,0,1])
# plt.show()


# # 【例2 83】  grid()设置网格格式。
# import matplotlib.pyplot as plt
# import numpy as np
# #fig画布；axes子图区域
# fig, axes = plt.subplots(1,3, figsize = (12,4))
# x = np.arange(1,11)
# axes[0].plot(x, x**3, 'g',lw=2)
# #开启网格
# axes[0].grid(True)
# axes[0].set_title('default grid')
# axes[1].plot(x, np.exp(x), 'r')
# #设置网格的颜色，线型，线宽
# axes[1].grid(color='b', ls = '-.', lw = 0.25)
# axes[1].set_title('custom grid')
# axes[2].plot(x,x)
# axes[2].set_title('no grid')
# fig.tight_layout()
# plt.show()
