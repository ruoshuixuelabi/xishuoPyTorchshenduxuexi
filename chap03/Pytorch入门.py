# 【例3 1】  Numpy数据转换成Pytorch张量。
# Numpy数据转换成Pytorch张量。
import numpy as np
import torch

print('#' * 20)
print('python列表转换成Pytorch张量:')
a = [1, 2, 3, 4]
print(torch.tensor(a))
print(torch.tensor(a).shape)
print('#' * 20)
print('查看张量数据类型')
print(torch.tensor(a).dtype)
print('#' * 20)
print('指定张量数据类型')
b = torch.tensor(a, dtype=float)
print(b.dtype)
print('#' * 20)
print('numpy数据转换为张量')
c = np.array([1, 2, 3, 4])
print(c.dtype)
d = torch.tensor(c)
print(d.dtype)
print('#' * 20)
print('列表嵌套创建张量')
e = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(e.shape)
print(e)
print('#' * 20)
print('从torch.float转换到torch.int')
f = torch.randn(3, 3)
g = f.to(torch.int)
print(g.dtype)

# 【例3 2】  torch.tensor函数生成张量。
# torch.tensor函数生成张量。
a = torch.tensor([[2, 3], [100, 999], [8888, 9.999]], dtype=torch.float32)
print(a)
print(a.shape)
print(a.size(0))
print(a.size(1))
print(a.shape[1])
# torch.save()
torch.save(a, 'file.pt')
# 【例3 3】  torch.rand生成一个3*3的矩阵。
import torch

a = torch.rand([3, 3])
print(a)

# 【例3 4】  torch.randn生成一个2*3*5的张量。
import torch

a = torch.randn([2, 3, 5])
print(a)

# 【例3 5】  Torch可以生成全1张量。
import torch

a = torch.ones([2, 3, 4])
print(a)

#
# 【例3 6】  Torch可以生成全0张量。
import torch

a = torch.zeros([3, 3])
print(a)

# 【例3 7】  Torch可以生成单位张量。
import torch
a = torch.eye(3)
print(a)


# 【例3 8】  Torch.randint生成服从均匀分布的整数张量。
import torch
a = torch.randint(0, 100, [3, 3])
print(a)


# 【例3 9】  通过已知张量生成相同维度张量。
# 通过已知张量生成相同维度张量。
import torch
a = torch.randn([3, 3])
print('*'*20)
print('原始张量')
print(a)
print('相同维度的0张量')
print(torch.zeros_like(a))
print('相同维度的1张量')
print(torch.ones_like(a))
print('相同维度的[0,1]之间均匀分布张量')
print(torch.rand_like(a))
print('相同维度的[0,1]之间正态分布张量')
print(torch.randn_like(a))


# 【例3 10】  生成与已知张量数据类型相同的张量。
import torch
a = torch.randn([3, 3])
print('*'*20)
print('原始张量')
print(a.dtype)
print(a)
print('生成新张量1')
print(a.new_tensor([3, 4, 5]))
print(a.new_tensor([3, 4, 5]).dtype)
print('生成新张量2')
print(a.new_zeros([3, 4]))
print(a.new_zeros([3, 4]).dtype)
print('生成新张量3')
print(a.new_ones([3, 4]))
print(a.new_ones([3, 4]).dtype)


# 【例3 11】  Pytorch在不同设备上的张量。
# Pytorch在不同设备上的张量。
import torch

print('*' * 20)
print('获取一个CPU上的张量')
print(torch.randn(3, 3, device='cpu'))
print('*' * 20)
print('获取一个GPU上的张量')
# print(torch.randn(3, 3, device='cuda:0'))
print('*' * 20)
print('获取当前张量的设备')
# print(torch.randn(3, 3, device='cuda:0').device)
print('*' * 20)
print('张量从cpu移动到gpu')
# print(torch.randn(3, 3, device='cpu').cuda().device)
print('张量从gpu移动到cpu')
# print(torch.randn(3, 3, device='cuda:0').cpu().device)
print('张量保持设备不变')
# print(torch.randn(3, 3, device='cuda:0').cuda(0).device)

# 【例3 12】  Pytorch查看张量形状相关函数。
import torch
a = torch.randn(3, 4, 5)
print('*'*20)
print('获取张量维度数目')
print(a.ndimension())
print('*'*20)
print('获取张量元素个数')
print(a.nelement())
print('*'*20)
print('获取张量每个维度的大小')
print(a.size())
print('*'*20)


# 【例3 13】  view方法改变张量维度。
# view方法改变张量维度。
import torch
a = torch.randn(12)
print('*'*20)
print('改变维度为3*4')
print(a.view(3, 4))
print('*'*20)
print('改变维度为4*3')
print(a.view(4, 3))
print('*'*20)
print('使用-1改变维度为4*3')
print(a.view(-1, 3))


# 【例3 14】  reshape方法改变张量形状。
import torch
a = torch.randn(3, 4)
print('维度改变之前')
print(a)
print('维度改变之后')
b = a.reshape(4, 3)
print(b)


# 【例3 15】  Pytorch张量的索引和切片。
# Pytorch张量的索引和切片。
import torch
a = torch.randn(2, 3, 4)
print(a)
print('*'*20)
print('取张量第0维第1个，1维2个，2维3个元素')
print(a[1, 2, 3])
print('*'*20)
print('取张量第0维第1个，1维2个，2维3个元素')
print(a[:, 1:-1, 1:3])
print('*'*20)
print('更改元素的值')
a[1, 2, 3] = 100
print(a)
print('*'*20)
print('大于0的部分掩码')
print(a>0)
print('*'*20)
print('根据掩码选择张量的元素')
print(a[a>0])


# 【例3 16】  单个张量运算函数。
# 单个张量运算函数。
import torch
a = torch.rand(3, 4)
print('*'*20)
print('查看原张量')
print(a)
print('*'*20)
print('张量内部方法，计算原张量平方根')
print(a.sqrt())
print('*'*20)
print('函数形式，计算原张量平方根')
print(a.sqrt())
print('*'*20)
print('直接操作，计算原张量平方根')
print(a.sqrt_)
print('*'*20)
print('对所有元素求和')
print(torch.sum(a))
print('*'*20)
print('对第0维、1维元素求和')
print(torch.sum(a, [0, 1]))
print('*'*20)
print('对第0维、1维元素求平均')
print(torch.mean(a, [0, 1]))


# 【例3 17】  Pytorch张量四则运算。
# Pytorch张量四则运算。
import torch
a = torch.rand(2, 3)
b = torch.rand(2, 3)
print('*'*20)
print('加法实现方式1')
print(a.add(b))
print('加法实现方式2')
print(a + b)
print('*'*20)
print('减法实现方式1')
print(a - b)
print('减法实现方式2')
print(a.sub(b))
print('*'*20)
print('乘法实现方式1')
print(a * b)
print('乘法实现方式2')
print(a.mul(b))
print('*'*20)
print('除法实现方式1')
print(a/b)
print('除法实现方式2')
print(a.div(b))


# 【例3 18】  Pytorch极值计算。
# Pytorch极值计算。
import torch
# 构建一个3*4的张量
a = torch.randn(3, 4)
print('*'*20)
print(a)
print('*'*20)
print('查看第0维极大值所在位置：')
print(torch.argmax(a, 0))
print('*'*20)
print('内置方法调用函数，查看第0维极小值所在位置：')
print(a.argmin(0))
print('*'*20)
print('沿着最后一维返回极大值和极大值的位置：')
print(torch.max(a, -1))
print('*'*20)
print('沿着最后一维返回极小值和极小值的位置：')
print(a.min(-1))


# 【例3 19】  Pytorch张量排序。
# Pytorch张量排序。
import torch
# 构建一个3*4的张量
a = torch.randn(3, 4)
print('*'*20)
print('沿着最后一维进行排序：')
print(a.sort(-1))


# 【例3 20】  Pytorch矩阵乘法。
# Pytorch矩阵乘法。
import torch
# 构建一个3*4的张量
a = torch.randn(3, 4)
# 构建一个4*3的张量
b = torch.randn(4, 3)
print('*'*20)
print('调用函数，返回3*3的矩阵：')
print(torch.mm(a, b))
print('*'*20)
print('内置函数，返回3*3的矩阵：')
print(a.mm(b))
print('*'*20)
print('@运算乘法，返回3*3的矩阵：')
print(a@b)


# 【例3 21】  torch.bmm函数实现批次矩阵乘法。
# torch.bmm函数实现批次矩阵乘法。
import torch
# 构建一个2*3*4的矩阵
a = torch.randn(2, 3, 4)
# 构建一个2*4*3的矩阵
b = torch.randn(2, 4, 3)
print('*'*20)
print('内置函数，批次矩阵乘法：')
print(a.bmm(b))
print('*'*20)
print('函数形式，批次矩阵乘法：')
print(torch.bmm(a, b))
print('*'*20)
print('@符号，批次矩阵乘法：')
print(a@b)


# 【例3 22】  Pytorch张量拼接和分割。
import torch
# 生成4个随机张量
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = torch.randn(3, 4)
d = torch.randn(3, 2)
print('*'*20)
print('沿着最后一个维度堆叠返回一个3*4*3的张量：')
e = torch.stack([a, b, c], -1)
print(e.shape)
print('*'*20)
print('沿着最后一个维度拼接返回一个3*9的张量：')
f = torch.cat([a, b, c], -1)
print(f.shape)
# 随机生成一个3*6的张量
g = torch.randn(3, 6)
print('*'*20)
print('沿着最后一个维度分割为3个张量：')
print(g.split([1, 2, 3], -1))
print('*'*20)
print('把张量沿着最后一维分割，分割为三个张量，大小均为3*2：')
print(g.chunk(3, -1))


# 【例3 23】  Pytorch维度扩充和压缩。
# Pytorch维度扩充和压缩。
import torch
a = torch.randn(3, 4)
print('*'*20)
print('查看原张量a维度：')
print(a.size())
print('*'*20)
print('扩增最后一维维度：')
print(a.unsqueeze(-1).shape)
print('*'*20)
print('再次扩增最后一维维度：')
b = a.unsqueeze(-1).unsqueeze(-1)
print(b.shape)
print('*'*20)
print('压缩所有大小为1的维度：')
print(b.squeeze().size())


# 【例3 24】  nn.Conv2d()对图像进行卷积处理。
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 读取图片转化为灰度图，并转化为numpy数组
img = Image.open("imgs/000190.jpg")
img_gray = np.array(img.convert("L"), dtype=np.float32)
plt.figure(figsize=(6, 6))
plt.imshow(img_gray, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

# 将数组转化为张量
imh, imw = img_gray.shape
img_tensor = torch.from_numpy(img_gray.reshape(1, 1, imh, imw))

# 使用5*5的随机数构成的卷积核进行卷积操作
# 这里的卷积核是个比较神奇的卷积核，中间的数值比较大，两边的数值比较小
kernel_size = 5
kernel = torch.ones(kernel_size, kernel_size, dtype=torch.float32) * -1
kernel[2, 2] = 24
kernel = kernel.reshape((1, 1, kernel_size, kernel_size))
# 进行卷积操作
conv2d = nn.Conv2d(1, 2, (kernel_size, kernel_size), bias=False)
conv2d.weight.data[0] = kernel
imgconv2dout = conv2d(img_tensor)
# 进行维度的压缩，这样图像才能展示出来
imgconv2dout_img = imgconv2dout.data.squeeze()
print("卷积之后的尺寸为：{}".format(imgconv2dout_img.shape))

# 显示图片
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(imgconv2dout_img[0], cmap=plt.cm.gray)
plt.axis("off")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 2)
plt.imshow(imgconv2dout_img[1], cmap=plt.cm.gray)
plt.axis("off")
plt.show()

# 【例3 25】  Pytorch验证池化层参数stride。
import torch
import torch.nn as nn

# 仅定义一个 3x3 的池化层窗口
m = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

# 定义输入
# 四个参数分别表示 (batch_size, C_in, H_in, W_in)
# 分别对应，批处理大小，输入通道数，图像高度（像素），图像宽度（像素）
# 为了简化表示，我们只模拟单张图片输入，单通道图片，图片大小是6x6
input = torch.randn(1, 1, 6, 6)
print(input)
output = m(input)
print(output)


# 【例3 26】  Pytorch验证池化层参数ceil_mode参数。
import torch
import torch.nn as nn

# 仅定义一个 3x3 的池化层窗口
m = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)

# 定义输入
# 四个参数分别表示 (batch_size, C_in, H_in, W_in)
# 分别对应，批处理大小，输入通道数，图像高度（像素），图像宽度（像素）
# 为了简化表示，我们只模拟单张图片输入，单通道图片，图片大小是6x6
input = torch.randn(1, 1, 6, 6)
print(input)
output = m(input)
print('\n')
print(output)


# 【例3 27】  Pytorch验证池化层参数padding参数。
import torch
import torch.nn as nn

# 仅定义一个 3x3 的池化层窗口
m = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=(1, 1))

# 定义输入
# 四个参数分别表示 (batch_size, C_in, H_in, W_in)
# 分别对应，批处理大小，输入通道数，图像高度（像素），图像宽度（像素）
# 为了简化表示，我们只模拟单张图片输入，单通道图片，图片大小是6x6
input = torch.randn(1, 1, 6, 6)
print(input)
output = m(input)
print('\n')
print(output)


# 【例3 28】  Pytorch验证池化层参数return_indices。
import torch
import torch.nn as nn

# 仅定义一个 3x3 的池化层窗口
m = nn.MaxPool2d(kernel_size=(3, 3), return_indices=True)

# 定义输入
# 四个参数分别表示 (batch_size, C_in, H_in, W_in)
# 分别对应，批处理大小，输入通道数，图像高度（像素），图像宽度（像素）
# 为了简化表示，我们只模拟单张图片输入，单通道图片，图片大小是6x6
input = torch.randn(1, 1, 6, 6)
print(input)
output = m(input)
print(output)


# 【例3 29】  Pytorch池化处理实际图像。
## 池化层加入
from copy import deepcopy
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

image = Image.open('./imgs/bridge.jpg')
image = image.convert("L")
image_np = np.array(image)

h, w = image_np.shape
image_tensor = torch.from_numpy(image_np.reshape(1, 1, h, w)).float()

kersize = 5
ker = torch.ones(kersize, kersize, dtype=torch.float32) * -1
temp = deepcopy(ker)

ker[2,2] = 24
conv2d = torch.nn.Conv2d(1, 2, (kersize, kersize), bias=False)
ker = ker.reshape((1, 1, kersize, kersize))
conv2d.weight.data[0] = ker
conv2d.weight.data[1] = temp


image_out = conv2d(image_tensor)
# # 添加池化层----最大值池化
# maxpool = nn.MaxPool2d(2,stride=2)
# pool_out = maxpool(image_out)

# 添加池化层---平均值池化
average_pool = nn.AvgPool2d(2,stride=2)
pool_out = average_pool(image_out)

# # 添加池化层---自适应平均池化层
# adaverage_pool = nn.AdaptiveAvgPool2d(output_size=(100,100))
# pool_out = adaverage_pool(image_out)

x = torch.linspace(-6,6,100)
print(type(x))

print(x)
pool_out_min = pool_out.squeeze()
image_out = image_out.squeeze()

plt.figure(figsize=(18,18),frameon=True)
plt.subplot(2,2,1)
plt.imshow(pool_out_min[1].detach(), cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(image_out[1].detach(), cmap=plt.cm.gray)
plt.axis('off')


plt.subplot(2,2,3)
plt.imshow(pool_out_min[0].detach(), cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(image_out[0].detach(), cmap=plt.cm.gray)
plt.axis('off')
plt.show()


# 【例3 30】  Pytorch可视化常用激活函数。
import torch
import matplotlib.pyplot as plt

input= torch.linspace(-10,10,2000)
X = input.numpy()

#定义激活函数
y_relu = torch.relu(input).data.numpy()
y_sigmoid =torch.sigmoid(input).data.numpy()
y_tanh = torch.tanh(input).data.numpy()

plt.figure(1, figsize=(10, 8))
plt.subplot(221)
plt.plot(X, y_relu, c='red', label='relu')
plt.legend(loc='best')
plt.subplot(222)
plt.plot(X, y_sigmoid, c='black', label='sigmoid')
plt.legend(loc='best')
plt.subplot(223)
plt.plot(X, y_tanh, c='blue', label='tanh')
plt.legend(loc='best')
plt.show()


# 【例3 31】  Pytorch全连接层算法验证。
import torch
from torch import nn

input1 = torch.tensor([[10., 20., 30.]])
linear_layer = nn.Linear(3, 5)
linear_layer .weight.data = torch.tensor([[1., 1., 1.],
                                          [2., 2., 2.],
                                          [3., 3., 3.],
                                          [4., 4., 4.],
                                          [5., 5., 5.]])

linear_layer .bias.data = torch.tensor(0.6)
output = linear_layer(input1)
print(input1)
print(output, output.shape)


# 【例3 32】  创建一个张量，来跟踪与它相关的计算。
# 创建一个张量，来跟踪与它相关的计算。
import torch
x = torch.ones(2, 2, requires_grad=True)
print(x)
print('*'*20)
print('针对张量做一个操作：')
y = x + 2
print(y)
print(y.grad_fn)
print('*'*20)
print('针对张量y做更多操作：')
z = y * y * 3
out = z.mean()
print(z, out)


# 【例3 33】  张量梯度计算。
import torch
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
# 反向传播
out.backward()
print('*'*20)
print('x的梯度：')
print(x.grad)


# 【例3 34】  雅克比向量积梯度计算。
import torch
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print('*'*20)
print('查看张量y：')
print(y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print('*'*20)
print('x的梯度：')
print(x.grad)


# 【例3 35】  with torch.no_grad()停止跟踪求导。
# with torch.no_grad()停止跟踪求导。
import torch
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
