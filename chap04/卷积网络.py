
# # 【例4 1】  Numpy构建神经网络。
# # -*- coding: utf-8 -*-
# import numpy as np
#
# # N是批量大小; D_in是输入维度;
# # H是隐藏的维度; D_out是输出维度。
# N, D_in, H, D_out = 64, 1000, 100, 10
#
# # 创建随机输入和输出数据
# x = np.random.randn(N, D_in)
# y = np.random.randn(N, D_out)
#
# # 随机初始化权重
# w1 = np.random.randn(D_in, H)
# w2 = np.random.randn(H, D_out)
#
# learning_rate = 1e-6
# for t in range(500):
#     # 前向传递：计算预测值y
#     h = x.dot(w1)
#     h_relu = np.maximum(h, 0)
#     y_pred = h_relu.dot(w2)
#
#     # 计算和打印损失loss
#     loss = np.square(y_pred - y).sum()
#     if t % 100 == 0:
#         print(t, loss)
#
#     # 反向传播，计算w1和w2对loss的梯度
#     grad_y_pred = 2.0 * (y_pred - y)
#     grad_w2 = h_relu.T.dot(grad_y_pred)
#     grad_h_relu = grad_y_pred.dot(w2.T)
#     grad_h = grad_h_relu.copy()
#     grad_h[h < 0] = 0
#     grad_w1 = x.T.dot(grad_h)
#
#     # 更新权重
#     w1 -= learning_rate * grad_w1
#     w2 -= learning_rate * grad_w2


# # 【例4 2】  Pytorch建立两层神经网络，实现前向传播和反向传播。
# # -*- coding: utf-8 -*-
# import torch
#
#
# dtype = torch.float
# device = torch.device("cpu")
# # device = torch.device（“cuda：0”）＃取消注释以在GPU上运行
#
# # N是批量大小; D_in是输入维度;
# # H是隐藏的维度; D_out是输出维度。
# N, D_in, H, D_out = 64, 1000, 100, 10
#
# #创建随机输入和输出数据
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)
#
# # 随机初始化权重
# w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype)
#
# learning_rate = 1e-6
# for t in range(500):
#     # 前向传递：计算预测y
#     h = x.mm(w1)
#     h_relu = h.clamp(min=0)
#     y_pred = h_relu.mm(w2)
#
#     # 计算和打印损失
#     loss = (y_pred - y).pow(2).sum().item()
#     if t % 100 == 0:
#         print(t, loss)
#
#     # Backprop计算w1和w2相对于损耗的梯度
#     grad_y_pred = 2.0 * (y_pred - y)
#     grad_w2 = h_relu.t().mm(grad_y_pred)
#     grad_h_relu = grad_y_pred.mm(w2.t())
#     grad_h = grad_h_relu.clone()
#     grad_h[h < 0] = 0
#     grad_w1 = x.t().mm(grad_h)
#
#     # 使用梯度下降更新权重
#     w1 -= learning_rate * grad_w1
#     w2 -= learning_rate * grad_w2


# # 【例4 3】  Pytorch优化模块optim优化器调用。
# import torch
#
# # N是批大小；D是输入维度
# # H是隐藏层维度；D_out是输出维度
# N, D_in, H, D_out = 64, 1000, 100, 10
#
# # 产生随机输入和输出张量
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
#
# # 使用nn包定义模型和损失函数
# model = torch.nn.Sequential(
#           torch.nn.Linear(D_in, H),
#           torch.nn.ReLU(),
#           torch.nn.Linear(H, D_out),
#         )
# loss_fn = torch.nn.MSELoss(reduction='sum')
#
# # 使用optim包定义优化器（Optimizer）。Optimizer将会为我们更新模型的权重。
# # 这里我们使用Adam优化方法；optim包还包含了许多别的优化算法。
# # Adam构造函数的第一个参数告诉优化器应该更新哪些张量。
# learning_rate = 1e-4
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# for t in range(500):
#
#     # 前向传播：通过像模型输入x计算预测的y
#     y_pred = model(x)
#
#     # 计算并打印loss
#     loss = loss_fn(y_pred, y)
#     if t % 100 == 0:
#         print(t, loss.item())
#
#     # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)
#     optimizer.zero_grad()
#
#     # 反向传播：根据模型的参数计算loss的梯度
#     loss.backward()
#
#     # 调用Optimizer的step函数使它所有参数更新
#     optimizer.step()


# # 【例4 4】  Pytorch自定义torch.nn.Module的子类构建两层网络。
# import torch
#
# class TwoLayerNet(torch.nn.Module):
#     def __init__(self, D_in, H, D_out):
#         """
#         在构造函数中，我们实例化了两个nn.Linear模块，并将它们作为成员变量。
#         """
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = torch.nn.Linear(D_in, H)
#         self.linear2 = torch.nn.Linear(H, D_out)
#
#     def forward(self, x):
#         """
#         在前向传播的函数中，我们接收一个输入的张量，也必须返回一个输出张量。
#         我们可以使用构造函数中定义的模块以及张量上的任意的（可微分的）操作。
#         """
#         h_relu = self.linear1(x).clamp(min=0)
#         y_pred = self.linear2(h_relu)
#         return y_pred
#
# # N是批大小； D_in 是输入维度；
# # H 是隐藏层维度； D_out 是输出维度
# N, D_in, H, D_out = 64, 1000, 100, 10
#
# # 产生输入和输出的随机张量
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
#
# # 通过实例化上面定义的类来构建我们的模型。
# model = TwoLayerNet(D_in, H, D_out)
#
# # 构造损失函数和优化器。
# # SGD构造函数中对model.parameters()的调用，
# # 将包含模型的一部分，即两个nn.Linear模块的可学习参数。
# loss_fn = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
# for t in range(500):
#     # 前向传播：通过向模型传递x计算预测值y
#     y_pred = model(x)
#
#     #计算并输出loss
#     loss = loss_fn(y_pred, y)
#     if t % 100 == 0:
#         print(t, loss.item())
#
#     # 清零梯度，反向传播，更新权重
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()



# # 【例4 5】  Pytorch网络权重共享。
# import random
# import torch
#
# class DynamicNet(torch.nn.Module):
#     def __init__(self, D_in, H, D_out):
#         """
#         在构造函数中，我们构造了三个nn.Linear实例，它们将在前向传播时被使用。
#         """
#         super(DynamicNet, self).__init__()
#         self.input_linear = torch.nn.Linear(D_in, H)
#         self.middle_linear = torch.nn.Linear(H, H)
#         self.output_linear = torch.nn.Linear(H, D_out)
#
#     def forward(self, x):
#         """
#         对于模型的前向传播，我们随机选择0、1、2、3，
#         并重用了多次计算隐藏层的middle_linear模块。
#         由于每个前向传播构建一个动态计算图，
#         我们可以在定义模型的前向传播时使用常规Python控制流运算符，如循环或条件语句。
#         在这里，我们还看到，在定义计算图形时多次重用同一个模块是完全安全的。
#         这是Lua Torch的一大改进，因为Lua Torch中每个模块只能使用一次。
#         """
#         h_relu = self.input_linear(x).clamp(min=0)
#         for _ in range(random.randint(0, 3)):
#             h_relu = self.middle_linear(h_relu).clamp(min=0)
#         y_pred = self.output_linear(h_relu)
#         return y_pred
#
#
# # N是批大小；D是输入维度
# # H是隐藏层维度；D_out是输出维度
# N, D_in, H, D_out = 64, 1000, 100, 10
#
# # 产生输入和输出随机张量
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
#
# # 实例化上面定义的类来构造我们的模型
# model = DynamicNet(D_in, H, D_out)
#
# # 构造损失函数（loss function）和优化器（Optimizer）。
# # 用平凡的随机梯度下降训练这个奇怪的模型是困难的，所以我们使用了momentum方法。
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
# for t in range(500):
#
#     # 前向传播：通过向模型传入x计算预测的y。
#     y_pred = model(x)
#
#     # 计算并打印损失
#     loss = criterion(y_pred, y)
#     if t % 100 == 0:
#         print(t, loss.item())
#
#     # 清零梯度，反向传播，更新权重
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


# # 【例4 6】  Pytorch搭建一个全连接神经网络。
# import torch
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# import torch.nn.functional as F
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# torch.manual_seed(10)#固定每次初始化模型的权重
# training_step = 500#迭代此时
# batch_size = 512#每个批次的大小
# n_features = 32#特征数目
# M = 10000#生成的数据数目
# #生成数据
# data = np.random.randn(M,n_features)#随机生成服从高斯分布的数据
# target = np.random.rand(M)
#
# #特征归一化
# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit(data)
# data = min_max_scaler.transform(data)
#
# # 对训练集进行切割，然后进行训练
# x_train,x_val,y_train,y_val = train_test_split(data,target,test_size=0.2,shuffle=False)
#
# #定义网络结构
# class Net(torch.nn.Module):  # 继承 torch 的 Module
#
#     def __init__(self, n_features):
#         super(Net, self).__init__()     # 继承 __init__ 功能
#         self.l1 = nn.Linear(n_features,500)#特征输入
#         self.l2 = nn.ReLU()#激活函数
#         self.l3 = nn.BatchNorm1d(500)#批标准化
#         self.l4 = nn.Linear(500,250)
#         self.l5 = nn.ReLU()
#         self.l6 = nn.BatchNorm1d(250)
#         self.l7 = nn.Linear(250,1)
#         #self.l8 = nn.Sigmoid()
#     def forward(self, inputs):   # 这同时也是 Module 中的 forward 功能
#         # 正向传播输入值, 神经网络分析出输出值
#         out = torch.from_numpy(inputs).to(torch.float32)#将输入的numpy格式转换成tensor
#         out = self.l1(out)
#         out = self.l2(out)
#         out = self.l3(out)
#         out = self.l4(out)
#         out = self.l5(out)
#         out = self.l6(out)
#         out = self.l7(out)
#         #out = self.l8(out)
#         return out
#
#
# #定义模型
# model = Net(n_features=n_features)
#
# #定义优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 传入 net 的所有参数, 学习率
# #定义目标损失函数
# loss_func = torch.nn.MSELoss() #这里采用均方差函数
#
# #开始迭代
# for step in range(training_step):
#     M_train = len(x_train)
#     with tqdm(np.arange(0,M_train,batch_size), desc='Training...') as tbar:
#         for index in tbar:
#             L = index
#             R = min(M_train,index+batch_size)
#             #-----------------训练内容------------------
#             train_pre = model(x_train[L:R,:])     # 喂给 model训练数据 x, 输出预测值
#             train_loss = loss_func(train_pre, torch.from_numpy(y_train[L:R].reshape(R-L,1)).to(torch.float32))
#             val_pre = model(x_val)
#             val_loss = loss_func(val_pre, torch.from_numpy(y_val.reshape(len(y_val),1)).to(torch.float32))
#             #-------------------------------------------
#             tbar.set_postfix(train_loss=float(train_loss.data),val_loss=float(val_loss.data))#打印在进度条上
#             tbar.update()  # 默认参数n=1，每update一次，进度+n
#
#             #-----------------反向传播更新---------------
#             optimizer.zero_grad()   # 清空上一步的残余更新参数值
#             train_loss.backward()         # 以训练集的误差进行反向传播, 计算参数更新值
#             optimizer.step()        # 将参数更新值施加到 net 的 parameters 上



# # 【例4 7】  Pytorch搭建全连接神经网络，并打印查看网络结构。
# import torch
# import torch.nn as nn
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# torch.manual_seed(10)#固定每次初始化模型的权重
# training_step = 500#迭代此时
# batch_size = 512#每个批次的大小
# n_features = 32#特征数目
# M = 10000#生成的数据数目
# #生成数据
# data = np.random.randn(M,n_features)#随机生成服从高斯分布的数据
# target = np.random.rand(M)
#
# #特征归一化
# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit(data)
# data = min_max_scaler.transform(data)
#
# # 对训练集进行切割，然后进行训练
# x_train,x_val,y_train,y_val = train_test_split(data,target,test_size=0.2,shuffle=False)
#
# #定义网络结构
# class Net(torch.nn.Module):  # 继承 torch 的 Module
#
#     def __init__(self, n_features):
#         super(Net, self).__init__()     # 继承 __init__ 功能
#         self.l1 = nn.Linear(n_features,500)#特征输入
#         self.l2 = nn.ReLU()#激活函数
#         self.l3 = nn.BatchNorm1d(500)#批标准化
#         self.l4 = nn.Linear(500,250)
#         self.l5 = nn.ReLU()
#         self.l6 = nn.BatchNorm1d(250)
#         self.l7 = nn.Linear(250,1)
#         self.l8 = nn.Sigmoid()
#     def forward(self, inputs):   # 这同时也是 Module 中的 forward 功能
#         # 正向传播输入值, 神经网络分析出输出值
#         out = torch.from_numpy(inputs).to(torch.float32)#将输入的numpy格式转换成tensor
#         out = self.l1(out)
#         out = self.l2(out)
#         out = self.l3(out)
#         out = self.l4(out)
#         out = self.l5(out)
#         out = self.l6(out)
#         out = self.l7(out)
#         out = self.l8(out)
#         return out
#
#
# #定义模型
# model = Net(n_features=n_features)
# print(model)
