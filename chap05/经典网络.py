


# # 【例4 8】  Pytorch查看Vgg19网络结构。
# import torchvision
# model = torchvision.models.vgg19()
# print(model)


# # 【例4 9】  Pytorch残差构建模块封装成类。
# import torch
# import torch.nn.functional as F
# class ResidualBlock(torch.nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.channels = channels
#
#         self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         y = F.relu(self.conv1(x))
#         y = self.conv2(y)
#         return F.relu(x + y)
#
# net = ResidualBlock(4)
# print(net)


# # 【例4 10】  Pytorch实现嵌入残差模块的网络模型。
# import torch
# import torch.nn.functional as F
#
# class ResidualBlock(torch.nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.channels = channels
#
#         self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         y = F.relu(self.conv1(x))
#         y = self.conv2(y)
#         return F.relu(x + y)
#
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
#         self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
#         self.mp = torch.nn.MaxPool2d(2)
#
#         self.rblock1 = ResidualBlock(16)
#         self.rblock2 = ResidualBlock(32)
#
#         self.fc = torch.nn.Linear(512, 10)
#
#     def forward(self, x):
#         # Flatten data from (n,1,28,28) to (n,784)
#         in_size = x.size(0)
#         x = self.mp(F.relu(self.conv1(x)))
#         x = self.rblock1(x)
#         x = self.mp(F.relu(self.conv2(x)))
#         x = self.rblock2(x)
#         x = x.view(in_size, -1)  # flatten
#         #         print(x.size(1))
#         return self.fc(x)
#
#
# model = Net()
# print(model)

# # 【例4 11】  Pytorch实现ResNet。
# import torch.nn as nn
# import torch
#
# '''
# 对应18层，34层的残差结构
# '''
#
#
# class BasicBlock(nn.Module):
#     expansion = 1  # 判断每一个卷积块中，卷积核的个数会不会有变化
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):  # downsample表示是否有升维操作
#         super(BasicBlock, self).__init__()
#         # output = (input - kernel_size + 2*padding)/stride + 1
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                                kernel_size=3, stride=stride, padding=1,
#                                bias=False)  # stride=1表示option A；stride=2表示optionB 使用BN不需要偏置bias
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                                kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# '''
# 50层，101层，152层
# '''
#
#
# class Bottleneck(nn.Module):
#     """
#     注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
#     但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
#     这么做的好处是能够在top1上提升大概0.5%的准确率。
#     可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
#     """
#     expansion = 4
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                                kernel_size=1, stride=1, bias=False)  # squeeze channels
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         # -----------------------------------------
#         self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                                kernel_size=3, stride=stride, bias=False, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         # -----------------------------------------
#         self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
#                                kernel_size=1, stride=1, bias=False)  # unsqueeze channels
#         self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self,
#                  block,  # 残差结构
#                  blocks_num,
#                  num_classes=1000,
#                  include_top=True,
#                  groups=1,
#                  width_per_group=64):
#         super(ResNet, self).__init__()
#         self.include_top = include_top
#         self.in_channel = 64
#
#         self.groups = groups
#         self.width_per_group = width_per_group
#
#         self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
#                                padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channel)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, blocks_num[0])  # 对应结构图中conv2_x，下面同理
#         self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
#         if self.include_top:
#             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
#             self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#     '''
#     block: BasicBlock或Bottleneck
#     channel: 残差结构中的卷积核个数
#     block_num：这一层有多少残差结构，例：34的第一层有三个，第二层有四个
#     '''
#
#     def _make_layer(self, block, channel, block_num, stride=1):
#         downsample = None
#         # 快捷连接虚线部分
#         if stride != 1 or self.in_channel != channel * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(channel * block.expansion))
#
#         layers = []
#         # 搭建每一个conv的第一层
#         layers.append(block(self.in_channel,
#                             channel,
#                             downsample=downsample,
#                             stride=stride,
#                             groups=self.groups,
#                             width_per_group=self.width_per_group))
#         self.in_channel = channel * block.expansion
#
#         for _ in range(1, block_num):
#             layers.append(block(self.in_channel,
#                                 channel,
#                                 groups=self.groups,
#                                 width_per_group=self.width_per_group))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         if self.include_top:
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.fc(x)
#
#         return x
#
#
# def resnet34(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnet34-333f7ec4.pth
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
#
#
# def resnet50(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnet50-19c8e357.pth
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
#
#
# def resnet101(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
#
#
# def resnext50_32x4d(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
#     groups = 32
#     width_per_group = 4
#     return ResNet(Bottleneck, [3, 4, 6, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)
#
#
# def resnext101_32x8d(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
#     groups = 32
#     width_per_group = 8
#     return ResNet(Bottleneck, [3, 4, 23, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)



# # 【例4 12】  Pytorch实现深度可分离卷积。
# import torch.nn as nn
# # 深度可分离卷积
# class SeparableConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False):
#         super(SeparableConv2d, self).__init__()
#
#         # 逐通道卷积：groups=in_channels=out_channels
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
#                                bias=bias)
#         # 逐点卷积：普通1x1卷积
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
#                                    bias=bias)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pointwise(x)
#         return x


# 【例4 13】  Pytorch实现XceptionNet。
# import torch.nn as nn
#
# class Xception(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(Xception, self).__init__()
#         self.num_classes = num_classes  # 总分类数
#
#         ################################## 定义 Entry flow ###############################################################
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(64)
#         # do relu here
#
#         # Block中的参数顺序：in_filters,out_filters,reps,stride,start_with_relu,grow_first
#         self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
#         self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
#         self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
#
#         ################################### 定义 Middle flow ############################################################
#         self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#
#         self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#
#         #################################### 定义 Exit flow ###############################################################
#         self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
#
#         self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
#         self.bn3 = nn.BatchNorm2d(1536)
#
#         # do relu here
#         self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
#         self.bn4 = nn.BatchNorm2d(2048)
#
#         self.fc = nn.Linear(2048, num_classes)
#         ###################################################################################################################
#
#         # ------- init weights --------
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#         # -----------------------------
#
#     def forward(self, x):
#         ################################## 定义 Entry flow ###############################################################
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#
#         ################################### 定义 Middle flow ############################################################
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.block6(x)
#         x = self.block7(x)
#         x = self.block8(x)
#         x = self.block9(x)
#         x = self.block10(x)
#         x = self.block11(x)
#
#         #################################### 定义 Exit flow ###############################################################
#         x = self.block12(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu(x)
#
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x


# 【例4 14】  Pytorch实现XceNet的block。
# import torch.nn as nn
#
#
# class Block(nn.Module):
#     def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
#         #:parm reps:块重复次数
#         super(Block, self).__init__()
#
#         # Middle flow无需做这一步，而其余块需要，以做跳连
#         # 1）Middle flow输入输出特征图个数始终一致，且stride恒为1
#         # 2）其余块需要stride=2，这样可以将特征图尺寸减半，获得与最大池化减半特征图尺寸同样的效果
#         if out_filters != in_filters or strides != 1:
#             self.skip = nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=strides, bias=False)
#             self.skipbn = nn.BatchNorm2d(out_filters)
#         else:
#             self.skip = None
#
#         self.relu = nn.ReLU(inplace=True)
#         rep = []
#
#         filters = in_filters
#         if grow_first:
#             rep.append(self.relu)
#             # 这里的卷积不改变特征图尺寸
#             rep.append(SeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False))
#             rep.append(nn.BatchNorm2d(out_filters))
#             filters = out_filters
#
#         for i in range(reps - 1):
#             rep.append(self.relu)
#             # 这里的卷积不改变特征图尺寸
#             rep.append(SeparableConv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False))
#             rep.append(nn.BatchNorm2d(filters))
#
#         if not grow_first:
#             rep.append(self.relu)
#             # 这里的卷积不改变特征图尺寸
#             rep.append(SeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False))
#             rep.append(nn.BatchNorm2d(out_filters))
#
#         if not start_with_relu:
#             rep = rep[1:]
#         else:
#             rep[0] = nn.ReLU(inplace=False)
#
#         # Middle flow 的stride恒为1，因此无需做池化，而其余块需要
#         # 其余块的stride=2，因此这里的最大池化可以将特征图尺寸减半
#         if strides != 1:
#             rep.append(nn.MaxPool2d(kernel_size=3, stride=strides, padding=1))
#         self.rep = nn.Sequential(*rep)
#
#     def forward(self, inp):
#         x = self.rep(inp)
#
#         if self.skip is not None:
#             skip = self.skip(inp)
#             skip = self.skipbn(skip)
#         else:
#             skip = inp
#
#         x += skip
#         return x


# 【例4 15】  Pytorch构建分类器，并查看其字典状态。
import torch.nn as nn
import torch.optim as optim
import torch.functional as F


class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
model = TheModelClass()

# 初始化优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 打印模型的状态字典
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 打印优化器的状态字典
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])