
# # 固定风格迁移
#
# # 导入模块
# from __future__ import print_function
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# from PIL import Image
# import matplotlib.pyplot as plt
#
# import torchvision.transforms as transforms
# import torchvision.models as models
#
# import copy
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # 载入图像
# imsize = 512 if torch.cuda.is_available() else 128
#
# loader = transforms.Compose([
#     transforms.Resize([imsize, imsize]),
#     transforms.ToTensor()])
#
#
# def image_loader(image_name):
#     image = Image.open(image_name)
#     # fake batch dimension required to fit network's input dimensions
#     image = loader(image).unsqueeze(0)
#     return image.to(device, torch.float)
#
# content_img = image_loader('./style_img.jpeg')
# style_img = image_loader('./content_img.jpeg')
#
# print(style_img.shape)
# print(content_img.shape)
#
#
# assert style_img.size() == content_img.size(), \
#     "风格图像和内容图像需要大小一致"
#
#
#
# unloader = transforms.ToPILImage()
#
# plt.ion()
#
# def imshow(tensor, title=None):
#     image = tensor.cpu().clone()
#     image = image.squeeze(0)
#     image = unloader(image)
#     plt.imshow(image)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001) # pause a bit so that plots are updated
#
#
# plt.figure()
# imshow(style_img, title='Style Image')
#
# plt.figure()
# imshow(content_img, title='Content Image')
#
#
#
# # 损失函数
# # 内容损失
# class ContentLoss(nn.Module):
#
#     def __init__(self, target,):
#         super(ContentLoss, self).__init__()
#         self.target = target.detach()
#
#     def forward(self, input):
#         self.loss = F.mse_loss(input, self.target)
#         return input
#
#
#
# # 风格损失
# def gram_matrix(input):
#     a, b, c, d = input.size()
#
#     features = input.view(a * b, c * d)
#
#     G = torch.mm(features, features.t())
#
#     # 归一化
#     # 通过除以元素数目归一化
#     return G.div(a * b * c * d)
#
#
# class StyleLoss(nn.Module):
#
#     def __init__(self, target_feature):
#         super(StyleLoss, self).__init__()
#         self.target = gram_matrix(target_feature).detach()
#
#     def forward(self, input):
#         G = gram_matrix(input)
#         self.loss = F.mse_loss(G, self.target)
#         return input
#
#
#
# # 导入模型
# cnn = models.vgg19(pretrained=True).features.to(device).eval()
#
#
# cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
# cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
#
# # 定义模块归一化
# class Normalization(nn.Module):
#     def __init__(self, mean, std):
#         super(Normalization, self).__init__()
#         # .view the mean and std to make them [C x 1 x 1] so that they can
#         # directly work with image Tensor of shape [B x C x H x W].
#         # B is batch size. C is number of channels. H is height and W is width.
#         self.mean = torch.tensor(mean).view(-1, 1, 1)
#         self.std = torch.tensor(std).view(-1, 1, 1)
#
#     def forward(self, img):
#         # normalize img
#         return (img - self.mean) / self.std
#
#
# # 期望的计算内容损失和风格损失的特征层
# content_layers_default = ['conv_4']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
#
# def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
#                                style_img, content_img,
#                                content_layers=content_layers_default,
#                                style_layers=style_layers_default):
#     # normalization module
#     normalization = Normalization(normalization_mean, normalization_std).to(device)
#
#     content_losses = []
#     style_losses = []
#
#     model = nn.Sequential(normalization)
#
#     i = 0
#     for layer in cnn.children():
#         if isinstance(layer, nn.Conv2d):
#             i += 1
#             name = 'conv_{}'.format(i)
#         elif isinstance(layer, nn.ReLU):
#             name = 'relu_{}'.format(i)
#             layer = nn.ReLU(inplace=False)
#         elif isinstance(layer, nn.MaxPool2d):
#             name = 'pool_{}'.format(i)
#         elif isinstance(layer, nn.BatchNorm2d):
#             name = 'bn_{}'.format(i)
#         else:
#             raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
#
#         model.add_module(name, layer)
#
#         if name in content_layers:
#             # 添加内容损失
#             target = model(content_img).detach()
#             content_loss = ContentLoss(target)
#             model.add_module("content_loss_{}".format(i), content_loss)
#             content_losses.append(content_loss)
#
#         if name in style_layers:
#             # 添加风格损失
#             target_feature = model(style_img).detach()
#             style_loss = StyleLoss(target_feature)
#             model.add_module("style_loss_{}".format(i), style_loss)
#             style_losses.append(style_loss)
#
#     for i in range(len(model) - 1, -1, -1):
#         if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
#             break
#
#     model = model[:(i + 1)]
#
#     return model, style_losses, content_losses
#
#
# input_img = content_img.clone()
# # input_img = torch.randn(content_img.data.size(), device=device)
#
# plt.figure()
# imshow(input_img, title='Input Image')
#
#
#
# # 梯度下降法
# def get_input_optimizer(input_img):
#     optimizer = optim.LBFGS([input_img])
#     return optimizer
#
# # 定义迁移网络
# def run_style_transfer(cnn, normalization_mean, normalization_std,
#                        content_img, style_img, input_img, num_steps=1000,
#                        style_weight=1000000, content_weight=1):
#     print('Building the style transfer model..')
#     model, style_losses, content_losses = get_style_model_and_losses(cnn,
#         normalization_mean, normalization_std, style_img, content_img)
#
#     # We want to optimize the input and not the model parameters so we
#     # update all the requires_grad fields accordingly
#     input_img.requires_grad_(True)
#     model.requires_grad_(False)
#
#     optimizer = get_input_optimizer(input_img)
#
#     print('Optimizing..')
#     run = [0]
#     while run[0] <= num_steps:
#
#         def closure():
#             # correct the values of updated input image
#             with torch.no_grad():
#                 input_img.clamp_(0, 1)
#
#             optimizer.zero_grad()
#             model(input_img)
#             style_score = 0
#             content_score = 0
#
#             for sl in style_losses:
#                 style_score += sl.loss
#             for cl in content_losses:
#                 content_score += cl.loss
#
#             style_score *= style_weight
#             content_score *= content_weight
#
#             loss = style_score + content_score
#             loss.backward()
#
#             run[0] += 1
#             if run[0] % 50 == 0:
#                 print("run {}:".format(run))
#                 print('Style Loss : {:4f} Content Loss: {:4f}'.format(
#                     style_score.item(), content_score.item()))
#                 print()
#
#             return style_score + content_score
#
#         optimizer.step(closure)
#
#     # a last correction...
#     with torch.no_grad():
#         input_img.clamp_(0, 1)
#
#     return input_img
#
#
# output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, input_img)
#
# plt.figure()
# imshow(output, title='Output Image')
#
# plt.ioff()
# plt.show()



# 快速风格迁移

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models


import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 模型加载选择GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(device)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))



# ResidualBlock 残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1)
        )
    def forward(self, x):
        return F.relu(self.conv(x) + x)



# 定义图像转换网络
class ImfwNet(nn.Module):
    def __init__(self):
        super(ImfwNet, self).__init__()
        # 下采样
        self.downsample = nn.Sequential(
            nn.ReflectionPad2d(padding = 4), # 使用边界反射填充
            nn.Conv2d(3, 32, kernel_size = 9, stride = 1),
            nn.InstanceNorm2d(32, affine = True), # 像素值上做归一化
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 2),
            nn.InstanceNorm2d(64, affine = True),
            nn.ReLU(),
            nn.ReflectionPad2d(padding = 1),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2),
            nn.InstanceNorm2d(128, affine = True),
            nn.ReLU()
        )
        # 5个残差连接
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        # 上采样
        self.unsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.InstanceNorm2d(64, affine = True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.InstanceNorm2d(32, affine = True),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size = 9, stride = 1, padding = 4)
        )
    def forward(self, x):
        x = self.downsample(x) # 输入像素值在-2.1-2.7之间
        x = self.res_blocks(x)
        x = self.unsample(x) # 输出像素值在-2.1-2.7之间
        return x


myfwnet = ImfwNet().to(device)


# 定义图像预处理
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256), # 图像尺寸为256*256
    transforms.ToTensor(), # 转为0-1的张量
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
                         # 像素值转为-2.1-2.7
])



# 从文件夹中读取数据
dataset = ImageFolder(r'./data/', transform = data_transform)
# 每个batch使用4张图像
data_loader = Data.DataLoader(dataset, batch_size = 8, shuffle = True,
                              num_workers = 0, pin_memory = True)

# 读取预训练的VGG16网络
vgg16 = models.vgg16(pretrained = True)
# 不需要分类器，只需要卷积层和池化层
vgg = vgg16.features.to(device).eval()


# 定义一个读取风格图像函数，并将图像进行必要的转化
def load_image(img_path, shape = None):
    image = Image.open(img_path)
    size = image.size
    if shape is not None:
        size = shape # 如果指定了图像尺寸就转为指定的尺寸
    # 使用transforms将图像转为张量，并标准化
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(), # 转为0-1的张量
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225])
    ])
    # 使用图像的RGB通道，并添加batch维度
    image = in_transform(image)[:3, :, :].unsqueeze(dim = 0)
    return image


# 定义一个将标准化后的图像转化为便于利用matplotlib可视化的函数
def im_convert(tensor):
    '''
    将[1, c, h, w]维度的张量转为[h, w, c]的数组
    因为张量进行了表转化，所以要进行标准化逆变换
    '''
    tensor = tensor.cpu()
    image = tensor.data.numpy().squeeze() # 去除batch维度的数据
    image = image.transpose(1, 2, 0) # 置换数组维度[c, h, w]->[h, w, c]
    # 进行标准化的逆操作
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1) # 将图像的取值剪切到0-1之间
    return image


# 读取风格图像
style = load_image('./data/ants/148715752_302c84f5a4.jpg', shape = (256, 256)).to(device)
# style = load_image('./style.jpeg', shape = (256, 256)).to(device)
# 可视化图像
plt.figure()
plt.imshow(im_convert(style))
plt.axis('off')
plt.show()


# 定义计算格拉姆矩阵
def gram_matrix(tensor):
    '''
    计算表示图像风格特征的Gram矩阵，它最终能够在保证内容的情况下，
    进行风格的传输。tensor：是一张图像前向计算后的一层特征映射
    '''
    # 获得tensor的batch_size, channel, height, width
    b, c, h, w = tensor.size()
    # 改变矩阵的维度为(深度, 高*宽)
    tensor = tensor.view(b, c, h * w)
    tensor_t = tensor.transpose(1, 2)
    # 计算gram matrix，针对多张图像进行计算
    gram = tensor.bmm(tensor_t) / (c * h * w)
    return gram


# 定义一个用于获取图像在网络上指定层的输出的方法
def get_features(image, model, layers = None):
    '''
    将一张图像image在一个网络model中进行前向传播计算，
    并获取指定层layers中的特征输出
    '''
    # 将映射层名称与论文中的名称相对应
    if layers is None:
        layers = {'3': 'relu1_2',
                  '8': 'relu2_2',
                  '15': 'relu3_3', # 内容图层表示
                  '22': 'relu4_3'} # 经过ReLU激活后的输出
    features = {} # 获得的每层特征保存到字典中
    x = image # 需要获取特征的图像
    # model._modules是一个字典，保存着网络model每层的信息
    for name, layer in model._modules.items():
        # 从第一层开始获取图像的特征
        x = layer(x)
        # 如果是layers参数指定的特征，就保存到features中
        if name in layers:
            features[layers[name]] = x
    return features


# 计算风格图像的风格表示
style_layer = {'3': 'relu1_2',
               '8': 'relu2_2',
               '15': 'relu3_3',
               '22': 'relu4_3'}
content_layer = {'15': 'relu3_3'}
# 内容表示的图层，均使用经过relu激活后的输出
style_features = get_features(style, vgg, layers = style_layer)
# 为我们的风格表示计算每层的格拉姆矩阵，使用字典保存
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}



# 网络训练，定义三种损失的权重
style_weight = 1e5
content_weight = 1
tv_weight = 1e-5
# 定义优化器
optimizer = optim.Adam(myfwnet.parameters(), lr = 1e-3)

if __name__ == '__main__':
    myfwnet.train()
    since = time.time()
    for epoch in range(501):
        print('Epoch: {}'.format(epoch + 1))
        content_loss_all = []
        style_loss_all = []
        tv_loss_all = []
        all_loss = []
        for step, batch in enumerate(data_loader):
            optimizer.zero_grad()

            # 计算使用图像转换网络后，内容图像得到的输出
            content_images = batch[0].to(device)
            transformed_images = myfwnet(content_images)
            transformed_images = transformed_images.clamp(-2.1, 2.7)

            # 使用VGG16计算原图像对应的content_layer特征
            content_features = get_features(content_images, vgg, layers = content_layer)

            # 使用VGG16计算\hat{y}图像对应的全部特征
            transformed_features = get_features(transformed_images, vgg)

            # 内容损失
            # 使用F.mse_loss函数计算预测(transformed_images)和标签(content_images)之间的损失
            content_loss = F.mse_loss(transformed_features['relu3_3'], content_features['relu3_3'])
            content_loss = content_weight * content_loss

            # 全变分损失
            # total variation图像水平和垂直平移一个像素，与原图相减
            # 然后计算绝对值的和即为tv_loss
            y = transformed_images # \hat{y}
            tv_loss = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
            tv_loss = tv_weight * tv_loss

            # 风格损失
            style_loss = 0
            transformed_grams = {layer: gram_matrix(transformed_features[layer]) for layer in transformed_features}
            for layer in style_grams:
                transformed_gram = transformed_grams[layer]
                # 是针对一个batch图像的Gram
                style_gram = style_grams[layer]
                # 是针对一张图像的，所以要扩充style_gram
                # 并计算计算预测(transformed_gram)和标签(style_gram)之间的损失
                style_loss += F.mse_loss(transformed_gram,
                                    style_gram.expand_as(transformed_gram))
            style_loss = style_weight * style_loss

            # 3个损失加起来，梯度下降
            loss = style_loss + content_loss + tv_loss
            loss.backward(retain_graph = True)
            optimizer.step()

            # 统计各个损失的变化情况
            content_loss_all.append(content_loss.item())
            style_loss_all.append(style_loss.item())
            tv_loss_all.append(tv_loss.item())
            all_loss.append(loss.item())
            if epoch  % 100 == 1:
                print('step: {}; content loss: {:.3f}; style loss: {:.3f}; tv loss: {:.3f}, loss: {:.3f}'.format(step, content_loss.item(), style_loss.item(), tv_loss.item(), loss.item()))
                time_use = time.time() - since
                print('Train complete in {:.0f}m {:.0f}s'.format(time_use // 60, time_use % 60))
                # 可视化一张图像
                plt.figure()
                im = transformed_images[1, ...] # 省略号表示后面的内容不写了
                plt.axis('off')
                plt.imshow(im_convert(im))
                plt.show()



    myfwnet.eval()
    for step, batch in enumerate(data_loader):
        content_images = batch[0].to(device)
        if step > 0:
            break
    plt.figure(figsize=(16, 4))
    for ii in range(4):
        im = content_images[ii, ...]
        plt.subplot(1, 4, ii + 1)
        plt.axis('off')
        plt.imshow(im_convert(im))
    plt.show()
    transformed_images = myfwnet(content_images)
    transformed_images = transformed_images.clamp(-2.1, 2.7)
    plt.figure(figsize=(16, 4))
    for ii in range(4):
        im = im_convert(transformed_images[ii, ...])
        plt.subplot(1, 4, ii + 1)
        plt.axis('off')
        plt.imshow(im)
    plt.show()

