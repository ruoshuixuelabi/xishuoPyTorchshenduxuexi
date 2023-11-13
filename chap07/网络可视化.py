

# # 建立一个网络模型，并使用HiddenLayer可视化。
# import torch
# import torchvision
# import hiddenlayer as h
#
# model = torchvision.models.alexnet()
# x = torch.randn([3, 3, 224, 224])
# NetGraph = h.build_graph(model, x)
# NetGraph.save('./model.png', format='png')
# print(model)


# # Pytorch建立模型，并使用PytorchViz可视化。
# import torch
# from torchvision.models import AlexNet
# from torchviz import make_dot
#
# x = torch.rand(8, 3, 256, 512)
# model = AlexNet()
# y = model(x)
# print(y)
# g = make_dot(y)
# g.render('espnet_model', view=False)
#
#
# # PytorchViz可视化，查询整个模型的参数量信息。
# import torch
# from torchvision.models import AlexNet
# from torchviz import make_dot
#
# x = torch.rand(8, 3, 256, 512)
# model = AlexNet()
# y = model(x)
# # print(y)
# # g = make_dot(y)
# # g.render('espnet_model', view=False)
#
# params = list(model.parameters())
# k = 0
# for i in params:
#         l = 1
#         print("该层的结构：" + str(list(i.size())))
#         for j in i.size():
#                 l *= j
#         print("该层参数和：" + str(l))
#         k = k + l
# print("总参数数量和：" + str(k))


# # 创建一个SummaryWriter
# from tensorboardX import SummaryWriter
#
# writer1 = SummaryWriter('runs/exp')
# writer2 = SummaryWriter()
# writer3 = SummaryWriter(comment='resnet')


# # SummaryWriter添加数字。
# from tensorboardX import SummaryWriter
# writer = SummaryWriter('runs/scalar_example')
# for i in range(10):
#     writer.add_scalar('quadratic', i**2, global_step=i)
#     writer.add_scalar('exponential', 2**i, global_step=i)


# # SummaryWriter添加图片。
# from tensorboardX import SummaryWriter
# import cv2 as cv
#
#
# writer = SummaryWriter('runs/image_example')
#
# for i in range(1, 5):
#     print(i)
#     writer.add_image('countdown',
#                      cv.cvtColor(cv.imread('{}.png'.format(i)), cv.COLOR_BGR2RGB),
#                      global_step=i,
#                      dataformats='HWC')


# # SummaryWriter添加直方图。
# from tensorboardX import SummaryWriter
# import numpy as np
#
# writer = SummaryWriter('runs/histogram_example')
# writer.add_histogram('normal_centered', np.random.normal(0, 1, 1000), global_step=1)
# writer.add_histogram('normal_centered', np.random.normal(0, 2, 1000), global_step=50)
# writer.add_histogram('normal_centered', np.random.normal(0, 3, 1000), global_step=100)


# # SummaryWriter添加嵌入向量。
# from tensorboardX import SummaryWriter
# import torchvision
#
# writer = SummaryWriter('runs/embedding_example')
# mnist = torchvision.datasets.MNIST('mnist', download=True)
# writer.add_embedding(
#     mnist.train_data.reshape((-1, 28 * 28))[:100, :],
#     metadata=mnist.train_labels[:100],
#     label_img=mnist.train_data[:100, :, :].reshape((-1, 1, 28, 28)).float() / 255,
#     global_step=0
# )