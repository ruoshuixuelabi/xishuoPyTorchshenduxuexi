
# # 猫狗数据预处理
# import os, shutil
# import numpy as np
# import pdb
#
# random_state = 30
# np.random.seed(random_state)
#
# original_dataset_dir = './data/xxx'  # 自定义路径
# total_num = int(len(os.listdir(original_dataset_dir)) / 2)
# random_idx = np.array(range(total_num))
# np.random.shuffle(random_idx)
# base_dir = 'xxx'   # 自定义路径
# if not os.path.exists(base_dir):
#     os.mkdir(base_dir)
#
# sub_dirs = ['train', 'test']
# animals = ['cats', 'dogs']
# train_idx = random_idx[: int(total_num * 0.9)]
# test_idx = random_idx[int(total_num * 0.9):]
# numbers = [train_idx, test_idx]
# for idx, sub_dir in enumerate(sub_dirs):
#     dir = os.path.join(base_dir, sub_dir)
#     if not os.path.exists(dir):
#         os.mkdir(dir)
#     for animal in animals:
#         animal_dir = os.path.join(dir, animal)
#         if not os.path.exists(animal_dir):
#             os.mkdir(animal_dir)
#         fnames = [animal[: -1] + '.{}.jpg'.format(i) for i in numbers[idx]]
#         for fname in fnames:
#             src = os.path.join(original_dataset_dir, fname)
#             dst = os.path.join(animal_dir, fname)
#             shutil.copyfile(src, dst)
#         print(dir + 'total images : %d' % (len(os.listdir(animal_dir))))


# Pytorch建立网络猫狗分类。
from __future__ import print_function, division
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
import torch.optim as optim

torch.manual_seed(1)
epochs = 10  #自己根据需要定义
batch_size = 16  #根据设备配置
num_workers = 0  #线程数，根据设备自行配置
use_gpu = torch.cuda.is_available()

data_transform = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(root='/train/', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataset = datasets.ImageFolder(root='/test/', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if use_gpu:
    net = Net().cuda()
else:
    net = Net()
print(net)

cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

net.train()
for epoch in range(epochs):
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, train_labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(train_labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        _, train_predicted = torch.max(outputs.data, 1)

        train_correct += (train_predicted == labels.data).sum()
        loss = cirterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_total += train_labels.size(0)
    print('train %d epoch loss: %.3f acc: %.3f ' % (
    epoch + 1, running_loss / train_total, 100 * train_correct / train_total))

    correct = 0
    test_loss = 0.0
    test_total = 0
    net.eval()
    for data in test_loader:
        images, labels = data
        if use_gpu:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = cirterion(outputs, labels)
        test_loss += loss.item()
        test_total += labels.size(0)
        correct += (predicted == labels.data).sum()

    print('test %d epoch loss: %.3f acc: %.3f' % (epoch + 1, test_loss / test_total, 100 * correct / test_total))