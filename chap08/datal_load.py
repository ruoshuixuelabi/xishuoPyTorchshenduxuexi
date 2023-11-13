
# # Pytorch加载常见数据集。
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
#
# def dataloader(dataset, input_size, batch_size, split='train'):
#     transform = transforms.Compose([
#         					transforms.Resize((input_size, input_size)),
#        					    transforms.ToTensor(),
#         					transforms.Normalize(mean=[0.5], std=[0.5])
#     ])
#     if dataset == 'mnist':
#         data_loader = DataLoader(
#             datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
#             batch_size=batch_size, shuffle=True)
#     elif dataset == 'fashion-mnist':
#         data_loader = DataLoader(
#             datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
#             batch_size=batch_size, shuffle=True)
#     elif dataset == 'cifar10':
#         data_loader = DataLoader(
#             datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
#             batch_size=batch_size, shuffle=True)
#     elif dataset == 'svhn':
#         data_loader = DataLoader(
#             datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
#             batch_size=batch_size, shuffle=True)
#     elif dataset == 'stl10':
#         data_loader = DataLoader(
#             datasets.STL10('data/stl10', split=split, download=True, transform=transform),
#             batch_size=batch_size, shuffle=True)
#     elif dataset == 'lsun-bed':
#         data_loader = DataLoader(
#             datasets.LSUN('data/lsun', classes=['bedroom_train'], transform=transform),
#             batch_size=batch_size, shuffle=True)
#
#     return data_loader


# # 查看打印face_landmarks.csv的前四个锚点。
# import pandas as pd
# landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
#
# n = 65
# img_name = landmarks_frame.iloc[n, 0]
# landmarks = landmarks_frame.iloc[n, 1:].values
# landmarks = landmarks.astype('float').reshape(-1, 2)
#
# print('Image name: {}'.format(img_name))
# print('Landmarks shape: {}'.format(landmarks.shape))
# print('First 4 Landmarks: {}'.format(landmarks[:4]))


# # 在指定人脸上显示锚点。
# import pandas as pd
# import matplotlib.pyplot as plt
# from skimage import io, transform
# import os
#
# landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
#
# n = 32
# img_name = landmarks_frame.iloc[n, 0]
# landmarks = landmarks_frame.iloc[n, 1:].values
# landmarks = landmarks.astype('float').reshape(-1, 2)
#
#
# def show_landmarks(image, landmarks):
#     """显示带有锚点的图片"""
#     plt.imshow(image)
#     plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
#     plt.pause(0.001)
#
# plt.figure()
# show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
#                landmarks)
# plt.show()


# # Pytorch自定义人脸数据集类。
# from __future__ import print_function, division
# import os
# import pandas as pd
# from skimage import io, transform
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
#
#
# class FaceLandmarksDataset(Dataset):
#
#
#     def __init__(self, csv_file, root_dir, transform=None):
#
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample


# # 实例化FaceLandmarksDataset类，可视化查看数据集。
# from __future__ import print_function, division
# import os
# import pandas as pd
# from skimage import io, transform
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
#
#
# class FaceLandmarksDataset(Dataset):
#
#
#     def __init__(self, csv_file, root_dir, transform=None):
#
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
#
#
# face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
#                                     root_dir='data/faces/')
#
# fig = plt.figure()
# def show_landmarks(image, landmarks):
#     """显示带有锚点的图片"""
#     plt.imshow(image)
#     plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
#     plt.pause(0.001)
#
# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#
#     print(i, sample['image'].shape, sample['landmarks'].shape)
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_landmarks(**sample)
#
#     if i == 3:
#         plt.show()
#         break


# Pytorch预处理faces数据集。
# import torch
# from skimage import io, transform
# class Rescale(object):
#
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size
#
#         new_h, new_w = int(new_h), int(new_w)
#
#         img = transform.resize(image, (new_h, new_w))
#
#         # h and w are swapped for landmarks because for images,
#         # x and y axes are axis 1 and 0 respectively
#         landmarks = landmarks * [new_w / w, new_h / h]
#
#         return {'image': img, 'landmarks': landmarks}
#
#
# class RandomCrop(object):
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         h, w = image.shape[:2]
#         new_h, new_w = self.output_size
#
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)
#
#         image = image[top: top + new_h,
#                       left: left + new_w]
#
#         landmarks = landmarks - [left, top]
#
#         return {'image': image, 'landmarks': landmarks}
#
#
# class ToTensor(object):
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         # 交换颜色轴因为
#         # numpy包的图片是: H * W * C
#         # torch包的图片是: C * H * W
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'landmarks': torch.from_numpy(landmarks)}



# # torchvision.transforms.Compose实现组合数据预处理。
# import pandas as pd
# import matplotlib.pyplot as plt
# from skimage import io, transform
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from torchvision import transforms, utils
#
# class FaceLandmarksDataset(Dataset):
#
#
#     def __init__(self, csv_file, root_dir, transform=None):
#
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
#
#
# class Rescale(object):
#
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size
#
#         new_h, new_w = int(new_h), int(new_w)
#
#         img = transform.resize(image, (new_h, new_w))
#
#         # h and w are swapped for landmarks because for images,
#         # x and y axes are axis 1 and 0 respectively
#         landmarks = landmarks * [new_w / w, new_h / h]
#
#         return {'image': img, 'landmarks': landmarks}
#
#
# class RandomCrop(object):
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         h, w = image.shape[:2]
#         new_h, new_w = self.output_size
#
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)
#
#         image = image[top: top + new_h,
#                       left: left + new_w]
#
#         landmarks = landmarks - [left, top]
#
#         return {'image': image, 'landmarks': landmarks}
#
#
# class ToTensor(object):
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         # 交换颜色轴因为
#         # numpy包的图片是: H * W * C
#         # torch包的图片是: C * H * W
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'landmarks': torch.from_numpy(landmarks)}
#
#
#
# face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
#                                     root_dir='data/faces/')
#
#
# def show_landmarks(image, landmarks):
#     plt.imshow(image)
#     # plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
#     plt.pause(0.001)
#
# scale = Rescale(256)
# crop = RandomCrop(128)
# composed = transforms.Compose([Rescale(256),
#                                RandomCrop(224)])
#
# # 在样本上应用上述的每个变换。
# fig = plt.figure()
# sample = face_dataset[32]
# for i, tsfrm in enumerate([scale, crop, composed]):
#     transformed_sample = tsfrm(sample)
#
#     ax = plt.subplot(1, 3, i + 1)
#     plt.tight_layout()
#     ax.set_title(type(tsfrm).__name__)
#     show_landmarks(**transformed_sample)
#
# plt.show()
