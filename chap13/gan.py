
# # 生成手写体数字图片
# import os
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.utils import save_image
#
# z_dimension = 100
#
#
# transform = transforms.Compose([
#     #将PILImage或者numpy的ndarray转化成Tensor,这样才能进行下一步归一化
#     transforms.ToTensor(),
#     #transforms.Normalize(mean,std)参数：
#     transforms.Normalize([0.5], [0.5]),
# ])
#
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
#
# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
#
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.dis = nn.Sequential(
#             nn.Linear(784, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.dis(x)
#         return x
#
#
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.gen = nn.Sequential(
#             nn.Linear(z_dimension, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 784),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         x = self.gen(x)
#         return x
#
#
#
# def to_img(x):
#     out = 0.5 * (x + 1)
#     out = out.view(-1, 1, 28, 28)
#     return out
#
# D = Discriminator().to('cpu')
# G = Generator().to('cpu')
#
#
# criterion = nn.BCELoss()
# D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
# G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
#
# os.makedirs("MNIST_FAKE", exist_ok=True)
#
# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     #将模型调整到训练状态
#     D.train()
#     G.train()
#     all_D_loss = 0.
#     all_G_loss = 0.
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to('cpu'), targets.to('cpu')
#         #num_img即为图片的数量
#         num_img = targets.size(0)
#         #real的标签是1，fake的标签是0
#         real_labels = torch.ones_like(targets, dtype=torch.float)
#         fake_labels = torch.zeros_like(targets, dtype=torch.float)
#         #把输入的28*28图片压平成784，便于输入D进行运算
#         inputs_flatten = torch.flatten(inputs, start_dim=1)
#
#         # Train Discriminator
#         real_outputs = D(inputs_flatten)
#         real_outputs = real_outputs.squeeze(-1)
#         print('real_outputs size is {}'.format(real_outputs.size()))
#         print('real_labels size is {}'.format(real_labels.size()))
#         D_real_loss = criterion(real_outputs, real_labels)
#
#         z = torch.randn((num_img, z_dimension))
#         fake_img = G(z)
#         fake_outputs = D(fake_img.detach())
#         fake_outputs = fake_outputs.squeeze(-1)
#         D_fake_loss = criterion(fake_outputs, fake_labels)
#
#         D_loss = D_real_loss + D_fake_loss
#         D_optimizer.zero_grad()
#         D_loss.backward()
#         D_optimizer.step()
#
#         # Train Generator
#         z = torch.randn((num_img, z_dimension))
#         fake_img = G(z)
#         G_outputs = D(fake_img)
#         G_outputs = G_outputs.squeeze(-1)
#         G_loss = criterion(G_outputs, real_labels)
#         G_optimizer.zero_grad()
#         G_loss.backward()
#         G_optimizer.step()
#
#         all_D_loss += D_loss.item()
#         all_G_loss += G_loss.item()
#         print('Epoch {}, d_loss: {:.6f}, g_loss: {:.6f} '
#               'D real: {:.6f}, D fake: {:.6f}'.format
#               (epoch, all_D_loss/(batch_idx+1), all_G_loss/(batch_idx+1),
#                torch.mean(real_outputs), torch.mean(fake_outputs)))
#
#     # Save generated images for every epoch
#     fake_images = to_img(fake_img)
#     save_image(fake_images, 'MNIST_FAKE/fake_images-{}.png'.format(epoch + 1))
#
# for epoch in range(100):
#     train(epoch)


# 生成人像数据图片
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# 为再现性设置随机seem
manualSeed = 999
#manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = r"J:\data\image and vision computing\Celeb Faces Attributes Dataset (CelebA)\archive"
workers = 0
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 6
lr = 0.0002
beta1 = 0.5
ngpu = 1


dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# 创建加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# 选择运行在上面的设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# # 绘制部分的输入图像
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



# 生成器代码
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是Z，进入卷积
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
# print(netG)



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# 创建判别器
netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
# print(netD)


# 初始化BCELoss函数
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


if __name__ =='__main__':
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # 对于数据加载器中的每个batch
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            output = output.to(torch.float32)
            label = label.to(torch.float32)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    # 从数据加载器中获取一批真实图像
    real_batch = next(iter(dataloader))

    # 绘制真实图像
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # 在最后一个epoch中绘制伪图像
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()



