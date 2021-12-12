import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


EPOCH = 1
BATCH_SIZE = 64
LR = 0.001 

train_dataset = datasets.MNIST(
    root='data', 
    train=True,  
    transform=transforms.ToTensor(),  # 因为此数据集是黑白的，修改原始数据集的格式，把0-255的颜色映射到0、1两种
    download=False  # 第一次运行程序需要设置为True来下载数据集
    )

# 看第一张图片长啥样
# plt.imshow(train_dataset.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_dataset.train_labels[0])
# plt.show()

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=0,
    )

test_dataset = datasets.MNIST(
    root='data',
    train=False,  # 设置为False，表示是测试数据集，就不需要训练
    download=True
    )
test_x = Variable(torch.unsqueeze(test_dataset.data, dim=1)).type(torch.FloatTensor)[:2000]/255.
test_y = test_dataset.targets[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 输入的图片是 28 * 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 二维卷积滤波器filter
                in_channels=1,  # 因为是黑白的，就只有1个通道，彩色的就有RGB 3个通道
                out_channels=16,  # 相当于filter的个数，16个不同的滤波器
                kernel_size=5,  # filter 宽高设置为 5*5
                stride=1,  # filter每次移动的步长
                padding=2,  # 扩展图片的边界，扩展2层，值设置为0，0是黑色，255是白色;
            ),
            # 上一个filter的输入图片宽度是28，padding填充后变成32，滤波器是5*5，stride是1，padding是2
            # 那么图片宽度的输出就是 (pic_width - filter_width + padding *2) / stride + 1 = (28 - 5 + 4) / 1 +1 = 28
            # 高度也是一样的计算方式
            # filter 处理后，-> 16个通道的图片，宽高 28 * 28
            nn.ReLU(),  # 设置激活函数，不变 -> (16，28，28)
            nn.MaxPool2d(
                kernel_size=2,  # 相当于 2*2 的filter，选出 2*2 里最大的那个值
            ),  # 设置Max池化层，变成 -> (16, 14, 14)
        )

        self.conv2 = nn.Sequential(
            # 因为conv1卷积层输出了16层的图片，所以这一层的接收层in_channels就是16
            # 把 16 层加工成 32 层
            nn.Conv2d(16, 32, 5, 1, 2),  # -> (32, 14, 14)
            nn.ReLU(),  # -> (32, 14, 14)
            nn.MaxPool2d(2)  # -> (32, 7, 7)
        )
        self.out = nn.Linear(in_features=32 * 7 * 7, out_features=10)  # 手写体总共0-9这10类

    # 把图片拉平成一维向量
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)  # (batch, 32*7*7)
        return self.out(x)

cnn=CNN()
# print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)
        out_put = cnn(b_x)
        loss = loss_func(out_put, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每 50 步看一下训练效果
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.4f' % accuracy)

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction')
print(test_y[:10].numpy(), 'real')