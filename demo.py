import torch
import torch.utils.data
from torch.nn.functional import *
import matplotlib.pyplot as plt


LR = 0.01  # 学习率
BATCH_SIZE = 32
EPOCH = 5

# 造点假数据，生成一元二次函数的点，加一点随机浮动
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # unsqueeze 把一维tensor升到二维
y = x.pow(2) + 0.2*torch.rand(x.size())

# 批训练，DataLoader把数据集打包成一批一批的，每批数据集的最大数据条数就是batch_size
torch_dataset = torch.utils.data.TensorDataset(x, y)
loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,)


# 定义网络模型
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)

# 定义损失函数, 均方差损失函数适用于本案例的回归问题
loss_func = torch.nn.MSELoss()

# 定义优化器用于加速网络训练, 随机梯度下降(stochastic gradient descent)
# 把 net.parameters() 传给它，在后续的step步骤优化器优化的就是传入的这些参数了
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

plt.ion()

for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    # 每次读取的是 DataLoader 中的一批数据，按批次训练
    # step 是批次序号，(b_x, b_y)是这一批数据的输入x向量和相应的标签
    for step, (b_x, b_y) in enumerate(loader):
        # x参数输入网络，从网络输出的值就是与输入值对应的输出预测值
        prediction = net(x)
        # 预测值和真实值的差值就是损失值loss
        # 训练的方向就是 loss 最小化问题
        loss = loss_func(prediction, y)
        # 每次训练都会把梯度保留在 optimizer 里，所以对于新批次的数据训练需要把梯度值清零
        optimizer.zero_grad()
        # 损失反向传播，计算出梯度
        loss.backward()
        # 以指定的学习率来优化更新网络中的梯度参数
        optimizer.step()

        if step % 5 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
            plt.pause(0.1)

plt.ioff()
plt.show()

# 保存训练完的模型
torch.save(net.state_dict(), "net_params.pkl")