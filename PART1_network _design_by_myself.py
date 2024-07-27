import torch
from torch import nn# 神经网络
from torch.optim import SGD# 优化器
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
# 设置 Matplotlib 后端
matplotlib.use('Agg')

# 数据准备(X,Y)我这里以乘法为例
x = [[2, 2], [3, 4], [5, 6], [7, 8]]
y = [[4], [12], [30], [56]]
# 将数据转换为张量
X = torch.tensor(x).float()
Y = torch.tensor(y).float()

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)


# 定义神经网络
class MyNeuralNet(nn.Module):
    def __init__(self):# 初始化网络结构
        super().__init__()# 继承父类 nn.Module
        self.input_to_hidden_layer = nn.Linear(2, 8)# 输入层到隐藏层的线性层
        self.hidden_layer_activation = nn.ReLU()# 隐藏层激活函数
        self.hidden_to_output_layer = nn.Linear(8, 1)# 隐藏层到输出层的线性层

    def forward(self, x):# 前向传播
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x


# 初始化神经网络
mynet = MyNeuralNet().to(device)# 将网络结构放置在GPU上

# 打印神经网络结构和参数
print("网络结构:")
print(mynet)
print("\n网络参数:")
for param in mynet.parameters():
    print(param)

# 定义损失函数
loss_func = nn.MSELoss()# 均方差损失函数

# 定义优化器
opt = SGD(mynet.parameters(), lr=0.001)# 随机梯度下降法，学习率为0.001

# 训练循环
loss_history = []
num_epochs = 50# 训练轮数为50

for epoch in range(num_epochs):
    opt.zero_grad()  # 梯度清零
    output = mynet(X)  # 前向传播
    loss_value = loss_func(output, Y)  # 计算损失
    loss_value.backward()  # 反向传播
    opt.step()  # 更新参数
    loss_history.append(loss_value.item())  # 记录损失值
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss_value.item()}')# 打印损失值

# 绘制损失曲线
plt.plot(loss_history)
plt.title('Loss variation over increasing epochs')
plt.xlabel('Epochs（训练周期）')
plt.ylabel('Loss Value（损失值）')
plt.savefig('loss_curve.png')
