'''我们引入一种反馈机制，用来训练那些错误标签，从而降低标签噪声导致的影响'''
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# 定义训练的设备
device = torch.device("cuda")

# 准备数据集
train_data = torchvision.datasets.CIFAR10("./dataset1", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset1", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# print("训练数据集的长度为：{}".format(train_data_size)) # 训练数据集的长度为50000
# print("测试数据集的长度为：{}".format(test_data_size)) # 测试数据集的长度为10000

# 利用DataLoader来加载数据集
train_data_dataloader = DataLoader(train_data, batch_size=64,shuffle=True) # 训练的时候随机采样
test_data_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络模型
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 创建网络模型
module = MyModule()
'''if torch.cuda.is_available():
    module=module.cuda() # 利用GPU进行训练'''
module.to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
'''if torch.cuda.is_available():
    loss_fn=loss_fn.cuda() # 利用GPU进行训练'''
loss_fn.to(device)

# 定义优化器（进行梯度下降法）
learning_rate = 0.01
optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)  # 传入神经网络各层级的参数（卷积层、线性层的权重），从而进行优化

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮次
epoch = 10
# 添加tensorboard
writer = SummaryWriter("logs_all_correct2_model")
start_time = time.time()  # 记录开始时间
for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i + 1))

    # 训练步骤开始
    module.train()  # 让网络进入训练模式
    for TrainData in train_data_dataloader:
        imgs, targets = TrainData  # 这里的数据imgs和targerts都可以用cuda训练
        '''if torch.cuda.is_available():
            imgs=imgs.cuda()
            targets=targets.cuda()'''
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = module(imgs)
        loss = loss_fn(outputs, targets) # 计算二进制交叉熵损失函数

        # 引入反馈机制,首先检测并过滤错误的标签
        predicted_labels = torch.argmax(outputs, dim=1)
        # 这里求使得输出的f(x)最大的x
        # 因为通过神经网络最后一层(可能是归一化过的线性层，也许会有10个输出神经元)，从这10个神经元输出中找到有最大值输出的下标将其作为预测
        incorrect_labels = predicted_labels != targets  # 获取到不正确的预测的索引
        images_correction = imgs[incorrect_labels]  # 获取不正确的预测的输入图像
        labels_correction = targets[incorrect_labels]  # 获取不正确的预测的实际标签

        # 将错误标签的样本重新加入到训练集中/单独训练具有错误标签的样本
        # new_images = torch.cat((imgs, images_correction))
        new_images=images_correction
        # new_labels = torch.cat((targets, labels_correction))
        new_labels=labels_correction

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 通过梯度优化网络权重

        # 获取到错误标签的数据我们再训练一次进行反向传播优化权重
        new_outputs=module(new_images)
        new_loss=loss_fn(new_outputs,new_labels)
        optimizer.zero_grad()
        new_loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练次数: {}, Loss: {}, 耗费时间: {}".format(total_train_step, loss.item(), end_time - start_time))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 通过测试数据集的损失函数值来反映是否训练完全，注意这里测试的时候不能够用优化器调优，所以用代码with torch.no_grad()
    # 测试步骤开始
    module.eval()  # 让网络进入评估模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for TestData in test_data_dataloader:
            imgs, targets = TestData
            '''if torch.cuda.is_available():
                imgs=imgs.cuda()
                targets=targets.cuda()'''
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = module(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

        print("整体测试集上的Loss:{}".format(total_test_loss))
        print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
        total_test_step = total_test_step + 1

    torch.save(module, "module_{}.pth".format(i + 1))
    print("模型已保存")

writer.close()
