# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import paddle
from paddle.nn import Conv2D, MaxPool2D
import paddle.nn.functional as F
import matplotlib.pyplot as plt 


# 加载MNIST数据 MNIST数据属于 torchvision 包自带的数据,可以直接接调用
train_dataset = dsets.MNIST(root='./data',    # 文件存放路径
                            train=True,        # 提取训练集
                            # 将图像转化为 Tensor，在加载數据时，就可以对图像做预处理
                            transform=transforms.ToTensor(),
                            download=True)      # 当找不到文件的时候，自动下載
# 加载测试数据集
test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# 定义vgg网络
class VGG(paddle.nn.Layer):
    def __init__(self):
        super(VGG, self).__init__()

        in_channels = [1, 64, 128, 256, 512, 512]
        # 定义第一个block，包含两个卷积
        self.conv1_1 = Conv2D(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)     
        self.conv1_2 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)     
        # 定义第二个block，包含两个卷积
        self.conv2_1 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)  
        self.conv2_2 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)
        # 定义第三个block，包含三个卷积
        self.conv3_1 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_2 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_3 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        # 定义第四个block，包含三个卷积
        self.conv4_1 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_2 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_3 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        # 定义第五个block，包含三个卷积
        self.conv5_1 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_2 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_3 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)

        # 使用Sequential 将全连接层和relu组成一个线性结构（fc + relu）
        # 当输入为224x224时，经过五个卷积块和池化层后，特征维度变为[512x7x7]
        self.fc1 = paddle.nn.Sequential(paddle.nn.Linear(512 * 7 * 7, 4096), paddle.nn.ReLU())
        self.drop1_ratio = 0.5
        self.dropout1 = paddle.nn.Dropout(self.drop1_ratio, mode='upscale_in_train')
        # 使用Sequential 将全连接层和relu组成一个线性结构（fc + relu）
        self.fc2 = paddle.nn.Sequential(paddle.nn.Linear(4096, 4096), paddle.nn.ReLU())

        self.drop2_ratio = 0.5
        self.dropout2 = paddle.nn.Dropout(self.drop2_ratio, mode='upscale_in_train')
        self.fc3 = paddle.nn.Linear(4096, 10)

        self.relu = paddle.nn.ReLU()
        self.pool = MaxPool2D(stride=2, kernel_size=2)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool(x)

        x = paddle.flatten(x, 1, -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class MyTrain():
    
    def __init__(self,model, opt,loss_fun,epochs, gen_path_name):
        self.model = model
        self.opt = opt
        self.epochs = epochs
        self.gen_path_name = gen_path_name
        
        #绘制准确率和损失图片
        self.x = []
        self.y = []
        self.z = []
        self.p = []

    def train(self, train_dataset, test_dataset, batch_size):

        #定义训练集
        train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        #开启GPU
        use_gpu = True
        paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

        model.train()
        for epoch in range(self.epochs):
            #训练
            acc_list = []
            train_loss_list = []
            for batch_id, data in enumerate(train_loader()):
                x_data = data[0]
                y_data = data[1]
                predicts = model(x_data)
                y_data = y_data.numpy().reshape(len(y_data), -1)
                y_data = paddle.to_tensor(y_data)
                

                #计算损失
                acc_np = paddle.metric.accuracy(predicts, y_data)
                loss = loss_fun(predicts, y_data)
                train_loss_list.append(loss.numpy())
                loss.backward()
                if batch_id % 300 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {}, acc is : {}".format(epoch, batch_id, loss.numpy(), acc_np.numpy()))
                self.opt.step()
                self.opt.clear_grad()

            train_loss_mean = np.array(train_loss_list).mean()
            self.p.append(train_loss_mean)

            #评估
            acc_list = []
            avg_loss_list = []
            for batch_id, data in enumerate(test_loader()):
                x_data = data[0]
                y_data = data[1]
                predicts = model(x_data)
                y_data = y_data.numpy().reshape(len(y_data), -1)
                y_data = paddle.to_tensor(y_data)
                
                acc_np = paddle.metric.accuracy(predicts, y_data)
                avg_np = loss_fun(predicts, y_data)
                acc_list.append(acc_np.numpy())
                avg_loss_list.append(avg_np.numpy())
            
            acc_val_mean = np.array(acc_list).mean()
            avg_loss_mean = np.array(avg_loss_list).mean()
            self.x.append(epoch)
            self.y.append(acc_val_mean)
            self.z.append(avg_loss_mean)
            print("epoch:%d,平均损失率：%s,平均准确度：%s\n" % (epoch, avg_loss_mean, acc_val_mean))

        self.show_result()
        self.show_evaloss_train_loss()

    def show_result(self):


        #绘制损失图和准确率
        plt.figure(figsize=(25, 15))
        plt.title("Valid Accuracy and Loss")
        plt.ylabel("Accuracy/Loss")
        plt.xlabel("Times")

        plt.plot(self.x, self.y, '.-', label="Valid Accuracy")
        plt.plot(self.x, self.z, '.-', label="Avg Loss")

        plt.legend(loc="best")
        for x_label, y_label in zip(self.x, self.y):
            plt.text(x_label, y_label, (format(y_label, ".3f")), ha="center", va="bottom", fontsize=12)
        for x_label, z_label in zip(self.x, self.z):
            plt.text(x_label, z_label, (format(z_label, ".3f")), ha='center', va='bottom', fontsize=12)

        print("准确率与损失图")
        save_path_name = "output" + '/' + "0" + "_"  + self.gen_path_name + "_" + str(opt._learning_rate)  + ".jpg"
        plt.savefig(save_path_name)
        plt.show()
        plt.close()

        
old = pd.DataFrame(columns = [0.01, 0.008, 0.004, 0.002, 0.001, 0.0008, 0.0004, 0.0001], index = ['vgg'])

for learn in  old.columns:


    #定义参数
    model = VGG()  
    opt = paddle.optimizer.Momentum(learning_rate=learn, momentum=0.9, parameters=model.parameters())
    loss_fun = F.cross_entropy #自定义损失函数
    epcohs = 20 #定义周期数

    #训练
    my_train = MyTrain(model, opt,loss_fun, epcohs, "VGG")
    my_train.train(train_dataset, test_dataset, 100)