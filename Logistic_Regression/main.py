import torch
import torch.nn as nn
import matplotlib as plt
import numpy as np
import swanlab
torch.manual_seed(10)

# step1.生成数据
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums,2)                  # 类别0 数据 shape（100，2）
x0 = torch.normal(mean_value*n_data,1) + bias       # 类别0 标签 shape（100，1）
y0 = torch.zeros(sample_nums)                       

x1 = torch.normal(-mean_value*n_data,1) + bias      # 类别1 数据 shape（100，2）
y1 = torch.ones(sample_nums)                        # 类别1 标签 shape（100，1）

train_x =torch.cat((x0,x1),0)
train_y =torch.cat((y0,y1),0)

# step2.选择模型
class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.features = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

lr_net =LR()                                       # 实例化逻辑回归模型

# step3.选择损失函数
loss_fn = nn.BCELoss()


# step4.选择优化器
lr = 0.01                                         # 学习率

optimizer =torch.optim.SGD(lr_net.parameters(),lr=lr,momentum=0.9)

# step5.实验记录初始化
swanlab.init(experiment_name="Basic_Logistic",config={'learning_rate': 0.01},
  )


# step6.模型训练
for iteration in range(1000):
    # 前向传播
    y_pred = lr_net(train_x)
    
    # 计算 loss
    loss = loss_fn(y_pred.squeeze(),train_y)
    
    # 计算 准确率
    mask =y_pred.ge(0.5).float().squeeze()          # 以0.5为阈值分类
    correct = (mask == train_y).sum()               # 计算正确预测的样本个数
    acc = correct.item() / train_y.size(0)          # 计算分类准确率
        
    # 记录实验数据
    swanlab.log({"loss": loss,"accuracy":acc})
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    