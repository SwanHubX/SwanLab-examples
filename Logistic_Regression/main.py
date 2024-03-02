import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from torchview import draw_graph
import subprocess
graphviz.set_jupyter_format('png')
import swanlab
torch.manual_seed(10)

# step1.生成数据
batch_size = 1
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

model_graph = draw_graph(lr_net, graph_name="lr_net",input_size=(batch_size, 2), device="cpu",save_graph=True)


def dot_to_png(dot_file_path, output_path):
    command = f"dot -Tpng {dot_file_path} -o {output_path}"
    subprocess.run(command, shell=True)

# 示例用法
dot_file_path = "lr_net.gv"  # DOT文件路径
output_path = "lr_net.png"  # 输出PNG图像路径

dot_to_png(dot_file_path, output_path)

# step3.选择损失函数
loss_fn = nn.BCELoss()


# step4.选择优化器
lr = 0.01                                         # 学习率

optimizer =torch.optim.SGD(lr_net.parameters(),lr=lr,momentum=0.9)

# step5.实验记录初始化
swanlab.init(experiment_name="Basic_Logistic",config={'learning_rate': 0.01},
  )

swanlab.log({'modelgraph':swanlab.Image(output_path)})


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
    
    # 清空梯度
    optimizer.zero_grad()
    
    if iteration % 20 == 0:



        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()
       
        swanlab.log({"test": swanlab.Image(plt)})
        plt.clf() 


        if acc > 0.99:
            break
    
    