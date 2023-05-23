import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input):
        return self.linear(input)

# 在每个 GPU 上分别创建一个线性层
my_linears = [MyLinear(256, 4096).to(f'cuda:{i}') for i in range(4)]

data_in = torch.randn([100, 1024]).to('cuda:0')

# 把 data_in 的第二维拆分成4份，分别放在4个不同的GPU上
inputs = data_in.split(256, dim=1)  # 100 / 4 = 25
inputs = [input.to(f'cuda:{i}') for i, input in enumerate(inputs)]

# 并行地在4个不同的GPU上执行线性运算
outputs = [my_linears[i](inputs[i]) for i in range(4)]

# 把4个不同的结果移动到同一个GPU（这里是 'cuda:0'）
outputs = [output.to('cuda:0') for output in outputs]

data_out = torch.cat(outputs, dim=1).to('cuda:0')

