import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, input):
        return self.linear(input)

# 创建4个不同的线性层，分别放在4个不同的GPU上
my_linears = [MyLinear(256, 4096).to(f'cuda:{i}') for i in range(4)]

n = 100
# 假设 data_in 的形状是 [n, 256]
data_in = torch.randn([n, 256]).to('cuda:0')

# 把 data_in 拆分成4份，分别放在4个不同的GPU上#默认且切一个维度
inputs = torch.nn.parallel.scatter(data_in, target_gpus=list(range(4)), dim=0)

# 并行地在4个不同的GPU上执行线性运算
outputs = [my_linears[i](inputs[i]) for i in range(4)]

# 把4个不同的结果汇总到一个GPU（这里是 'cuda:0'）
data_out = torch.nn.parallel.gather(outputs, target_device='cuda:0', dim=0)

print(data_out)
