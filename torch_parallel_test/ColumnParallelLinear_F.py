import torch
import torch.nn as nn
import torch.nn.functional as F

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size

        self.weight = nn.Parameter(torch.Tensor(out_features // world_size, in_features))#列切
        self.bias = nn.Parameter(torch.Tensor(out_features // world_size))

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


# 创建4个不同的线性层，分别放在4个不同的GPU上
my_linears = [ColumnParallelLinear(2048, 8192, 4).to(f'cuda:{i}') for i in range(4)]

# data_in广播到4个不同gpu上
data_in = torch.randn([4, 512, 2048])
datas_in = [data_in.to(f'cuda:{i}') for i in range(4)]

# 并行地在4个不同的GPU上执行线性运算
outputs = [my_linears[i](datas_in[i]) for i in range(4)]

# 把4个不同的结果汇总到一个GPU（这里是 'cuda:0'）
data_out = torch.nn.parallel.gather(outputs, target_device='cuda:0', dim=2)

print(data_out.shape)
