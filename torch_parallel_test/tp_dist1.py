import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class ParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features // world_size))#列切
        self.bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


def worker(rank):
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=rank, world_size=4)

    linear = ParallelLinear(256, 4096, 4).cuda()
    
    # 创建和接收数据
    if rank == 0:
        data_in = torch.randn(100, 256).cuda()
        chunks = torch.chunk(data_in, 4, dim=1)  # 按照第二个维度分割数据
        for i in range(1, 4):
            dist.send(chunks[i].contiguous(), dst=i)
        data_in = chunks[0]
    else:
        data_in = torch.zeros(100, 64).cuda()
        dist.recv(data_in, src=0)

    data_out = linear(data_in)

    tensor_list = [torch.zeros_like(data_out) for _ in range(4)]
    dist.all_gather(tensor_list, data_out)

    if rank == 0:
        print(torch.cat(tensor_list, dim=1).shape)

def main():
    torch.multiprocessing.spawn(worker, nprocs=4)

if __name__ == '__main__':
    main()
