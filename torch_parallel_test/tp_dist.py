import torch
import torch.nn as nn
import torch.distributed as dist

def worker(rank):
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=rank, world_size=4)

    # 创建一个在当前 GPU 上的线性层
    linear = nn.Linear(256, 4096).cuda()
    data_in = torch.randn(25, 256).cuda()  # 每个进程处理 1/4 的数据

    data_out = linear(data_in)

    # 收集所有进程的结果
    tensor_list = [torch.zeros_like(data_out) for _ in range(4)]
    dist.all_gather(tensor_list, data_out)

    # 只在 rank 0 的进程上打印结果
    if rank == 0:
        print(torch.cat(tensor_list, dim=1).shape)

def main():
    # 创建四个进程，每个进程在一个 GPU 上运行
    torch.multiprocessing.spawn(worker, nprocs=4)

if __name__ == '__main__':
    main()

