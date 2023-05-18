
"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, size):
    # create a group with all processors
    group = dist.new_group(list(range(size)))
    tensor = torch.tensor([rank], dtype=torch.float32)
    # create an empty list we will use to hold the gathered values
    tensor_list = [torch.empty(1) for i in range(size)]
    # sending all tensors to the others
    dist.all_gather(tensor_list, tensor, group=group)
    # all ranks will have [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
    print(f"[{rank}] data = {tensor_list}")

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
