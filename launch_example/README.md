
## Ref

```
  https://blog.csdn.net/magic_ll/article/details/122359490

多机多卡 的分布式
在0号机器上调用
python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr='172.18.39.122' --master_port='29500' train.py
在1号机器上调用
python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr='172.18.39.122' --master_port='29500' train.py
注意:

命令中的【–master_addr=‘172.18.39.122’】指的是0号机器的IP，在0号机器上运行的命令中【node_rank】必须为0
只有当【nnodes】个机器全部运行，代码才会进行分布式的训练操作，否则一直处于等待状态

======================================================
单机多卡 训练
只需要说明 想要使用GPU的[编号]、[数量]即可。由于不需要不同机器之间的通信，就少了其余4个参数的设定
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 train.py

===================================================================

单机单卡训练
当工程提供的是分布式训练代码，但我们只想用单张显卡运行。
机器上只有一张显卡：
python -m torch.distributed.launch train.py
机器上有多张显卡：
export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.launch train.py

```
