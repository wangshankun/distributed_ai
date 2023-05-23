
## torch实现tensor parallel的行列切

```
1. torch.cuda stream/synchronize级源语和tensor.to(device)方式实现, 最容易侵入同步bug难调
2. torch.nn.parallel配合tensor.to(device)方式实现(单机多卡单线程) ，侵入算子
3. torch.distributed接口实现,需要配合torch.multiprocessing多线程实现(多机多卡多线程)，侵入整网重定义
```
