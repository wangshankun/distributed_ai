import torch
import torch.nn.functional as F

# 创建多个 CUDA 流
streams = [torch.cuda.Stream(device=i) for i in range(4)]

# 创建输入数据和权重
out = torch.randn(32, 512, 2048).cuda() #32个输入token，词向量512
wi_data = torch.randn(8192, 2048).cuda()
bi_data = torch.randn(8192).cuda()
wo_data = torch.randn(2048, 8192).cuda()
bo_data = torch.randn(2048).cuda()

#out_chunks = torch.chunk(out, 1, dim=0) #MLP 输入不切割，只切权重
#实现MLP算子的tensor parallel，将重分割为多个(例子是4个)部分
wi_data_chunks = torch.chunk(wi_data, 4, dim=0)#mlp第一个liner的权重列切
bi_data_chunks = torch.chunk(bi_data, 4, dim=0)#mlp第一个liner的权重列切

wo_data_chunks = torch.chunk(wo_data, 4, dim=1)#mlp第二个liner的权重行切

# 在每个 CUDA 流上并行执行线性计算
results = []
for i in range(4):
    with torch.cuda.stream(streams[i]):
        out = out.to(device=i)
        bo_data = bo_data.to(device=i)

        wi_data_chunk = wi_data_chunks[i].to(device=i)
        bi_data_chunk = bi_data_chunks[i].to(device=i)
        wo_data_chunk = wo_data_chunks[i].to(device=i)

        out = F.linear(out, wi_data_chunk, bi_data_chunk)
        out = F.relu(out, inplace=True)
        out = F.linear(out, wo_data_chunk, bo_data)
        results.append(out)
# 等待所有 CUDA 流完成计算
torch.cuda.synchronize()

# 收集所有的结果并求和
data_out = torch.sum(torch.stack([r.to('cuda:0') for r in results]), dim=0)

