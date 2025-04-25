import torch
import torch.nn as nn
import torch.nn.functional as F

# 仅用方差进行缩放 仅保留缩放参数 因为不减均值 所以偏移参数不必要
class RmsNorm(nn.Module):
    def __init__(self,feature_size,eps=1e-6):
        super().__init__()
        self.eps = eps
        # nn.Parameter自动注册为模型的可训练参数
        self.alpha = nn.Parameter(torch.ones(feature_size))
    
    def forward(self,x):
        # mean_2 batch_size seq_len 1
        # keepdim 保持维度 能进行后续的广播操作
        mean_2 = x.pow(2).mean(-1,keepdim=True)
        x = x*torch.rsqrt(mean_2+self.eps)

        return self.alpha*x
    
# 测试
batch_size = 2
seq_len = 4
d_model = 8
embedding = torch.randn(batch_size,seq_len,d_model)
rms_norm = RmsNorm(d_model)
output = rms_norm(embedding)
print(output.size())
print(output)