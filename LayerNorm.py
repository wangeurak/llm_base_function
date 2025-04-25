import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self,feature_size,eps=-1e6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(feature_size))
        self.bias = nn.Parameter(torch.zeros(feature_size))
    
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        x = (x-mean)/(std+self.eps)

        return self.alpha*x+self.bias
    
# 测试
batch_size = 2
seq_len = 4
d_model = 8
embedding = torch.randn(batch_size,seq_len,d_model)
layer_norm = LayerNorm(d_model)
output = layer_norm(embedding)
print(output.size())