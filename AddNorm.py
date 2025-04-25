import torch
import torch.nn as nn
import torch.nn.functional as F

class AddNorm(nn.Module):
    def __init__(self,feature_size,dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,residual):
        x = x + self.dropout(residual)
        return self.layer_norm(x)

# 测试
batch_size = 2
seq_len = 4
d_model = 8
embedding = torch.randn(batch_size,seq_len,d_model)
add_norm = AddNorm(d_model)
output = add_norm(embedding,embedding)
print(output.size())