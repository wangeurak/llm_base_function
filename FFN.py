import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 测试
batch_size = 2
seq_len = 4
d_model = 8
d_ff = 16
embedding = torch.randn(batch_size,seq_len,d_model)
ffn = FFN(d_model,d_ff)
output = ffn(embedding)
print(output.size())