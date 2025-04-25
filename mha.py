import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,h):
        super().__init__()
        self.d_model = d_model
        self.h = h

        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model,d_model,bias=False)
        self.w_k = nn.Linear(d_model,d_model,bias=False)
        self.w_v = nn.Linear(d_model,d_model,bias=False)
        self.w_o = nn.Linear(d_model,d_model,bias=False)

    def attention(self,query,key,value,mask=None):
        # batch_size h seq_len seq_len
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(self.d_k)
        # 掩码机制 为0则直接赋值-1e9
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0,-1e9)
        # batch_size h seq_len seq_len
        attention_scores = F.softmax(attention_scores, dim=-1)
        # batch_size h seq_len d_k
        return (attention_scores @ value), attention_scores
    
    def forward(self,q,k,v,mask=None):
        # batch_size seq_len d_model
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # batch_size seq_len h d_k -> batch_size h seq_len d_k
        query = query.view(query.size(0),query.size(1),self.h,self.d_k).transpose(1,2)
        key = key.view(key.size(0),key.size(1),self.h,self.d_k).transpose(1,2)
        value = value.view(value.size(0),value.size(1),self.h,self.d_k).transpose(1,2)

        # batch_size h seq_len d_k
        x,self.attention_scores = self.attention(query,key,value,mask)
        # contiguous()是为了保证内存连续性 transpose不改变存储顺序但view要求存储顺序连续，因此中间需要contiguous()
        # batch_size h seq_len d_k -> batch_size seq_len h*d_k=d_model
        x = x.transpose(1,2).contiguous().view(x.size(0),-1,self.h*self.d_k)

        # batch_size seq_len d_model
        return self.w_o(x)
    
# 测试
batch_size = 2
seq_len = 4
d_model = 8
h = 2
embedding = torch.randn(batch_size,seq_len,d_model)
mha = MultiHeadAttention(d_model,h)
output = mha(embedding,embedding,embedding)
print(output.size())

