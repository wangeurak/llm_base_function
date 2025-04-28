import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
# 官方的实现方式与数学定义并非完全一致(考虑内存的连续性与分块操作)，此处采取与数学定义一致的方案
class MyRope(nn.Module):
    def __init__(self,head_dim,max_seq_len=512):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0/(10000**(torch.arange(0,self.head_dim,2).float()/head_dim))
        self.register_buffer('inv_freq',inv_freq)

    def forward(self,x):
        # batch_size,seq_len,n_heads,head_dim
        batch_size,seq_len,_,_ = x.size()
        device = x.device
        pos = torch.arange(seq_len,device=device).float()
        # seq_len * head_dim/2
        freqs = torch.einsum('i,j->ij',pos,self.inv_freq)
        #(1,seq_len,1,head_dim/2)
        cos = freqs.cos().unsqueeze(0).unsqueeze(2)
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)

        x_rotated = self._apply_rope(x,cos,sin)

        return x_rotated
    def _apply_rope(self,x,cos,sin):
        x_even = x[...,::2]
        x_odd = x[...,1::2]
        rotated_1 = x_even*cos - x_odd*sin
        rotated_2 = x_even*sin + x_odd*cos
        x_rotated = torch.stack([rotated_1,rotated_2],dim=-1)
        # 内存不连续时view会报错
        return x_rotated.flatten(-2)