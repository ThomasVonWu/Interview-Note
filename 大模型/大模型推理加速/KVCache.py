# 如何理解attention中的KV Cache， 相比较传统attention其加速机制是什么，手撕代码实现？

import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class SelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, bias=False, dropout=0.1, block_size=1024):
        super.__init__()
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        self.n_embed = n_embed
        self.n_head = n_head
        self.dropout = dropout

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )  # (bs,1,block_size, block_size) block_size代表的是模型处理的最大长度

    def forward(self, x):
        bs, seq_len, c = x.size()

        # 计算Q、K、V
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        # 拆分多头
        q = q.view(bs, seq_len, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(bs, seq_len, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(bs, seq_len, self.n_head, c // self.n_head).transpose(1, 2)

        # Q和K计算注意力矩阵
        attn = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(c // self.n_head)
        )  # (bs, self.n_head, seq_len, seq_len)
        attn = attn.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("inf"))
        attn = F.softmax(attn, dim=-1)  # (bs, self.n_head, seq_len, seq_len)
        attn = self.attn_dropout(attn)

        # 聚合value
        y = attn @ v  # (bs, self.n_head, seq_len, c // self.n_head)
        y = y.transpose(1, 2).contiguous().view(bs, seq_len, c)  # (bs, seq_len, c)
        y = self.resid_dropout(self.c_proj(attn))
        return y


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, bias=False, dropout=0.1, block_size=1024):
        super.__init__()
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        self.n_embed = n_embed
        self.n_head = n_head
        self.dropout = dropout

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )  # (bs,1,block_size, block_size) block_size代表的是模型处理的最大长度

    def forward(self, x, cache: Tuple[torch.tensor, torch.tensor]):
        bs, seq_len, c = x.size()

        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        # 变更点1
        if cache is not None:
            pk, pv = cache
            k = torch.cat([pk, k], dim=-2)  # (bs, seq_len' + seq_len, c)
            v = torch.cat([pv, v], dim=-2)  # (bs, seq_len' + seq_len, c)
            cache = (k, v)
        kv_seq_len = k.shape[-2]

        # 变更点2
        q = q.view(bs, seq_len, self.n_head, c // self.n_head).transpose(1, 2)
        # k = k.view(bs, seq_len, self.n_head, c // self.n_head).transpose(1, 2)
        # v = v.view(bs, seq_len, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(bs, kv_seq_len, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(bs, kv_seq_len, self.n_head, c // self.n_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(c // self.n_head)
        )  # (bs, self.n_head, seq_len, kv_seq_len)
        attn = attn.masked_fill(
            self.mask[:, :, kv_seq_len - seq_len : kv_seq_len, :kv_seq_len] == 0,
            float("inf"),
        )
        attn = F.softmax(attn, dim=-1)  # (bs, self.n_head, seq_len, kv_seq_len)
        attn = self.attn_dropout(attn)

        y = attn @ v  # (bs, self.n_head, seq_len, c // self.n_head)
        y = y.transpose(1, 2).contiguous().view(bs, seq_len, c)  # (bs, seq_len, c)
        y = self.resid_dropout(self.c_proj(attn))
        return y, cache


# reference:
# https://www.zhihu.com/question/596900067/answer/3437033433
