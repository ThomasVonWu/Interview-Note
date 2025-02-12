import torch
import torch.nn as nn
import torch.nn.functional as F

"""
简单的写一个transformer 推理类, next token 预测, 注意attn_mask的设置。

"""


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=1000,
        embed_size=32,
        hidden_size=64,
        num_heads=4,
        block_size=1024,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=num_heads, batch_first=True
        )
        self.proj = nn.Linear(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.num_heads = num_heads

        self.register_buffer(
            "mask", torch.triu(torch.ones((block_size, block_size)), diagonal=1).bool()
        )

    def forward(self, input_ids, past_key_values=None):
        x = self.embedding(input_ids)  # (b=1, s=1) => (b=1, s=1, embed_size=32)

        b = x.size(0)
        seq_len = x.size(1)
        # attn_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()

        if past_key_values is not None:
            key = value = torch.cat(
                [past_key_values[..., -embed_size:], x], dim=1
            )  # (b, s+s', c) 假设kv 的 d 维度相等
        else:
            key = value = x

        past_key_values = torch.cat([key, value], dim=-1)  # (b, s+s', 2c)

        kv_seq_len = key.shape[-2]

        attn_mask = self.mask[kv_seq_len - seq_len : kv_seq_len, :kv_seq_len]
        attn_mask = attn_mask.expand(b * self.num_heads, seq_len, kv_seq_len)

        output, _ = self.attention(
            query=x, key=key, value=value, attn_mask=attn_mask
        )  # (b,s+s',embed_size)
        output = self.proj(output)
        logits = self.fc(output[:, -1:, :])  # 只取最后一个位置输出

        return logits, past_key_values


# 初始化模型
vocab_size = 1000
embed_size = 32
hidden_size = 64
model = SimpleTransformer(vocab_size, embed_size, hidden_size)
model.eval()

# 设定初始输入
input_ids = torch.tensor([[1]])  # 假设 1 代表起始 token # (b, s)
past_key_values = None
next_token = input_ids

generated = [next_token]
print(generated)

# 进行推理，逐步生成三个 token
num_tokens_to_generate = 3
for _ in range(num_tokens_to_generate):
    with torch.no_grad():
        logits, past_key_values = model(
            next_token, past_key_values=past_key_values
        )  # logits (1,1,1000)

    next_token = torch.argmax(logits, dim=-1)  # (b, s)

    # 拼接输出
    generated.append(next_token)


print("Generated token sequence:", generated)
