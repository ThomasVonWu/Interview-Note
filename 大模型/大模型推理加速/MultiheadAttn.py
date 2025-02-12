import torch

"""
Q1 ) 如何理解 torch.nn.MultiheadAttention 中的 attn_mask 和 is_causal ?
Q2 )  query, key, value 应该怎么设置维度, 注意事项 ?

torch.nn.MultiheadAttention
默认设置下, 所有的K对Q都是可见的:
    attn_mask=None,  is_causal=False
如果改写为推理模式, attn_mask=*** 和is_causal=True 只能二选一指定，
情况一 :  is_causal=True
    RuntimeError: Need attn_mask if specifying the is_causal hint. You may use the Transformer module method `generate_square_subsequent_mask` to create this mask
情况二 : attn_mask=***
    执行无报错, 但是需要关注的是attn_mask设置的形状,
    一般输入query shape=(b,s,d)
    所以, attn_mask shape=(b*num_heads, s_q, s_k)

注意点：
    key, value 要求输入是(b,s,d) 维度相同
    若key, value 的 d 不相等，则内部还是会将其映射到相同的维度：
    def _in_projection():
        '''
            Output: in output triple :math:`(q', k', v')`,
                - q': :math:`[Qdims..., Eq]`
                - k': :math:`[Kdims..., Eq]`
                - v': :math:`[Vdims..., Eq]`
        '''
        Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
        assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
        assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
        assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
        assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
        assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
        assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
"""

b = 2
s = 3
d = 2 * 4
num_heads = 2

attn = torch.nn.MultiheadAttention(
    embed_dim=8, num_heads=num_heads, dropout=0, batch_first=True
)

x = torch.randn(b, s, d)
query = x[:, -1:, :]  # (1,1,8)
# key = x[..., : num_heads * 3] 这里d不一致会有问题
key = x  # (1,3,8)
value = x

print(query.shape, key.shape, value.shape)

attn_mask = (
    torch.triu(torch.ones(1, s), diagonal=1)
    .bool()
    .unsqueeze(0)
    .expand(b * num_heads, 1, s)
)
print(attn_mask, attn_mask.shape)

y, _ = attn(query=query, key=key, value=value, attn_mask=attn_mask)

print(x)
print(y)
