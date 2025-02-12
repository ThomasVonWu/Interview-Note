# 手撕绝对位置编码
"""

待实现
yarn rope (ref deepseekv3) / ntk rope (https://zhuanlan.zhihu.com/p/675243992)
平替版alibi (ref https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L1011)
"""
import math
import torch
import torch.nn as nn


# SinusoidalPE abspe
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super(AbsolutePositionalEncoding, self).__init__()

        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_seq_len, 1)
        i = torch.arange(0, dim, dtype=torch.float)  # (dim, )
        angle_rates = torch.pow(1 / 10000.0, (2 * (i // 2) / dim))  # (dim, )
        pos_embedding = pos * angle_rates  # (max_seq_len, dim)

        pos_embedding[:, 0::2] = torch.sin(pos_embedding[:, 0::2])
        pos_embedding[:, 1::2] = torch.cos(pos_embedding[:, 1::2])
        # print(pos_embedding)
        self.register_buffer(
            "pos_embedding", pos_embedding.unsqueeze(0)
        )  # (1, max_seq_len, dim)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
        Returns:
            输出张量，形状为 (batch_size, seq_len, dim)
        """
        return x + self.pos_embedding[:, : x.size(1)]


# SinusoidalPE Acclerate abspe
class AbsolutePositionalEncoding2(nn.Module):
    """
    a^b = exp(blog(a))
    10000^(-2i/d) = exp(-2i/d*log10000)
    充分利用硬件加速:
        在当代的处理器中，无论是 CPU 还是 DSP，基本都有针对以的指数函数和对数函数的硬件加速机制。
        要么是有数学协处理器进行加速，要么就是有专门的加速指令。但是，对于以不定常数为底数的指数
        与对数运算，可能就没有这样的加速机制了。因此，在当代处理器中，以e为底的指数与对数运算效
        率通常会比较高。
    提高计算的数值稳定性:
        在很多数值计算中，使用以e为底数的对数和指数的形式可以提供更好的数值稳定性，尤其是当指数的
        底数比较大时（例如 10000 这样的较大数）。直接计算大数的指数数值可能导致浮点数溢出或者精度
        损失，而用以e（数值是 2.71828）为底的对数和指数的组合则可以避免这些问题。
    """

    def __init__(self, dim: int, max_seq_len: int = 512):
        super(AbsolutePositionalEncoding2, self).__init__()

        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_seq_len, 1)
        i = torch.arange(0, dim, dtype=torch.float)  # (dim, )
        angle_rates = torch.exp(-math.log(10000.0) * (2 * (i // 2) / dim))
        pos_embedding = pos * angle_rates  # (max_seq_len, dim)

        pos_embedding[:, 0::2] = torch.sin(pos_embedding[:, 0::2])
        pos_embedding[:, 1::2] = torch.cos(pos_embedding[:, 1::2])
        self.register_buffer(
            "pos_embedding", pos_embedding.unsqueeze(0)
        )  # (1, max_seq_len, dim)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
        Returns:
            输出张量，形状为 (batch_size, seq_len, dim)
        """
        return x + self.pos_embedding[:, : x.size(1)]


def test_abspe_consistency():
    # 定义模型参数
    dim = 512
    max_seq_len = 100
    batch_size = 2
    seq_len = 50

    # 初始化位置编码层
    pos_encoder = AbsolutePositionalEncoding(dim, max_seq_len)
    pos_encoder2 = AbsolutePositionalEncoding2(dim, max_seq_len)

    # 输入张量
    x = torch.randn(batch_size, seq_len, dim)

    # 添加位置编码
    x_with_pos = pos_encoder(x)  # (2, 50, 512)
    x_with_pos2 = pos_encoder2(x)  # (2, 50, 512)

    err = (x_with_pos - x_with_pos2).abs().max()
    print(err)
    assert err < 1e-4


#  Learned parameters abspe
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=50000):
        super(LearnedPositionalEncoding, self).__init__()

        self.pe = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.normal_(self.pe, mean=0, std=0.02)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
        Returns:
            输出张量，形状为 (batch_size, seq_len, dim)
        """
        x = x + self.pe[:, x.size(1)]
        return x


class LLaMARoPE(nn.Module):
    """
    转换为复变函数张量计算：
    `\begin{bmatrix} q_0 + iq_1\\ q_2 + iq_3 \\ . \\ . \\ . \\ q_{d-2} + iq_{d-1}  \end{bmatrix} ⊗  \begin{bmatrix} cosm\theta_0 + isinm\theta_0\\ cosm\theta_1 + isinm\theta_1 \\ . \\ . \\ . \\ cosm\theta_{\frac{d}{2}-1} + isinm\theta_{\frac{d}{2}-1}  \end{bmatrix} = \begin{bmatrix} q_0cosm\theta_0 - q1sinm\theta_0 + i(q_1cosm\theta_0 +q_0sinm\theta_0) \\ q_2cosm\theta_1 - q_3sinm\theta_1 + i(q_3cosm\theta_1 + q_2sinm\theta_1) \\ . \\ . \\ . \\ q_{d-2}cosm\theta_{\frac{d}{2}-1} - q_{d-1}sinm\theta_{\frac{d}{2}-1} + i(q_{d-1}cosm\theta_{\frac{d}{2}-1} + q_{d-2}sinm\theta_{\frac{d}{2}-1})  \end{bmatrix}`
    qk              : (bs, seq_len, d//2)
    freqs_cis : (1, seq_len, d//2) 旋转量
    qk * freqs_cis 逐元素乘
    """

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

    def precompute_freqs_cis(self):
        pos = torch.arange(self.max_seq_len).float()
        # freqs = 1.0 / (
        #     self.base
        #     ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        # )
        freqs = torch.exp(
            torch.arange(0, self.dim, 2)[: (self.dim // 2)].float()
            * -(math.log(10000.0) / self.dim)
        )
        freqs = torch.outer(pos, freqs)  # (max_seq_len, d // 2)
        self.freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # (max_seq_len, d // 2) complex64

    # def reshape_for_broadcast(self, x: torch.FloatTensor):
    # """
    # freqs_cis 在bs轴增加一个维度，默认1。
    # return: (1, seq_len, d//2)
    # """
    # ndim = x.ndim
    # assert self.freqs_cis.shape == (x.shape[1], x.shape[-1]),  f"{self.freqs_cis.shape}, {x.shape[1]}, {x.shape[-1]}."
    # shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # self.freqs_cis = self.freqs_cis.view(*shape)

    def apply_rotary_emb(self, x: torch.FloatTensor):
        """
        x : q,k (batch_size, seq_len, d), d默认是偶数
        """
        x_ = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        )  # (bs, seq_len, d) => (bs, seq_len, d//2, 2) => 复数张量(bs, seq_len, d//2)
        # self.reshape_for_broadcast(x_)
        self.freqs_cis = self.freqs_cis.unsqueeze(0)
        x = torch.view_as_real(x_ * self.freqs_cis[:, : x.shape[1], :]).flatten(2)
        return x


class MyRoPE(nn.Module):
    """显式构建：
    `\begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ . \\ . \\ . \\ q_{d-2} \\ q_{d-1} \end{bmatrix} ⊗ \begin{bmatrix} cosm\theta_0 \\ cosm\theta_0 \\ cosm\theta_1\\ cosm\theta_1 \\ . \\ . \\ . \\ cosm\theta_{\frac{d}{2}-1} \\ cosm\theta_{\frac{d}{2}-1} \end{bmatrix}  + \begin{bmatrix} -q_1 \\ q_0 \\ -q_3 \\ q_2 \\ . \\ . \\ . \\ -q_{d-1} \\ q_{d-2} \end{bmatrix} ⊗ \begin{bmatrix} sinm\theta_0 \\ sinm\theta_0 \\ sinm\theta_1\\ sinm\theta_1 \\ . \\ . \\ . \\ sinm\theta_{\frac{d}{2}-1} \\ sinm\theta_{\frac{d}{2}-1} \end{bmatrix}`
    """

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

    def precompute_freqs_cis(self):
        pos = torch.arange(self.max_seq_len).float()
        # freqs = 1.0 / (
        #     self.base
        #     ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        # )
        freqs = torch.exp(
            torch.arange(0, self.dim, 2)[: (self.dim // 2)].float()
            * -(math.log(10000.0) / self.dim)
        )
        self.freqs_cis = torch.outer(pos, freqs)  # (max_seq_len, d // 2)

    def apply_rotary_emb(self, x: torch.FloatTensor):
        """
        x : q/k shape(b,s,d)
        """
        pos_emb = (
            torch.stack([torch.sin(self.freqs_cis), torch.cos(self.freqs_cis)], dim=-1)
            .float()
            .reshape(self.max_seq_len, -1)
        )  #  (max_seq_len, d)
        pos_emb = pos_emb.unsqueeze(0)  # (1, max_seq_len, d)

        # Extract and duplicate cosine and sine embeddings
        # [sin(mt0),cos(mt0),sin(mt1),cos(mt1),...] => [cos(mt0), cos(mt0), cos(mt1), cos(mt1), ...]
        cos_emb = pos_emb[:, : x.size(1), 1::2].repeat_interleave(
            2, dim=-1
        )  # (1, seq_len, d)
        sin_emb = pos_emb[:, : x.size(1), ::2].repeat_interleave(
            2, dim=-1
        )  # (1, seq_len, d)

        # [q0, q1, q2, q3, ..., q_d-2, q_d-1] => [-q1, q0, -q3, q2, ..., -q_d-1, q_d-2]
        x_ = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape(
            x.size()
        )  # (bs, seq_len, d)

        x = x * cos_emb + x_ * sin_emb
        return x


def test_rope_consistency():
    # 定义模型参数
    dim = 512
    max_seq_len = 100
    batch_size = 2
    seq_len = 50

    # 初始化位置编码层
    pos_encoder = LLaMARoPE(dim, max_seq_len)
    pos_encoder2 = MyRoPE(dim, max_seq_len)

    pos_encoder.precompute_freqs_cis()
    pos_encoder2.precompute_freqs_cis()

    # 输入张量
    x = torch.randn(batch_size, seq_len, dim)

    # 添加位置编码
    x_with_pos = pos_encoder.apply_rotary_emb(x)  # (2, 50, 512)
    x_with_pos2 = pos_encoder2.apply_rotary_emb(x)  # (2, 50, 512)
    err = (x_with_pos - x_with_pos2).abs().max()
    print(err)


class ALiBiPE(nn.Module):
    """复现源代码中ALiBi相对位置编码
    这里假设了QK的seq_len 长度一致
    """

    def __init__(self, n_heads: int, max_seq_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.get_slopes()
        self.get_biases()

    def get_slopes(
        self,
    ) -> torch.Tensor:
        """
        生成每个注意力头的斜率参数 m
        Return:
             (n_heads, )
        """
        self.slopes = 2 ** (torch.arange(1, self.n_heads + 1) * (-8 / self.n_heads))
        print(self.slopes)

    def get_biases(
        self,
    ) -> torch.Tensor:
        """
        动态生成偏置矩阵:
        Return:
            self.bias shape=(1, n_heads, max_seq_len, max_seq_len)
            self.pos_diff shape=(max_seq_len, max_seq_len) 构造如下：
            [ 0,  0,  0,  0,  0]
            [-1,  0,  0,  0,  0]
            [-2, -1,  0,  0,  0]
            [-3, -2, -1,  0,  0]
            [-4, -3, -2, -1,  0]
        """
        pos = torch.arange(self.max_seq_len).float()
        self.pos_diff = torch.arange(self.max_seq_len).view(-1, 1) - torch.arange(
            self.max_seq_len
        ).view(1, -1)
        self.pos_diff = -torch.tril(self.pos_diff)  # (max_seq_len, max_seq_len)
        self.bias = self.slopes.unsqueeze(-1).unsqueeze(-1) * self.pos_diff.unsqueeze(0)
        self.bias = self.bias.unsqueeze(0)

    def apply_alibi_attention(
        self, attn_scores: torch.Tensor, mask_value=1e-6
    ) -> torch.Tensor:
        """
        Apply alibi position encoding mask to attention scores.

        Args:
            attn_scores (torch.Tensor): Attention scores without softmax of shape (batch_size, n_heads, q_seq_len, k_seq_len).

        Returns:
            torch.Tensor: The attention scores with alibi mask applied.
        """
        q_seq_len = attn_scores.shape[-2]
        k_seq_len = attn_scores.shape[-1]

        # Apply a mask based on position difference, setting elements below the diagonal to a large negative number (e.g., -1e4)

        causal_mask = torch.triu(
            torch.ones([self.max_seq_len, self.max_seq_len]), diagonal=1
        )
        causal_mask = causal_mask.masked_fill(
            causal_mask == 1, float("-inf")
        )  # (max_seq_len, max_seq_len)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, max_seq_len, max_seq_len)

        # alibi_mask needs to be broadcasted to match the attention scores shape (1, 1, seq_len, seq_len)
        print(self.bias)
        alibi_mask = causal_mask + self.bias  # (1, n_heads,  max_seq_len, max_seq_len)

        # Apply the mask to the attention scores
        attn_scores = attn_scores + alibi_mask[:, :, :q_seq_len, :k_seq_len]

        return attn_scores


def test_ablibipe():
    max_seq_len = 5
    seq_len = 3
    n_heads = 8
    atten_scores = torch.randn(1, n_heads, seq_len, seq_len)
    ablibipe = ALiBiPE(n_heads, max_seq_len)
    dis = ablibipe.apply_alibi_attention(atten_scores)
    print(dis)  # (1, n_heads, seq_len, seq_len, n_heads)


if __name__ == "__main__":
    test_abspe_consistency()  # tensor(1.9073e-06)
    test_rope_consistency()  # tensor(4.7684e-07)
    test_ablibipe()
