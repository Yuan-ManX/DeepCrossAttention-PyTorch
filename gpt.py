from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm
from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding


def exists(v):
    """
    检查一个值是否存在（即不为None）。

    Args:
        v: 任意类型的值。

    Returns:
        bool: 如果值不为None，则返回True；否则返回False。
    """
    return v is not None


def default(v, d):
    """
    如果值存在（即不为None），则返回该值；否则返回默认值。

    Args:
        v: 任意类型的值。
        d: 默认值。

    Returns:
        任意类型: 如果v不为None，则返回v；否则返回d。
    """
    return v if exists(v) else d


class Attention(Module):
    """
    自注意力机制（Attention）类。

    该类实现了多头自注意力机制，用于处理输入特征。
    它结合了RMS归一化、旋转位置编码、多头注意力计算和输出线性变换。

    这个类实现了多头自注意力机制。
    1. **RMS归一化**: 对输入进行RMS归一化。
    2. **线性变换**: 将输入转换为查询（Q）、键（K）和值（V）。
    3. **重排张量**: 重排张量形状以适应多头注意力。
    4. **旋转位置编码**: 应用旋转位置编码。
    5. **缩放点积注意力**: 计算缩放点积注意力。
    6. **重排张量**: 重排张量形状以恢复原始形状。
    7. **输出线性变换**: 应用输出线性变换，得到最终输出。

    Args:
        dim (int): 输入特征的维度大小。
        dim_head (int, optional): 每个注意力头的维度大小，默认为64。
        heads (int, optional): 多头注意力的头数，默认为8。
    """
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        # 定义RMS归一化层
        self.norm = RMSNorm(dim)
        
        # 多头注意力的头数
        self.heads = heads
        # 计算内部维度
        dim_inner = heads * dim_head

        # 定义旋转位置编码层
        self.rotary_embed = RotaryEmbedding(dim_head)

        # 定义线性变换层，将输入维度转换为内部维度
        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_k = nn.Linear(dim, dim_inner, bias = False)
        self.to_v = nn.Linear(dim, dim_inner, bias = False)

        # 使用Einops的Rearrange操作重排张量形状
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # 定义输出线性变换层，将内部维度转换回输入维度
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x
    ):
        """
        前向传播方法，执行自注意力机制的计算。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。

        Returns:
            torch.Tensor: 经过自注意力机制处理后的输出张量，形状为 (batch_size, sequence_length, dim)。
        """
        # 应用RMS归一化
        x = self.norm(x)

        # 计算查询（Q）向量
        q = self.to_q(x)
        # 计算键（K）向量
        k = self.to_k(x)
        # 计算值（V）向量
        v = self.to_v(x)

        # 重排张量形状以适应多头注意力
        q, k, v = map(self.split_heads, (q, k, v))

        # 应用旋转位置编码
        q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)

        # 执行缩放点积注意力计算
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True   # 是否为因果注意力（自回归模型）
        )

        # 重排张量形状以恢复原始形状
        out = self.merge_heads(out)

        # 应用输出线性变换
        return self.to_out(out)


def FeedForward(dim, expansion_factor = 4.):
    """
    构建前馈神经网络（Feed-Forward Network, FFN）层。

    该函数创建一个前馈神经网络层，通常用于Transformer模型中。
    它由RMS归一化、线性变换、GELU激活函数和另一个线性变换组成。

    Args:
        dim (int): 输入特征的维度大小。
        expansion_factor (float, optional): 扩展因子，用于计算隐藏层的维度，默认为4。

    Returns:
        nn.Sequential: 包含前馈神经网络层的前馈神经网络。
    """
    # 计算隐藏层的维度
    dim_hidden = int(dim * expansion_factor)

    return nn.Sequential(
        RMSNorm(dim),  # 应用RMS归一化
        Linear(dim, dim_hidden),  # 第一个线性变换，将维度从 `dim` 扩展到 `dim_hidden`
        nn.GELU(),  # 应用GELU激活函数
        Linear(dim_hidden, dim)   # 第二个线性变换，将维度从 `dim_hidden` 恢复回 `dim`
    )


class GPT(Module):
    """
    GPT（Generative Pre-trained Transformer）类。

    该类实现了GPT模型，用于生成任务。
    它结合了嵌入层、多层自注意力机制、前馈神经网络层、归一化层和输出线性变换。

    这个类实现了GPT模型，用于生成任务。
    1. **嵌入层**: 将词汇索引转换为嵌入向量。
    2. **多层自注意力和前馈神经网络**: 多层自注意力和前馈神经网络层交替堆叠。
    3. **归一化**: 对输出进行归一化。
    4. **输出线性变换**: 将归一化后的输出转换为logits。
    5. **损失计算**: 如果需要返回损失，则计算交叉熵损失。

    Args:
        num_tokens (int): 词汇表的大小。
        dim (int): 模型的维度大小。
        depth (int): Transformer的层数。
        dim_head (int, optional): 每个注意力头的维度大小，默认为64。
        heads (int, optional): 多头注意力的头数，默认为8。
        ff_expansion_factor (float, optional): 前馈神经网络中扩展因子，默认为4。
    """
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_expansion_factor = 4.
    ):
        super().__init__()
        # 定义嵌入层，将词汇索引转换为嵌入向量
        self.token_emb = nn.Embedding(num_tokens, dim)

        # 初始化层列表
        layers = []
        for _ in range(depth):
            # 定义自注意力层
            attn = Attention(dim = dim, dim_head = dim_head, heads = heads)
            # 定义前馈神经网络层
            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)

            # 将自注意力和前馈神经网络层添加到层列表中
            layers.append(ModuleList([attn, ff]))

        # 将层列表转换为 ModuleList
        self.layers = ModuleList(layers)

        # 定义归一化层
        self.norm = RMSNorm(dim)
        # 定义输出线性变换层，将维度从 `dim` 转换为 `num_tokens`
        self.to_logits = Linear(dim, num_tokens, bias = False)
 
    def forward(
        self,
        ids,
        return_loss = False
    ):
        """
        前向传播方法，执行GPT模型的计算。

        Args:
            ids (torch.Tensor): 输入token的ID张量，形状为 (batch_size, sequence_length)。
            return_loss (bool, optional): 是否返回损失，默认为False。

        Returns:
            torch.Tensor or tuple: 如果 `return_loss` 为False，则返回logits；否则，返回交叉熵损失。
        """
        if return_loss:
            # 分离输入和标签
            ids, labels = ids[:, :-1], ids[:, 1:]

        # 将token ID转换为嵌入向量
        tokens = self.token_emb(ids)

        for attn, ff in self.layers:
            # 应用自注意力层并添加残差连接
            tokens = attn(tokens) + tokens
            # 应用前馈神经网络层并添加残差连接
            tokens = ff(tokens) + tokens

        # 应用归一化
        embed = self.norm(tokens)

        # 应用输出线性变换，得到logits
        logits = self.to_logits(embed)

        if not return_loss:
            # 如果不需要返回损失，则返回logits
            return logits

        # 如果需要返回损失，则计算交叉熵损失
        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
