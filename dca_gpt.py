import torch
from torch import nn, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm
import einx
from einops import rearrange, einsum
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding


# ein notation

# b - batch
# n -sequence
# h - heads
# l - logits
# o - number of grn outputs
# y - laYer


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
    与之前的实现不同，此实现将查询（Q）、键（K）和值（V）的计算分开处理，并接受单独的输入。

    这个类实现了多头自注意力机制。
    1. **RMS归一化**: 对查询（Q）、键（K）和值（V）的输入进行RMS归一化。
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

        # 定义查询（Q）、键（K）和值（V）的计算，使用RMS归一化和线性变换
        self.to_q = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim_inner, bias = False))
        self.to_k = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim_inner, bias = False))
        self.to_v = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim_inner, bias = False))

        # 使用Einops的Rearrange操作重排张量形状
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # 定义输出线性变换层，将内部维度转换回输入维度
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        q_input,
        k_input,
        v_input
    ):
        """
        前向传播方法，执行自注意力机制的计算。

        Args:
            q_input (torch.Tensor): 查询（Q）的输入张量，形状为 (batch_size, sequence_length, dim)。
            k_input (torch.Tensor): 键（K）的输入张量，形状为 (batch_size, sequence_length, dim)。
            v_input (torch.Tensor): 值（V）的输入张量，形状为 (batch_size, sequence_length, dim)。

        Returns:
            torch.Tensor: 经过自注意力机制处理后的输出张量，形状为 (batch_size, sequence_length, dim)。
        """
        # 计算查询（Q）向量
        q = self.to_q(q_input)
        # 计算键（K）向量
        k = self.to_k(k_input)
        # 计算值（V）向量
        v = self.to_v(v_input)

        # 重排张量形状以适应多头注意力
        q, k, v = map(self.split_heads, (q, k, v))

        # 应用旋转位置编码
        q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)

        # 执行缩放点积注意力计算
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True  # 是否为因果注意力（自回归模型）
        )

        # 重排张量形状以恢复原始形状
        out = self.merge_heads(out)

        # 应用输出线性变换
        return self.to_out(out)


def FeedForward(dim, expansion_factor = 4.):
    """
    构建前馈神经网络（Feed-Forward Network, FFN）层。

    该函数创建一个前馈神经网络层，通常用于Transformer模型中。
    它由线性变换、GELU激活函数和另一个线性变换组成。

    Args:
        dim (int): 输入特征的维度大小。
        expansion_factor (float, optional): 扩展因子，用于计算隐藏层的维度，默认为4。

    Returns:
        nn.Sequential: 包含前馈神经网络层的前馈神经网络。
    """
    # 计算隐藏层的维度
    dim_hidden = int(dim * expansion_factor)

    return nn.Sequential(
        Linear(dim, dim_hidden),  # 第一个线性变换，将维度从 `dim` 扩展到 `dim_hidden`
        nn.GELU(),  # 应用GELU激活函数
        Linear(dim_hidden, dim)  # 第二个线性变换，将维度从 `dim_hidden` 恢复回 `dim`
    )


class GRN(Module):
    """
    GRN（Generalized Residual Networks）类。

    该类实现了一个通用的残差网络结构，用于处理输入特征并进行多层次的聚合和变换。

    参数:
        dim (int): 输入特征的维度大小。
        num_layers (int): 网络的层数。
        num_outputs (int, optional): 输出的数量，默认为1。
    """
    def __init__(
        self,
        dim,
        num_layers,
        num_outputs = 1
    ):
        super().__init__()

        # 记录输出的数量
        self.num_outputs = num_outputs
        # 记录网络的层数
        self.num_layers = num_layers

        # 定义一个顺序模块，用于对输入进行归一化和线性变换
        self.to_aggregate = nn.Sequential(
            RMSNorm(dim),  # 对输入特征进行RMS归一化
            Linear(dim, num_outputs, bias = False),  # 将维度从dim映射到num_outputs，不使用偏置
            Rearrange('... outputs -> outputs ...')  # 重排张量的维度，将'outputs'维度移到最前面
        )

        # 定义一个可学习的偏置参数，形状为 (num_outputs, num_layers)
        self.bias = nn.Parameter(torch.zeros(num_outputs, num_layers))

        # 初始化线性层的权重为全零
        nn.init.zeros_(self.to_aggregate[-2].weight)

        # 对偏置参数进行初始化，将最后一列设置为1，其余为0
        with torch.no_grad():
            self.bias[:, -1] = 1.

    def forward(
        self,
        tokens_across_depth # Float['y b n d']
    ):
        assert self.num_layers == tokens_across_depth.shape[0]

        # 对输入进行聚合变换，输出形状为 (num_outputs, 批量大小, 序列长度, 特征维度)
        aggregate = self.to_aggregate(tokens_across_depth)

        # 将聚合结果与偏置参数进行逐元素相加，形状为 (num_outputs, 批量大小, 序列长度, 特征维度)
        # 然后应用ReLU激活函数
        aggregate = einx.add('o y ..., o y -> o y ...', aggregate, self.bias).relu()

        # 使用爱因斯坦求和约定进行张量操作：
        # 将tokens_across_depth与aggregate进行相乘，输出形状为 (num_outputs, 批量大小, 序列长度, 特征维度)
        output = einsum(tokens_across_depth, aggregate, 'y b n d, o y b n -> o b n d')

        # 如果输出数量为1，则将输出张量的维度进行重排，去掉第一个维度
        if self.num_outputs == 1:
            output = rearrange(output, '1 ... -> ...')

        return output


class DCABlock(Module):
    """
    DCABlock类，实现了一个深度交叉注意力块（Deep Cross Attention Block）。

    该块结合了GRN（通用残差网络）、注意力机制和前馈神经网络，用于处理输入特征。

    参数:
        dim (int): 输入特征的维度大小。
        grn_num_layers (int): GRN模块的层数。
        dim_head (int, optional): 注意力机制中每个头的维度，默认为64。
        heads (int, optional): 注意力机制中的头数，默认为8。
        ff_expansion_factor (float, optional): 前馈神经网络中隐藏层扩展因子，默认为4.0。
    """
    def __init__(
        self,
        dim,
        *,
        grn_num_layers,
        dim_head = 64,
        heads = 8,
        ff_expansion_factor = 4.
    ):
        super().__init__()

        # 定义GRN模块，用于处理输入特征的查询（q）、键（k）和值（v）
        self.qkv_grn = GRN(dim, num_layers = grn_num_layers, num_outputs = 3)

        # 定义注意力机制模块
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # 定义前馈神经网络前的归一化层
        self.pre_ff_norm = RMSNorm(dim)

        # 定义前馈神经网络
        self.ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)

    def forward(
        self,
        tokens_across_depth # Float['depth b n d']
    ):
        # 应用GRN，获取查询（Q）、键（K）和值（V）
        q_input, k_input, v_input = self.qkv_grn(tokens_across_depth)

        # 保留残差连接的输入
        residual = q_input

        # 应用自注意力机制
        attn_out = self.attn(q_input, k_input, v_input)

        # 应用前馈神经网络前的归一化，并添加残差连接
        ff_input = self.pre_ff_norm(attn_out + residual)

        # 应用前馈神经网络
        ff_out = self.ff(ff_input)

        # 返回输出，并添加自注意力机制的输出
        return ff_out + attn_out


class DCAGPT(Module):
    """
    DCAGPT 类，实现了一个基于深度交叉注意力机制的 Transformer 模型，用于处理序列数据。

    该模型结合了词嵌入、多个 DCA 块、归一化层和输出层，能够捕捉序列中的复杂依赖关系。

    参数:
        num_tokens (int): 词汇表大小，即模型能够处理的唯一标记的数量。
        dim (int): 模型的维度，即每个标记的嵌入向量长度。
        depth (int): DCA 块的层数，即模型的深度。
        past_layers_k (int, optional): 用于选择过去层的超参数 k，默认为 2。
        dim_head (int, optional): 每个注意力头的维度，默认为 64。
        heads (int, optional): 注意力头的数量，默认为 8。
        ff_expansion_factor (float, optional): 前馈神经网络中隐藏层的扩展因子，默认为 4.0。
    """
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        past_layers_k = 2,
        dim_head = 64,
        heads = 8,
        ff_expansion_factor = 4.
    ):
        super().__init__()
        # 定义词嵌入层，将输入的标记索引转换为维度为 dim 的嵌入向量
        self.token_emb = nn.Embedding(num_tokens, dim)

        # 定义超参数 k，用于选择过去层的子集以提高效率
        self.past_layers_k = past_layers_k

        # 初始化 DCA 块列表
        dca_blocks = []
        for i in range(depth):
            # 初始化 DCA 块，传入必要的参数
            # grn_num_layers 根据当前层数动态调整，确保不超过 past_layers_k * 2
            dca = DCABlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_expansion_factor = ff_expansion_factor,
                grn_num_layers = min(past_layers_k * 2, i + 1)
            )

            dca_blocks.append(dca)
        # 使用 ModuleList 包装 DCA 块列表，以便在 forward 中迭代调用
        self.dca_blocks = ModuleList(dca_blocks)

        # 初始化最终的全局归一化层（GRN），层数设为 depth + 1
        self.final_grn = GRN(dim, num_layers = depth + 1)

        # 初始化 RMS 归一化层，用于对输出进行归一化处理
        self.norm = RMSNorm(dim)
        # 线性输出层，将归一化后的向量映射回词汇表大小，用于预测下一个标记
        self.to_logits = Linear(dim, num_tokens, bias = False)
 
    def forward(
        self,
        ids,
        return_loss = False
    ):
        """
        前向传播方法，处理输入标记并生成输出 logits 或损失值。

        参数:
            ids (Tensor): 输入的标记索引张量，形状为 (batch_size, sequence_length)。
            return_loss (bool): 是否返回损失值。

        返回:
            Tensor: 如果 return_loss 为 False，返回 logits，形状为 (batch_size, sequence_length, num_tokens)。
                    如果 return_loss 为 True，返回损失值。
        """
        # 获取超参数 k 的值
        k = self.past_layers_k # k in paper

        if return_loss:
            # 如果需要计算损失，则将输入 ids 分割为输入和目标
            # 前面的所有标记作为输入，最后一个标记作为目标
            ids, labels = ids[:, :-1], ids[:, 1:]

        # 将输入的标记索引转换为嵌入向量，形状为 (batch_size, sequence_length, dim)
        tokens = self.token_emb(ids)

        # 初始化一个列表，用于存储每一层的输出
        all_tokens = [tokens]

        # 遍历所有 DCA 块
        for dca_block in self.dca_blocks:
            # 将当前所有层的输出堆叠起来，形状为 (num_layers, batch_size, sequence_length, dim)
            all_tokens_stacked = stack(all_tokens)
            num_layers = all_tokens_stacked.shape[0]

            # 根据当前层数确定需要包含的过去层
            if num_layers < (k * 2):
                # 如果总层数少于 2k，则将所有层的输出作为 DCA 块的输入
                dca_block_input = all_tokens_stacked
            else:
                # 否则，选择前 k 层和后 k 层的输出作为 DCA 块的输入
                dca_block_input = cat((
                    all_tokens_stacked[:k], # first k layers
                    all_tokens_stacked[-k:] # last k layers
                ))

            # 将选择的层输入 DCA 块，得到输出，形状为 (batch_size, sequence_length, dim)
            dca_out = dca_block(dca_block_input)

            # 将 DCA 块的输出添加到列表中，供下一层使用
            all_tokens.append(dca_out)

        # 将所有层的输出堆叠起来，形状为 (num_layers + 1, batch_size, sequence_length, dim)
        pooled_tokens = self.final_grn(stack(all_tokens))

        # 对堆叠后的输出进行归一化处理，形状保持不变
        embed = self.norm(pooled_tokens)

        # 通过线性层将归一化后的向量映射到词汇表大小，得到 logits，形状为 (batch_size, sequence_length, num_tokens)
        logits = self.to_logits(embed)

        if not return_loss:
            # 如果不需要返回损失，则返回 logits
            return logits

        # 如果需要返回损失，则计算交叉熵损失
        # 首先将 logits 的形状调整为 (batch_size, sequence_length, num_tokens)
        # 然后计算损失
        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
