import torch
from dca_gpt import DCAGPT


# 初始化 DCAGPT 模型
gpt = DCAGPT(
    num_tokens = 256,    # 词汇表大小为256
    dim = 512,           # 模型维度为512
    depth = 6,           # Transformer层数为6
    heads = 8,           # 多头注意力的头数为8
    dim_head = 64,       # 每个注意力头的维度为64
    past_layers_k = 2    # 使用过去2层的键和值进行注意力计算
)


# 生成随机输入ID张量，形状为 (2, 4096)
ids = torch.randint(0, 256, (2, 4096))


# 前向传播，获取logits，形状为 (2, 4096, 256)
logits = gpt(ids) # (2, 4096, 256)
