import math
import gzip
import random
import tqdm
import numpy as np
import torch
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from gpt import GPT
from dca_gpt import DCAGPT


# 设置训练和生成相关的超参数
NUM_BATCHES = int(1e5)            # 总批次数，设置为100,000
BATCH_SIZE = 4                    # 每个批次的样本数量，设置为4
GRAD_ACCUM_EVERY = 4              # 梯度累积的步数，设置为4，即每4个批次更新一次梯度
LEARNING_RATE = 1e-4              # 学习率，设置为0.0001
VALIDATE_EVERY = 100              # 每100个批次进行一次验证
PRIME_LENGTH = 128                # 预热序列长度，设置为128，用于模型生成时的初始输入
GENERATE_EVERY = 500              # 每500个批次进行一次生成
GENERATE_LENGTH = 512             # 每次生成的序列长度，设置为512
SEQ_LEN = 512                     # 输入序列的最大长度，设置为512

USE_DCA = True                    # 是否使用深度交叉注意力（DCA）模块，设置为True


def exists(v):
    """
    检查变量是否存在（即不为None）。

    参数:
        v: 任意变量

    返回:
        bool: 如果v不为None，返回True；否则返回False。
    """
    return v is not None


def cycle(loader):
    """
    创建一个无限循环的数据加载器。

    该函数接收一个数据加载器，并返回一个生成器，可以无限次地遍历数据加载器中的数据。
    这在训练过程中非常有用，可以反复使用数据加载器而无需手动重置。

    参数:
        loader (DataLoader): PyTorch的数据加载器

    返回:
        generator: 一个无限循环的数据生成器
    """
    while True:
        for data in loader:
            yield data


def decode_token(token):
    """
    将单个token解码为对应的字符。

    该函数将输入的token转换为对应的ASCII字符。为了确保字符可打印，token值被限制在32及以上。

    参数:
        token (int): 单个token的整数值

    返回:
        str: 对应的ASCII字符
    """
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    """
    将一系列tokens解码为对应的字符串。

    该函数将输入的tokens张量中的每个token转换为字符，并将它们连接成一个字符串。

    参数:
        tokens (Tensor): 一系列token的整数值，形状为 (sequence_length,)

    返回:
        str: 解码后的字符串
    """
    return "".join(list(map(decode_token, tokens)))


def log(t, eps = 1e-20):
    """
    计算输入张量的自然对数，并避免数值下溢。

    为了防止对数函数接收零或负数输入，使用一个非常小的值eps进行裁剪。

    参数:
        t (Tensor): 输入张量
        eps (float, optional): 最小值，默认为1e-20

    返回:
        Tensor: 输入张量的自然对数，形状与t相同
    """
    return torch.log(t.clamp(min = eps))


def gumbel_noise(t):
    """
    生成与输入张量形状相同的Gumbel噪声。

    Gumbel噪声常用于实现从离散分布中进行重参数化采样。

    参数:
        t (Tensor): 输入张量

    返回:
        Tensor: 与t形状相同的Gumbel噪声
    """
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    """
    使用Gumbel-Softmax技巧对输入张量进行采样。

    该函数通过对输入张量添加Gumbel噪声并进行softmax操作，实现从多项式分布中进行重参数化采样。
    temperature参数控制采样的平滑程度，temperature越低，采样结果越接近于argmax。

    参数:
        t (Tensor): 输入张量，通常是未归一化的logits
        temperature (float, optional): 温度参数，默认为1.
        dim (int, optional): 进行softmax操作的维度，默认为-1
        keepdim (bool, optional): 是否保留维度，默认为True

    返回:
        Tensor: 采样结果，形状与t相同
    """
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)


def top_k(logits, thres = 0.9):
    """
    对logits应用top-k过滤，保留前k个最大的logits，其余设置为负无穷大。

    该函数用于在生成过程中进行top-k采样，以防止生成低概率的token。

    参数:
        logits (Tensor): 未归一化的logits，形状为 (batch_size, vocab_size)
        thres (float, optional): 保留的比例，默认为0.9

    返回:
        Tensor: 过滤后的logits，形状与输入相同
    """
    # 计算需要保留的top-k数量
    k = math.ceil((1 - thres) * logits.shape[-1])
    # 获取前k个最大的值和对应的索引
    val, ind = torch.topk(logits, k)
    # 创建一个与logits形状相同的张量，填充负无穷大
    probs = torch.full_like(logits, float('-inf'))
    # 将前k个值填充到对应的位置
    probs.scatter_(-1, ind, val)
    return probs


def base_decoding(
    net,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    """
    使用基础解码方法从神经网络生成序列。

    该函数接收一个提示序列和生成参数，使用神经网络模型逐步生成后续的token。

    参数:
        net (Module): 神经网络模型，用于生成logits
        prompt (Tensor): 输入的提示序列，形状为 (batch_size, prompt_length)
        seq_len (int): 要生成的序列总长度
        temperature (float, optional): 温度参数，默认为1.
        filter_thres (float, optional): top-k过滤的阈值，默认为0.9

    返回:
        Tensor: 生成的结果序列，形状为 (batch_size, generate_length)
    """
    # 获取提示序列的长度，并复制提示序列
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    # 计算需要生成的token数量
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in range(sample_num_times):
        # 将当前输出输入模型，得到logits
        logits = net(out)
        # 选择最后一个时间步的logits
        logits = logits[:, -1]
        # 应用top-k过滤
        logits = top_k(logits, thres = filter_thres)
        # 使用Gumbel采样生成下一个token
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        # 将生成的token连接到当前输出上
        out = torch.cat((out, sample), dim = -1)

    # 返回生成的序列，去除提示部分
    return out[..., prompt_seq_len:]


# 根据 USE_DCA 的值选择模型
if USE_DCA:
    # 使用 DCAGPT 模型，传入以下参数：
    # num_tokens: 词汇表大小，设置为256
    # dim: 模型维度，设置为512
    # depth: DCA 块的层数，设置为6
    # past_layers_k: 过去层数 k，设置为2
    model = DCAGPT(
        num_tokens = 256,
        dim = 512,
        depth = 6,
        past_layers_k = 2  # 论文中的 `k` 值，表示模型在处理时考虑的过去层数    
    )
else:
    # 使用标准的 GPT 模型，传入以下参数：
    # num_tokens: 词汇表大小，设置为256
    # dim: 模型维度，设置为512
    # depth: 模型的层数，设置为6
    model = GPT(
        num_tokens = 256,
        dim = 512,
        depth = 6
    )

# 将模型移动到 GPU 上进行加速计算
model = model.cuda()


# 打开包含训练数据的压缩文件 './dataset/enwik8.gz'，并读取前95,000,000字节的数据
# enwik8 是一个常用的语言建模数据集，包含维基百科的部分内容
with gzip.open('./dataset/enwik8.gz') as file:
    # 读取数据并转换为 NumPy 数组
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    # 将数据分割为训练集和验证集，90,000,000 字节用于训练，剩余的用于验证
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)


# 定义一个自定义的数据集类，用于文本采样
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        """
        初始化 TextSamplerDataset。

        参数:
            data (Tensor): 文本数据张量。
            seq_len (int): 每个样本的序列长度。
        """
        super().__init__()
        # 存储文本数据
        self.data = data
        # 存储序列长度
        self.seq_len = seq_len

    def __len__(self):
        """
        返回数据集的大小，即样本数量。

        返回:
            int: 样本数量。
        """
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        """
        根据索引获取一个样本。

        参数:
            index (int): 样本索引。

        返回:
            Tensor: 包含一个完整序列的 Tensor，形状为 (seq_len + 1,)。
        """
        # 随机选择一个起始位置，确保序列不会超出数据范围
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        # 获取一个完整的序列，包括目标标签
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        # 将数据移动到 GPU
        return full_seq.cuda()


# 创建训练和验证数据集实例
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)

# 创建 DataLoader，用于批量加载数据
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# 定义优化器，使用 Adam 优化器，传入模型参数和学习率
optim = Adam(model.parameters(), lr = LEARNING_RATE)

# 将训练和验证的 DataLoader 转换为无限循环的生成器
train_loader = cycle(train_loader)
val_loader = cycle(val_loader)


# 训练循环，迭代 NUM_BATCHES 次
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):

    # 设置模型为训练模式
    model.train()
    
    # 梯度累积，每次迭代中累积 GRAD_ACCUM_EVERY 次的梯度
    for _ in range(GRAD_ACCUM_EVERY):
        # 获取一个批次的数据
        data = next(train_loader)
        # 前向传播，计算损失
        loss = model(data, return_loss = True)
        # 反向传播，累积梯度
        (loss / GRAD_ACCUM_EVERY).backward()
    # 打印当前批次的训练损失
    print(f"training loss: {loss.item():.3f}")

    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    # 更新模型参数
    optim.step()
    # 清空梯度
    optim.zero_grad()

    # 每 VALIDATE_EVERY 次迭代进行一次验证
    if i % VALIDATE_EVERY == 0:
        # 设置模型为评估模式
        model.eval()
        # 关闭梯度计算，节省内存
        with torch.no_grad():
            # 获取一个验证批次的数据
            valid_data = next(val_loader)

            # 计算验证损失
            loss = model(valid_data, return_loss = True)
            print(f"validation loss: {loss.item():.3f}")

    # 每 GENERATE_EVERY 次迭代进行一次生成
    if i % GENERATE_EVERY == 0:
        # 设置模型为评估模式
        model.eval()

        # 从验证数据集中随机选择一个样本作为提示
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.cuda()

        # 解码提示序列为字符串
        prime = decode_tokens(inp)
        print(f"\n{prime}\n")

        # 添加批次维度，形状为 (1, PRIME_LENGTH)
        prompt = inp[None, ...]

        # 使用基础解码方法生成新的序列
        sampled = base_decoding(model, prompt, GENERATE_LENGTH)

        # 解码生成序列为字符串
        base_decode_output = decode_tokens(sampled[0])

        print(f"\n{base_decode_output}\n")
