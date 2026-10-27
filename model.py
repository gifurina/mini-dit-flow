"""条件 DiT 的模型组件，按练习顺序逐步实现。"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """将连续图像划分为不重叠的 patch，并投影成 Transformer token。

    默认输入：[B, 1, 28, 28]，默认输出：[B, 49, 128]。
    每个 4×4 灰度 patch 含 16 个像素，使用同一个可学习投影映射到 128 维。
    输出顺序按 patch 的行、列排列：从左到右，再从上到下。

    参数：
        in_channels: 图像通道数，Fashion-MNIST 为 1。
        patch_size: 正方形 patch 的边长，必须能整除 img_size。
        emb_size: 每个 token 的特征维度。
        img_size: 输入正方形图像的边长像素数，Fashion-MNIST 为 28。
    """

    def __init__(self, in_channels=1, patch_size=4, emb_size=128, img_size=28):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size  # 默认每行、每列各有 7 个 patch。
        self.num_patches = self.grid_size ** 2  # 默认共 49 个 token。
        # 相当于有128个卷积核，每个卷积核大小为4x4，步幅为4，输入通道数为1，输出通道数为128，[B, 1, 28, 28] -> [B, 128, 7, 7]
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入浮点图像张量，返回 [batch_size, num_patches, emb_size]。"""
        x = self.proj(x)  # [B, 1, 28, 28] -> [B, 128, 7, 7]
        x = x.flatten(2)  # 合并两个空间维度：[B, 128, 49]
        x = x.transpose(1, 2)  # 将特征维度放到最后：[B, 49, 128]
        return x


class PositionalEncoding2D(nn.Module):
    """
    给图像 token 加上固定的二维正余弦位置编码。
    默认输入：[B, 49, 128]，默认输出：[B, 49, 128]。
    """

    def __init__(self, emb_size=128, grid_size=7):
        super().__init__()
        # indexing="ij" 使第一个网格表示行、第二个网格表示列。
        # y的每一列是对第一个张量的复制，x的每一行是对第二个张量的复制
        # y = [[0, 0, 0, 0, 0, 0, 0],
        #      [1, 1, 1, 1, 1, 1, 1],
        #      [2, 2, 2, 2, 2, 2, 2],
        #      [3, 3, 3, 3, 3, 3, 3],
        #      [4, 4, 4, 4, 4, 4, 4],
        #      [5, 5, 5, 5, 5, 5, 5],
        #      [6, 6, 6, 6, 6, 6, 6]]
        # x = [[0, 1, 2, 3, 4, 5, 6],
        #      [0, 1, 2, 3, 4, 5, 6],
        #      [0, 1, 2, 3, 4, 5, 6],
        #      [0, 1, 2, 3, 4, 5, 6],
        #      [0, 1, 2, 3, 4, 5, 6],
        #      [0, 1, 2, 3, 4, 5, 6],
        #      [0, 1, 2, 3, 4, 5, 6]]
        y, x = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),# [0, 1, 2, 3, 4, 5, 6]
            torch.arange(grid_size, dtype=torch.float32),# [0, 1, 2, 3, 4, 5, 6]
            indexing="ij",
        )

        # 每个坐标使用 emb_size/4 = 32 个频率，sin、cos 拼接后占 emb_size/2 维。
        # 频率 omega_i = 1 / 10000**(i / frequency_dim)
        frequency_dim = emb_size // 4
        frequencies = 1.0 / (10000 ** (torch.arange(frequency_dim).float() / frequency_dim)) #[32,]
        # * 是逐元素乘法，都广播为[49,32],前者按列广播，后者按行广播，结果是一个两个[49,32],即一个x角度表，一个y角度表。
        # x角度表每一行表示一个patch的x坐标乘以32个频率，后面同理，y角度表每一行表示一个patch的y坐标乘以32个频率

        x_angles = x.flatten()[:, None] * frequencies[None, :] 
        y_angles = y.flatten()[:, None] * frequencies[None, :] 
        position = torch.cat([
            x_angles.sin(), x_angles.cos(),
            y_angles.sin(), y_angles.cos(),
        ], dim=-1)  # [49, 128]
        # (0,0)
        # (1,0)
        # (2,0)
        # (3,0)
        # (4,0)
        # (5,0)
        # (6,0)
        # (0,1)
        # (1,1)
        # (2,1)
        # (3,1)
        # (4,1)
        # (5,1)
        # (6,1)
        # ...共49行，每行有一个(x,y)坐标，每行128维，前64维是x的sin/cos，后64维是y的sin/cos

        self.register_buffer("pos_embed", position.unsqueeze(0))  # [49, 128] -> [1, 49, 128]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed.to(dtype=x.dtype)


class TimeEmbedding(nn.Module):
    """
    将连续时间 t 编码为可学习的条件向量
    默认输入：[B,]，默认输出：[B, 128]。
    
    """

    def __init__(self, emb_size=128, frequency_dim=256, time_scale=1000.0):
        super().__init__()
        self.time_scale = time_scale
        half_dim = frequency_dim // 2
        frequencies = 1.0 / (10000 ** (torch.arange(half_dim).float() / half_dim)) # [128,]
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, emb_size), # 128 -> 128
            nn.SiLU(),
            nn.Linear(emb_size, emb_size), # 128 -> 128
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """每张图片提供一个浮点时间t，返回 [B, emb_size]。"""
        # 内部放大时间，使 [0, 1] 的变化在正余弦特征中更明显。
        # 逐元素乘法,time_scale为常数，前者为[B,1]广播到[B,128],后者为[1,128]广播为[B,128]
        angles = t.float()[:, None] * self.time_scale * self.frequencies[None, :] 
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, 256]
        return self.mlp(features.to(dtype=self.mlp[0].weight.dtype))


class LabelEmbedding(nn.Module):
    """
    将服装类别编号映射为可学习向量
    默认输入：[B,]，默认输出：[B,128]。
    输入为 torch.int64 标签，Fashion-MNIST 的标签范围是 0 到 9。
    """

    def __init__(self, num_classes=10, emb_size=128):
        super().__init__()
        #Embedding的输入必须是整数索引，等价一个[10,128]的查找表，输入0-9，输出对应行向量
        self.embedding = nn.Embedding(num_classes, emb_size)
        # 从较小的随机向量开始学习每个类别的表示。
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """输入整数标签 [B]，返回对应类别向量 [B, emb_size]。"""
        return self.embedding(labels)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    用条件生成的平移和缩放调制特征，同一图片的所有 token 共享参数。
    参数：
        x: [B, 49, 128]，Transformer block 的 token 特征。
        shift: [B, 128]，条件分支生成的平移向量。
        scale: [B, 128]，条件分支生成的缩放向量。
    """
    # 逐元素乘法 
    return x * (1 + scale[:, None, :]) + shift[:, None, :] 


class DiTBlock(nn.Module):
    """
    使用 adaLN-Zero 注入条件的 Transformer block。
    默认输入: x  [B, 49, 128], condition 为 [B, 128]
    默认输出: [B, 49, 128]
    condition 由 TimeEmbedding(t) + LabelEmbedding(labels) 得到。
    """

    def __init__(self, emb_size=128, num_heads=4, mlp_ratio=4):
        super().__init__()
        # 标准 LayerNorm 在归一化后还会进行仿射变换
        # 这里关闭 LayerNorm 的可学习仿射参数，改用条件分支提供的 shift/scale。
        self.norm1 = nn.LayerNorm(emb_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size, elementwise_affine=False, eps=1e-6)
        mlp_size = int(emb_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_size),  # 默认 128 -> 512
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_size, emb_size),  # 默认 512 -> 128
        )

        # 为注意力和 MLP 各生成 shift、scale、gate，共 6 组 [B, emb_size]。
        # 128 -> 6*128=768，最后 .chunk 为 6个[B,128]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_size, 6 * emb_size),
        )
        # 零初始化让残差门控初始为 0，整个 block 初始为恒等映射。
        # 注意力机制和mlp均是残差连接，门控经过学习从0开始变大，注意力和 MLP 分支才逐步参与特征更新。
        # 将adaLN_modulation零初始化是最稳定可控的选择
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """条件调制 -> 自注意力残差 -> 条件调制 -> MLP 残差。"""
        # 从condition生成6个[B,128]，分别用于注意力和MLP的shift、scale、gate
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )

        # 将condition生成的注入x，使用attn的shift和scale调制x的特征，得到h
        h1 = modulate(self.norm1(x), shift_attn, scale_attn)
        # 不需要返回注意力权重，直接返回更新后的 token 特征 h[B,49,128]
        h1, _ = self.attn(h1, h1, h1, need_weights=False)
        # 残差连接
        x = x + gate_attn[:, None, :] * h1

        # 将condition生成的注入x，使用mlp的shift和scale调制x的特征，得到h
        h2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        # 残差连接
        x = x + gate_mlp[:, None, :] * self.mlp(h2)
        return x


class FinalLayer(nn.Module):
    """
    条件归一化后，将每个 token 投影为 patch 像素速度。

    默认输入: x[B, 49, 128], condition[B, 128]， 
    默认输出: v[B, 49, 16],像素速度预测，后面经flow matching得到下一步图像。
    最终层只使用 shift 和 scale，
    不再使用残差门控，也不添加限制速度范围的激活函数。
    """

    def __init__(self, emb_size=128, patch_size=4, out_channels=1):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size, elementwise_affine=False, eps=1e-6)
        # 128 -> 2*128=256，最后 .chunk 为 shift 和 scale 两个 [B,128]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_size, 2 * emb_size),
        )
        # 128 -> 16，投影为每个 patch 的像素速度，Fashion-MNIST 为 1 个通道。
        self.linear = nn.Linear(emb_size, patch_size ** 2 * out_channels)
        # 零初始化，让仿射调制从恒等开始，让速度预测从0开始
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class Unpatchify(nn.Module):
    """
    按空间位置把每个 patch 的像素预测拼回图像，无可学习参数。

    默认输入：[B, 49, 16]，默认输出：[B, 1, 28, 28]。
    49 个 patch 按从左到右、从上到下排列；每个 patch 内部的展平顺序为
    [patch 内部行, patch 内部列, 通道]。多通道时，通道维变化最快。

    参数：
        out_channels: 输出图像的通道数，Fashion-MNIST 为 1。
        patch_size: 每个正方形 patch 的边长，需要与 PatchEmbedding 一致。
        img_size: 输出正方形图像的边长，必须能被 patch_size 整除。
    """

    def __init__(self, out_channels=1, patch_size=4, img_size=28):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_dim = patch_size ** 2 * out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 [B, num_patches, patch_dim] 还原为 [B, C, H, W]。"""
        batch_size = x.size(0)
        g, p, c = self.grid_size, self.patch_size, self.out_channels
        # 分开网格坐标和 patch 内部坐标：[B, 网格行, 网格列, 内部行, 内部列, C]。
        x = x.reshape(batch_size, g, g, p, p, c)
        # 相邻的“网格行、内部行”构成整图高度，“网格列、内部列”构成整图宽度。
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, 网格行, 内部行, 网格列, 内部列]
        # permute 后可能不连续，reshape 会在需要时整理内存，并保留梯度关系。[B,1,28,28]
        return x.reshape(batch_size, c, self.img_size, self.img_size)


class DiT(nn.Module):
    """
    用于类别条件 Flow Matching 的完整像素空间 DiT。

    输入：xt [B, 1, 28, 28]、连续时间 t [B,]、整数类别 labels [B,]。
    输出：同形状的速度场 [B, 1, 28, 28]
    """

    def __init__(self, img_size=28, patch_size=4, in_channels=1,
                 emb_size=128, depth=4, num_heads=4, mlp_ratio=4, num_classes=10):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size) # [B, 1, 28, 28] -> [B, 49, 128]
        self.position = PositionalEncoding2D(emb_size, img_size // patch_size) # 生成二维位置编码 [1, 49, 128]
        self.time_embed = TimeEmbedding(emb_size) # 将连续时间 t 编码为可学习的条件向量 [B, 128]
        self.label_embed = LabelEmbedding(num_classes, emb_size) # 将服装类别编号映射为可学习向量 [B, 128]
        # 创建4个dit block，对nn使用modulelist来存放多个nn，而不是使用list
        self.blocks = nn.ModuleList([
            DiTBlock(emb_size, num_heads, mlp_ratio) for _ in range(depth)
        ]) # 每个block输入x[B,49,128],condition[B,128]，输出[B,49,128]
        self.final_layer = FinalLayer(emb_size, patch_size, in_channels) # [B, 49, 128] -> [B, 49, 16]
        self.unpatchify = Unpatchify(in_channels, patch_size, img_size) # [B, 49, 16] -> [B, 1, 28, 28]

    def forward(self, xt: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        tokens = self.position(self.patch_embed(xt))  # 逐个patch嵌入 + 二维位置编码 [B, 49, 128]
        condition = self.time_embed(t) + self.label_embed(labels)  # 生成条件向量 [B, 128]
        for block in self.blocks:
            tokens = block(tokens, condition) # 做四次dit 自注意力+mlp，更新特征，输出[B,49,128]
        patches = self.final_layer(tokens, condition)  # 再经过一次映射得到速度场 [B,49,128] -> [B, 49, 16]
        return self.unpatchify(patches)  # unpatchify得到 flow matching可用的速度场 [B, 1, 28, 28]
