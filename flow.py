"""直线路径 Flow Matching：训练时回归速度，生成时积分速度场。"""

import torch
from torch.nn import functional as F


def flow_matching_loss(model, images, labels, generator=None):
    """
    以归一化真实图片 x1 和类别label为输入，返回标量 MSE 损失。

    输入
    - model: 训练好的 DiT 模型
    - images: 归一化的真实图片 [B, 1, 28, 28]
    - labels: 图片对应的类别标签 [B,]
    - generator: 随机数生成器,为None时使用pytorch默认的生成器

    每张图像独立采样噪声 x0 和时间 t，约定 t=0 是噪声、t=1 是图片。
    模型的 train/eval 状态由调用方控制。
    """
    x1 = images # [B, 1, 28, 28]
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator) # [B, 1, 28, 28]，纯噪声
    t = torch.rand(x1.size(0), device=x1.device, dtype=x1.dtype, generator=generator)
    time = t[:, None, None, None]  # [B] -> [B, 1, 1, 1]
    xt = (1 - time) * x0 + time * x1
    target_velocity = x1 - x0  # 目标速度，为xt对时间的导数 [B, 1, 28, 28]
    predicted_velocity = model(xt, t, labels) # 预测速度，为模型的输出 [B, 1, 28, 28]
    return F.mse_loss(predicted_velocity, target_velocity)


@torch.no_grad() # 装饰器，在函数执行时不计算梯度
def euler_sample(model, noise, labels, steps=50):
    """
    从给定噪声出发，沿 t: 0 -> 1 积分，返回生成图片张量。

    输入
    - model: 训练好的 DiT 模型
    - noise: 归一化初始的噪声张量 [B, 1, 28, 28]
    - labels: 图片对应的类别标签 [B,]
    - steps: 积分步数
    """
    dt = 1.0 / steps
    was_training = model.training
    model.eval()
    try:
        x = noise.clone()  # 保留原始噪声，便于用相同起点比较不同steps设置的效果。
        for i in range(steps):
            # torch.full可用按指定形状、用同一个常数填满创建的张量
            t = torch.full((x.size(0),), i * dt, device=x.device, dtype=x.dtype) # [B,]，每张图像的时间相同
            x = x + dt * model(x, t, labels)
        return x
    finally:
        # 即便采样中断，也恢复调用前的状态，方便训练过程中定期生成预览。
        model.train(was_training)
