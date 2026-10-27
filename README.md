# Mini DiT Flow

使用 **Flow Matching** 训练类别条件 **DiT**，从高斯噪声生成 Fashion-MNIST 的 `28×28` 灰度服装图片。项目直接在像素空间建模，适合阅读训练损失、Transformer 图像建模和 Euler 生成的核心代码。

训练、续训和生成均由 YAML 配置控制，命令行只需指定 `--config`。参数的逐项说明见 [小模型配置](configs/default.yaml) 和 [大模型配置](configs/dit_20m.yaml)。

## 快速开始

使用 Python 3.10。本机已有环境可通过 `conda activate d2l` 激活；使用其他环境时，同样安装项目依赖即可。

在项目根目录执行以下命令，所有配置中的相对路径都相对于运行命令时的工作目录：

```bash
python -m pip install -r requirements.txt

# 首次下载数据并保存真实图片预览；数据已有时可以省略 --download。
python data.py --download

# 训练小模型，然后使用最佳 checkpoint 生成十类图片。
python train.py --config configs/default.yaml
python sample.py --config configs/default.yaml
```

默认训练 30 轮，结果写入 `outputs/`；生成图片保存为 `outputs/samples/generated.png`。checkpoint、日志和图片由运行生成，生成前需先完成训练或准备兼容的 checkpoint。

## 配置与常用操作

### 选择实验配置

| 配置文件 | 模型 | 训练设置 | 训练输出目录 |
| --- | --- | --- | --- |
| [default.yaml](configs/default.yaml) | 128 维、4 层、4 个注意力头，约 1.28M 参数 | batch size 128，30 轮 | `outputs/` |
| [dit_20m.yaml](configs/dit_20m.yaml) | 512 维、4 层、8 个注意力头，约 19.85M 参数 | batch size 512，50 轮 | `outputs/dit-20m-b512-e50/` |

训练和生成共用一份完整实验配置。顶层设置随机种子、设备、CPU 线程数和训练输出目录；`data`、`model`、`train`、`sample` 分别设置数据、模型、训练和生成参数。

`config.py` 在启动时集中校验全部字段的类型、范围和模型维度关系，并按运行模式检查所需 checkpoint。生成或续训时，`model` 段须与 checkpoint 中保存的模型配置一致。从头训练时，`sample.checkpoint` 可以指向本次训练将要生成的文件。

保留配置中的全部字段；缺失字段或拼错字段会阻止启动。以下 YAML 片段只展示需要修改的字段，请在现有配置中修改并保留其余内容。

### 从头训练

确认 `train.resume: null`，然后运行：

```bash
python train.py --config configs/default.yaml
```

新建实验时，可复制一份完整配置，并修改 `output_dir`、`sample.checkpoint` 和 `sample.output`，让三者指向对应实验。`sample` 的两个路径是独立设置，不会随 `output_dir` 自动变化。同一输出目录中的同名产物会被覆盖。

`device: auto` 自动选择可用的 CUDA 或 CPU。大模型模板指定 `device: cuda`，运行前需要可用的 CUDA；也可以修改为 `auto` 或 `cpu`。

### 断点续训

保留原实验的模型、数据配置和 `output_dir`，修改：

```yaml
train:
  resume: outputs/checkpoints/last.pt
  epochs: 60
```

再运行同一训练命令。`epochs` 表示目标总轮数：从第 30 轮的 checkpoint 恢复、设为 60，会继续训练第 31～60 轮。程序恢复模型、优化器、混合精度 scaler 和随机状态。

### 生成全部类别或指定类别

在 `sample` 中设置要加载的 `checkpoint` 和生成图片的 `output`，然后运行：

```bash
python sample.py --config configs/default.yaml
```

| 设置 | 生成行为 |
| --- | --- |
| `class_id: null`、`samples_per_class: 8` | 生成全部十类，每类一行、每行 8 张，共 80 张 |
| `class_id: 9`、`num_samples: 16` | 只生成 16 张短靴图片 |
| `compare_steps: null`、`steps: 50` | 使用 50 步 Euler 积分生成 |

顶层 `seed` 控制生成的初始噪声。固定 seed、类别和数量，可以复用同一组起点比较效果。生成过程只做前向计算，不更新模型参数。

### 比较采样步数

修改 `sample` 段：

```yaml
sample:
  output: outputs/samples/comparison.png
  compare_steps: [10, 25, 50, 100]
```

运行同一生成命令，使用相同噪声分别生成四组图片。此时 `compare_steps` 覆盖 `steps`，输出为 `comparison_steps_10.png` 等图片，以及 `comparison_timings.csv`。计时包括积分计算，不包括模型加载和图片保存。

训练期间的预览始终使用 `sample.steps` 和 `sample.samples_per_class`，生成全部十类；指定类别、步数对比和 `sample.output` 仅影响独立生成。

### 训练大模型

```bash
python train.py --config configs/dit_20m.yaml
python sample.py --config configs/dit_20m.yaml
```

省略 `--config` 时，训练和生成默认使用 `configs/default.yaml`。

## 数据与独立预览

Fashion-MNIST 的官方训练集有 60000 张图片，默认划分为 55000 张训练图片和 5000 张验证图片；官方测试集另有 10000 张，仅在训练结束后用于最终评估。训练图片每轮打乱，验证和测试保持顺序，各集合均保留最后一个不足批次。

`data.py` 使用 torchvision 加载数据，将像素归一化到 `[-1, 1]`。每批图片形状为 `[B, 1, 28, 28]`，类别标签形状为 `[B]`。

| 编号 | 类别 | 编号 | 类别 |
| --- | --- | --- | --- |
| 0 | T 恤 / 上衣 | 5 | 凉鞋 |
| 1 | 裤子 | 6 | 衬衫 |
| 2 | 套衫 | 7 | 运动鞋 |
| 3 | 连衣裙 | 8 | 包 |
| 4 | 外套 | 9 | 短靴 |

独立预览用于查看真实数据，保留自己的命令行参数，不读取训练配置：

```bash
python data.py
python data.py --num-images 16 --nrow 4 --preview outputs/data-preview.png
```

预览会把像素还原为显示范围，并在每张图片下标注类别。

## 训练与生成原理

模型接收中间图像 `xt`、时间 `t` 和类别 `y`，输出与输入图像同形状的速度场。默认模型将图片切成 49 个 `4×4` patch，经二维位置编码、时间与类别条件、4 层使用 adaLN-Zero 的 DiT block，再还原为 `[B, 1, 28, 28]` 的速度预测。

训练时，对归一化真实图片 `x1` 采样高斯噪声 `x0` 和时间 `t`：

```text
x0 ~ N(0, I),    t ~ Uniform(0, 1)
xt = (1 - t) * x0 + t * x1
target_velocity = x1 - x0
loss = mean((model(xt, t, y) - target_velocity)²)
```

`train.py` 中标出的“训练核心”调用 `flow_matching_loss` 计算损失，再反向传播并更新参数。路径构造和损失公式在 `flow.py` 中实现。每轮结束后计算验证损失，保存模型，并按配置定期生成固定噪声预览。CUDA 上启用 AMP 时优先使用 BF16，否则使用 FP16 和梯度缩放。

生成时，从高斯噪声出发，按 Euler 方法从 `t=0` 积分到 `t=1`：

```text
dt = 1 / steps
x = noise
for i = 0, ..., steps - 1:
    x = x + dt * model(x, i * dt, y)
```

`sample.py` 中标出的“生成核心”调用 `euler_sample` 完成积分。显示或保存图片时，再将像素转为 `[0, 1]`。

## 运行产物

以下训练路径相对于 `output_dir`：

| 路径 | 内容 |
| --- | --- |
| `config.yaml` | 本次训练的配置快照；自动保存时不保留源 YAML 注释 |
| `checkpoints/last.pt` | 最近一轮的完整状态，用于续训 |
| `checkpoints/best.pt` | 验证损失最低的 checkpoint，用于生成和最终测试 |
| `checkpoints/epoch_XXX.pt` | 每隔 `train.sample_every` 轮及最后一轮的阶段状态 |
| `samples/epoch_XXX.png` | 对应阶段的固定噪声预览 |
| `logs/train.log` | 训练进度和最终评估日志 |
| `logs/loss.csv` | 每轮训练 MSE、验证 MSE 和训练/验证耗时 |
| `logs/metrics.json` | 最佳轮数、最佳验证 MSE 和测试 MSE |

完整 checkpoint 包含模型、优化器、scaler、配置、训练进度和随机状态。独立生成保存到 `sample.output`。

## 目录与代码阅读

| 路径 | 作用 |
| --- | --- |
| `configs/` | 含参数注释的运行配置 |
| `config.py` | 读取配置、集中校验、加载所需 checkpoint |
| `data.py` | 数据接口、归一化和图片预览 |
| `model.py` | 图像切块、条件编码、DiT block 和图像重建 |
| `flow.py` | Flow Matching 损失与 Euler 采样 |
| `train.py` | 准备训练、参数更新、验证、保存和最终测试 |
| `sample.py` | 准备类别与噪声、生成、图片保存和步数对比 |
| `tests/` | 配置校验、数值公式和模型连接测试 |
| `data/`、`outputs/` | 本地数据和运行产物 |

阅读主线可以从 `data.py`、`model.py`、`flow.py` 开始，再看 `train.py` 和 `sample.py` 中已标注的核心代码。配置解析、设备设置、日志和计时等准备工作有单独注释。

## 测试

运行自动化测试：

```bash
python -m unittest discover -s tests -v
```

目前共 9 项测试：6 项覆盖配置和 checkpoint 校验，3 项覆盖 Flow Matching 的路径/速度方向、Euler 积分以及完整 DiT 的输出形状和梯度传播。测试不会在正常训练或生成时自动执行。

MSE 衡量速度回归误差，生成质量还需结合图片判断。
