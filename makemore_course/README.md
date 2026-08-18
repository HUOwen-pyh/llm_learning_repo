# Makemore：从 Bigram 到 WaveNet

这是一套基于 Andrej Karpathy《Neural Networks: Zero to Hero》第 2～6 讲的
练习框架。它承接 `micrograd_course`，但从这里开始使用 PyTorch 张量与 autograd。

主线严格按照：

```text
Bigram
  -> MLP language model
  -> activations / gradients / BatchNorm
  -> manual tensor backpropagation
  -> WaveNet-inspired hierarchical model
```

本项目共有 52 个必做关卡。绘图、统计分析和张量形状追踪属于正式验收内容，
不是可跳过的附录。

## 安装

当前项目需要 Python 3.10+、PyTorch、Matplotlib 和 NumPy：

```powershell
cd C:\Users\misak\llmlearning\makemore_course
python -m pip install -r requirements.txt
```

如果需要 CUDA 版本的 PyTorch，请按照 PyTorch 官方安装页面选择与你机器匹配的命令，
然后再安装 `matplotlib` 和 `numpy`。

## 数据

仓库内置 `data/names_mini.txt`，供快速检查使用。完整训练使用 Karpathy 的
`names.txt`：

```powershell
python download_names.py
```

下载后文件位于 `data/names.txt`。

## 使用方式

先查看全部任务：

```powershell
python check.py --list
```

按顺序检查到某一关：

```powershell
python check.py 1
python check.py 10
```

也可以检查一个阶段：

```powershell
python check.py bigram
python check.py mlp
python check.py diagnostics
python check.py backprop
python check.py wavenet
```

第一次检查会停在第一个 `TODO`，这是预期行为。完成全部快速检查：

```powershell
python check.py all
```

完整数据训练与最终工件生成由各阶段 demo 负责：

```powershell
python demos/01_bigram.py --full
python demos/02_mlp.py --full
python demos/03_diagnostics.py --full
python demos/04_manual_backprop.py --full
python demos/05_wavenet.py --full
```

`python check.py all` 验收的是快速、确定性的函数关卡，不会替你跑数十万步训练。
五个 demo 完成后，再执行最终组合验收：

```powershell
python check.py --artifacts
```

它会按照固定清单检查每个 PNG、JSON、CSV 和 TXT 是否存在、非空且格式有效；
因此“函数全过但图表/实验没有真正生成”仍不算结课。

## 目录

```text
makemore_course/
├─ TASKS.md
├─ check.py
├─ requirements.txt
├─ download_names.py
├─ data/
│  ├─ README.md
│  └─ names_mini.txt
├─ artifacts/                  # 必做图表、CSV、JSON、样本输出
├─ demos/
│  ├─ 01_bigram.py
│  ├─ 02_mlp.py
│  ├─ 03_diagnostics.py
│  ├─ 04_manual_backprop.py
│  └─ 05_wavenet.py
├─ checks/                     # 快速、确定性的分关验收
└─ makemore/
   ├─ data.py
   ├─ bigram.py
   ├─ mlp.py
   ├─ training.py
   ├─ diagnostics.py
   ├─ layers.py
   ├─ manual_backprop.py
   ├─ wavenet.py
   ├─ plotting.py
   └─ reporting.py
```

## 范围说明

- Bigram 阶段同时实现计数模型和单层神经模型。
- MLP 阶段保留显式的 `C/W1/b1/W2/b2`，方便后续分析内部张量。
- BatchNorm 阶段必须检查训练态、推理态、running statistics 和 3D 轴问题。
- 手写反传阶段禁止在学生实现路径中调用 `loss.backward()`；checker 会用 autograd 作 oracle。
- 手写反传训练结束后还要校准 BN 的推理统计，采样时不能拿单个样本重新计算 batch 方差。
- WaveNet 阶段实现视频中的树状模型，不是完整的原版音频 WaveNet。
- 快速检查使用迷你语料；若 `data/names.txt` 已存在，还会额外核对官方语料的关键计数与 split 形状。

官方课程材料：

- https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/makemore
- https://github.com/karpathy/nn-zero-to-hero
