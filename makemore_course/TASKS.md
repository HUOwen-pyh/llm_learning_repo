# Makemore 任务清单

规则：每次只完成一个编号，搜索对应的 `TODO NN`。`python check.py N` 会从第 1 关检查到第 N 关。
快速检查只使用人工小张量或 `names_mini.txt`；完整语料训练由 demo 的 `--full` 模式执行。

所有绘图函数统一返回 `(fig, axes)`，不得调用 `plt.show()`。checker 不只数坐标轴：
每一张必做图都必须含有真实的 line、bar、histogram、image、scatter 或文字标注，空白 subplot 不算完成。
函数关卡全部通过后，还必须运行五个 demo，并用 `python check.py --artifacts` 验收固定工件清单。

## 总览

| 阶段 | 编号 | 主题 |
|---|---:|---|
| Bigram | 01–10 | 计数语言模型、NLL、采样、神经 Bigram、图表 |
| MLP | 11–20 | 上下文窗口、embedding、训练、评估、超参数、图表 |
| 内部诊断与 BatchNorm | 21–33 | 初始化、饱和、Kaiming、BN、激活/梯度/update 统计 |
| 手写张量反向传播 | 34–42 | 原子 CE、BN、embedding 累积、完整无 autograd 训练 |
| WaveNet-inspired | 43–52 | 8 字符上下文、树状结构、3D BN、形状与感受野分析 |

---

# 第一阶段：Bigram（01–10）

### 01. 读取语料与基础统计

实现 `load_words()` 和 `corpus_stats()`：去除空行、转小写，并报告名字数、最短/最长长度、平均长度和字符集合。

验收重点：统计必须来自清洗后的单词；长度统计不包含边界符。

### 02. 词表、编码与解码

实现 `Vocabulary` 和 `build_vocab()`。边界字符 `.` 固定为索引 0，其余字符稳定排序。编码后再解码必须还原原字符串。

### 03. 生成 bigram 训练对

实现 `build_bigram_pairs()`。单词 `emma` 应展开成：

```text
.->e  e->m  m->m  m->a  a->.
```

输出 `x`、`y` 都是 `torch.long` 的一维张量。

### 04. 27×27 计数矩阵

实现 `count_bigrams()`。重复 bigram 必须累加；总计数应等于所有名字的 `len(word)+1` 之和。

### 05. 概率、广播与 smoothing

实现 `normalize_counts(counts, smoothing)`：按行归一化，使用 `keepdim=True` 明确广播方向。加入伪计数后不允许出现零概率，每行和应为 1。

### 06. NLL 与 perplexity

实现 `bigram_nll()` 和 `perplexity()`。NLL 是正确目标字符概率的平均负对数；perplexity 为 `exp(nll)`。零概率必须得到明确的无限损失或通过 smoothing 避免。

### 07. 自回归采样

实现 `sample_bigram()`：从 `.` 开始，每次按当前行概率采样，采到 `.` 停止。必须支持固定 `torch.Generator` 和最大长度。

### 08. Bigram 三张必做图

实现并生成：

- `bigram_name_lengths.png`：名字长度直方图；
- `bigram_counts.png`：带字符和计数标注的 27×27 热力图；
- `bigram_next_char.png`：指定字符的下一字符概率柱状图。

绘图函数必须返回 `(fig, axes)`，不能在内部调用 `plt.show()`。

### 09. 神经 Bigram

实现 `neural_bigram_logits()`、`manual_softmax()`、`neural_bigram_loss()`：

```text
字符索引 -> one-hot -> W -> logits -> probabilities -> CE
```

必须证明 `one_hot(x) @ W` 与 `W[x]` 等价，并将手写 CE 与 `F.cross_entropy` 对照。

### 10. 训练、trigram 与模型比较

实现神经 Bigram 训练、L2 正则、`count_trigrams()`、`normalize_trigram_counts()` 和 `trigram_nll()`。
`smoothing_search()` 明确搜索的是**计数 trigram** 的伪计数强度：对每个 strength 只用 train words
建立 `(V,V,V)` 计数，沿最后一个“目标字符”轴归一化，再分别计算 train/dev trigram NLL。
本关 demo 可以对 words 做显式、互斥的按单词切片；第 11 关再把可复现的通用划分封装成 `split_words()`。

必须生成：

- `bigram_smoothing.png`：smoothing 强度与 train/dev NLL；
- `bigram_model_diff.png`：计数模型与神经模型概率差异热力图；
- `bigram_samples.txt`：固定种子的采样结果。

同时实现 `save_lines()`：每个样本写一行 UTF-8 文本，非空文件末尾保留换行。

---

# 第二阶段：MLP（11–20）

### 11. 按单词划分 train/dev/test

实现 `split_words()`：默认按单词以 80%/10%/10% 划分，再分别构造样本。
禁止先把 bigram/context 样本混合后随机切分，否则同一个名字会泄漏到多个集合。三个集合必须互斥、
合并后恰好恢复全部输入单词；固定 seed 必须可复现。

若存在官方 `data/names.txt`，额外 sanity check：总名字数 32033、bigram 样本数 228146；
seed=42 的 block=3 数据形状应为 train `(182441,3)`、dev `(22902,3)`、test `(22803,3)`。

### 12. 上下文窗口数据集

实现 `build_context_dataset(words, vocab, block_size=3)`。`emma` 应产生：

```text
...->e  ..e->m  .em->m  emm->a  mma->.
```

输出形状 `(N, block_size)` 和 `(N,)`。

### 13. Embedding lookup 与形状

实现 `embedding_lookup()`、`flatten_embeddings()` 和 `tensor_layout()`：

```text
C[X]                  -> (B, block_size, n_embd)
flatten_embeddings()  -> (B, block_size*n_embd)
```

`tensor_layout()` 必须记录 shape、stride、是否 contiguous；不可把 batch 轴错误地展平。

### 14. 初始化显式 MLP 参数

实现 `MLPConfig`、`MLPParameters`、`init_mlp()`、`clone_parameters()` 和 `parameter_count()`。
本阶段保留显式 `C/W1/b1/W2/b2`，不要先藏进 `torch.nn.Module`。克隆结果必须是数值相同、
storage 独立且 `requires_grad=True` 的叶张量，后续 LR sweep 不得污染正式参数。

官方配置 `vocab=27, block=3, emb=10, hidden=200` 的参数量应为 11897。

### 15. 前向传播与缓存

实现 `mlp_forward(..., return_cache=True)`，缓存：

```text
emb, embcat, hpreact, h, logits
```

对应形状为 `(B,3,10) -> (B,30) -> (B,200) -> (B,200) -> (B,27)`。

### 16. 稳定交叉熵

实现手写稳定 CE，并与 `F.cross_entropy` 对照。必须用极端 logits 演示直接 `exp()` 会溢出，而减去行最大值或官方 CE 仍为有限数。

### 17. Minibatch 与一步训练

实现可复现的 `sample_minibatch()` 和 `mlp_train_step()`。训练顺序固定为：前向、loss、清梯度、backward、更新。
在同一个小 batch 上用足够小的学习率时，一步更新后的 loss 必须低于更新前，借此排除梯度上升和漏更新。

### 18. Learning-rate range test

在对数区间内扫描学习率，记录学习率指数与 loss，生成 `mlp_lr_range.png`。函数必须在独立克隆上运行；
扫描前后逐个比较，调用者传入的正式参数不得改变。

### 19. 训练历史、三集合评估和容量实验

实现 `fit_mlp()`、`evaluate_mlp()` 与超参数实验记录。`TrainingHistory` 要分别保存 raw step、
smoothed step、eval step 及其对应值，不能用一条长度不匹配的 x 轴含糊地画三组曲线。必须生成：

- `mlp_training.png`：原始 loss、滑动平均 loss、train/dev loss；
- `mlp_capacity.png`：参数量与 dev loss；
- `mlp_experiments.json`：每组配置和指标。

实现 `save_json()`，它必须能直接保存 dataclass 或 dataclass 序列，而不是要求 demo 手工拆字段。

### 20. Embedding 与模型输出分析

训练二维 embedding 版本并生成：

- `mlp_embeddings.png`：带字符标签的二维 embedding；
- `mlp_prediction_frequency.png`：模型采样字符频率与真实频率；
- `mlp_entropy.png`：预测熵分布；
- `mlp_samples.txt`：自回归生成的名字。

采样必须从全 `.` context 开始、每步左移追加、遇 `.` 停止，并受 `max_length` 约束；固定 generator
要可复现。本框架用 `character_frequencies(..., ignore_index=0)` 让真实与生成数据都排除边界符，
排除后必须重新归一化；不得只在其中一边统计 `.`。

---

# 第三阶段：激活、梯度与 BatchNorm（21–33）

### 21. 均匀预测基线

实现 `uniform_nll(vocab_size)` 与 `initialization_report()`。27 类均匀预测的理论损失是 `log(27)`；报告 logits mean/std、CE、平均最大概率和平均熵。

### 22. 修复“自信地犯错”的输出层

比较未经缩放和小尺度输出层初始化。生成 `init_logits.png`，必须包含两组 logits 直方图和早期 loss 曲线；修复后的初始 CE 应更接近 `log(27)`。

### 23. tanh 饱和率与热图

实现 `tensor_stats()` 和 `tanh_saturation(threshold=0.97)`，生成：

- `tanh_histograms.png`：`hpreact` 与 `tanh(hpreact)` 直方图；
- `tanh_saturation.png`：batch×neuron 饱和布尔热图。

统计报告必须包含 mean/std/min/max/finite ratio/saturation ratio，并明确 std 使用的 correction；
checker 会用已知张量逐项核对公式，不能用占位常数。

### 24. Kaiming 与方差传播

实现 `kaiming_scale(fan_in, gain)` 和多层方差传播实验。比较 gain=1、5/3、3，生成 `variance_propagation.png`，展示各层 activation std 与 saturation。

### 25. 手写二维 BatchNorm

实现 `BatchNorm1d` 的训练态二维前向：按 batch 轴计算 mean/variance，使用可训练 gamma/beta，
并保存 running_mean/running_var。整套课程统一保存 variance（不是 std），训练态使用样本方差
`sum((x-mean)^2)/(n-1)`；gamma/beta 使用一维 `(C,)`，在 3D channel-last 输入上广播为 `(1,1,C)`。

### 26. train/eval 与 running stats

实现 momentum 更新、`train()` 和 `eval()`。eval 不得更新 buffer；单样本推理必须使用 running statistics，
同一样本的结果不能依赖同批 companions。完整深网的逐层校准放到第 29 关，避免本关依赖尚未实现的 DeepMLP。

### 27. Batch coupling 与 bias 冗余

验证同一样本处于两个不同 companion batch 时训练态 BN 输出会变化；验证训练态
`BN(xW+b)` 与 `BN(xW)` 近似相同。`bn_batch_coupling.png` 必须画前一个“同一样本、不同 batch”的对照，
bias 冗余则作为同时报告的数值不变量。

### 28. BatchNorm folding

先完成轻量 `Linear` 的构造、任意最后一维矩阵乘法与参数列表，再实现推理期
`fold_batchnorm(linear, bn)`，将 Linear+BN 合并为一个**新** Linear，不得原地修改输入层。
必须同时支持原 Linear 有/无 bias；随机输入 logits 的最大绝对误差必须小于 `1e-5`（不是依赖相对容差）；
生成 `bn_folding.png` 对照散点图。此变换只适用于固定 running statistics 的 eval 模式。

### 29. PyTorch 风格积木与深层 MLP

复用第 28 关的 `Linear`，实现 `Tanh`、`Sequential` 和 `DeepMLP`。每层保存 `.out`，所有参数不重复，
train/eval 模式能向下传播；实现 `calibrate_batchnorm()`，在完整校准集上逐层得到精确 mean/sample variance、
覆盖所有 BN，并在结束后恢复模型原先的 train/eval 状态。

官方五隐藏层配置参数量应为 47024。

### 30. 前向 activation statistics

对每个 Tanh 输出统计 mean/std/min/max/finite ratio/saturation，并生成 `activation_histograms.png`
和 `activation_summary.png`。必须覆盖所有隐藏层，且报告值会对照层的真实 `.out` 逐项核验。

### 31. 反向 activation-gradient statistics

前向后对中间输出调用 `retain_grad()`，反向后读取 `layer.out.grad`，生成 `activation_gradient_histograms.png`。禁止用参数梯度冒充激活梯度。

### 32. 参数梯度与 update:data

统计每个参数的 data std、grad mean/std、grad:data，以及：

```text
log10(std(learning_rate * grad) / std(parameter))
```

生成 `parameter_gradient_histograms.png` 和 `update_ratios.png`，图中包含经验参考线 `-3`。零方差参数必须显式标为 N/A 或使用记录在报告中的 epsilon。

### 33. 统一 dashboard 与全零初始化尸检

在同一 seed、同一 batch 下比较四个正式诊断配置：`gain_1`、`gain_5_3`、`gain_3`、`batchnorm`；
另做全零初始化尸检。生成：

- `diagnostics_dashboard.png`：激活、激活梯度、参数梯度、update:data；
- `zero_init_pathology.png`：各参数 grad norm 与预测字符边际分布；
- `diagnostics.json`：全部原始统计。

dashboard 的每个配置都必须包含 activation、activation-gradient、parameter-gradient 和随训练 step
记录的 update:data 四类真实数据。全零权重网络必须展示隐藏神经元对称、首轮隐藏权重梯度为零，
只有最终输出 bias（或最终 BN beta）首轮可获得非零梯度；训练后的所有 context 预测行应相同，
且该共同分布要与训练集 `bincount(y)/N` 的 unigram 边际分布对照。

---

# 第四阶段：手写张量反向传播（34–42）

学生实现路径不得调用 `loss.backward()` 或 `torch.autograd.grad()`；checker 会单独用 autograd 作为参考答案。

### 34. Debug 参数与原子化 forward cache

实现 `ManualParameters`、`ManualForwardCache`、`manual_forward()` 和严格的 `compare_grad()`。
为保证固定 seed 与 checker 完全一致，按下列顺序从同一个 generator 取样：

```text
C      = randn(vocab, emb)
W1     = randn(block*emb, hidden) * (5/3) / sqrt(block*emb)
b1     = randn(hidden) * 0.1
W2     = randn(hidden, vocab) * 0.1
b2     = randn(vocab) * 0.1
bngain = randn(1, hidden) * 0.1 + 1
bnbias = randn(1, hidden) * 0.1
```

这样 bias/gain 的非零小随机值不会掩盖错误；hidden=64 时总参数量为 4137。cache 包含 embedding、两层 Linear、tanh、BN 和原子化 CE 的全部中间张量，
包括 `bnmean/bndiff/bndiff2/bnvar/bnvar_inv/bnraw`；BN 必须明确使用 `/ (n-1)`。

### 35. 原子 CrossEntropy 反传

从 `loss -> logprobs -> probs -> counts -> normalized logits` 逐节点反传。reduction 后梯度形状必须恢复正确，分支贡献必须累加。
checker 的 autograd oracle 与学生 forward 使用隔离的参数/cache；不允许直接抄某个中间张量残留的 `.grad`。

### 36. 化简 CE backward 与 dlogits 热图

实现一行向量化的 softmax-cross-entropy 梯度，与原子实现及 autograd 对照。每行梯度和应接近 0。生成 `manual_dlogits.png`。

### 37. 第二层 Linear 与 tanh backward

实现通用 `linear_backward()` 和 `tanh_backward()`，核对 `dx/dW/db` 的形状和值。

### 38. 原子 BatchNorm backward

沿 gamma/beta、normalize、inverse std、sample variance、center、mean 的每个节点手推梯度；明确使用 `/ (n-1)` 的样本方差约定。
返回并审计 `dbnraw/dbnvar_inv/dbnvar/dbndiff2/dbndiff/dbnmean` 等全部原子节点，所有梯度必须先严格核对 shape，
再核对数值，不能让广播掩盖错误。

### 39. 化简 BatchNorm backward

实现 `batchnorm_backward_fused()`，与原子实现及 autograd 在多组 batch/channel 大小、负 gamma 和不同 eps 下对照；
至少覆盖三组 `(batch, channels, eps)`，并检查 `sum_batch(dx)≈0` 这一结构不变量。

### 40. 第一层 Linear 与 embedding scatter-add

embedding 的反向传播必须对重复字符索引做累加，使用 `index_add_`、`scatter_add_` 或等价逻辑，不能覆盖。

### 41. 完整手写 backward 与梯度审计

实现 `manual_backward()`，返回七个参数梯度 `C/W1/b1/W2/b2/bngain/bnbias`，以及
`dlogits/dh/dhpreact/dhprebn/demb` 等关键节点梯度。生成 `manual_grad_diff.png`，以对数尺度展示
manual 与独立 autograd oracle 的最大误差；exact 只作为信息项，shape、finite 与 allclose 才是硬门槛。

### 42. 无 autograd 训练闭环

实现 `manual_train_step()`；一步更新后的每个参数必须与 autograd 路径一致。学生训练路径不得通过
`enable_grad()` 偷跑 `.backward()`，完成后所有参数 `.grad` 仍应为 `None`。短程 smoke training 后实现
`calibrate_manual_batchnorm()` 与 `manual_inference_logits()`：校准集生成固定 BN 统计，单样本采样使用这些统计，
不能在 batch=1 上重新计算 `/ (n-1)` 方差。最终生成 `manual_training.png` 和 `manual_samples.txt`。

---

# 第五阶段：WaveNet-inspired 层次模型（43–52）

### 43. 8 字符上下文

复用数据构造逻辑，生成 `(N,8)` 的上下文。初始上下文为 8 个 `.`，每次预测后左移并追加字符。

### 44. 任意前导维度 Linear 与 Embedding

让 `Linear` 支持 `(..., fan_in) -> (..., fan_out)`，实现 `Embedding`。必须证明对时间轴的向量化 Linear 等价于显式 for-loop，这就是“convolution is a for loop”的核心直觉。

### 45. FlattenConsecutive

实现精确重排：

```text
(B,8,C) -> (B,4,2C) -> (B,2,4C) -> (B,8C)
```

必须用 `torch.arange` 验证相邻顺序；最后只允许 `squeeze(1)`，不能误删 batch size=1 的批次轴。

### 46. Sequential、模式传播与 shape trace

实现 `trace_shapes()` 和 `save_shape_table()`，记录每一层的 `index/name/input_shape/output_shape/parameter_count`，
逐层参数量之和必须等于模型总参数量，生成至少含上述列和每层记录的 `wavenet_shapes.csv`。

### 47. 8 字符扁平 MLP 基线

构建 `Embedding(27,10) -> Flatten(8) -> Linear(80,200,bias=False) -> BN -> Tanh
-> Linear(200,27,bias=True)`，参数量应为 22097。最终 Linear 的初始 weight 乘 0.1，避免一开始
对随机字符过度自信；checker 会核对层类型、顺序与 bias，而不只看总数。

### 48. 树状层次模型

实现通用 `build_hierarchical_model()`。`block_size` 必须是 `group_size` 的整数幂；每一级严格使用
`FlattenConsecutive(group) -> Linear(bias=False) -> BatchNorm -> Tanh`，最后使用带 bias 的输出 Linear，
并将其初始 weight 乘 0.1。通过可选 `batchnorm_factory(dim)` 构造每一级 BN，以便同一 builder 做错误轴/修复轴实验。
不能把三级结构硬编码为只支持 8/2；checker 还会测试 4/2、9/3 和非法配置。

官方 `block=8, group=2, embedding=24, hidden=128` 的完整层序与形状为：

```text
tokens                 (B,8)
Embedding(27,24)       (B,8,24)
FlattenConsecutive(2)  (B,4,48)
Linear(48,128,no bias) (B,4,128)
BatchNorm + Tanh       (B,4,128)
FlattenConsecutive(2)  (B,2,256)
Linear(256,128,no bias)(B,2,128)
BatchNorm + Tanh       (B,2,128)
FlattenConsecutive(2)  (B,256)
Linear(256,128,no bias)(B,128)
BatchNorm + Tanh       (B,128)
Linear(128,27,bias)    (B,27)
```

各部分参数量依次为 `648 + 6144 + 256 + 32768 + 256 + 32768 + 256 + 3483 = 76579`。

### 49. 感受野结构图

实现通用 `receptive_field_stages(block_size, group_size)`。默认 8/2 时各阶段每个节点覆盖宽度严格为
1、2、4、8，组内位置连续、不遗漏不重叠，最终节点覆盖全部过去 8 个字符；还要支持 9/3 等合法配置。
生成含节点和连接关系的 `wavenet_receptive_field.png`。

### 50. 修复 3D BatchNorm 轴问题

channel-last 三维张量 `(B,T,C)` 必须同时在 `(0,1)` 轴统计。训练时 running_mean/running_var
也必须来自这两个轴；eval 时 batch=1 可用固定统计推理且不更新 buffer。提供明确的“仅沿 batch 轴”故障模拟，
使实验 harness 能用同一配置重跑 bug/fixed 对照；生成 `wavenet_bn_axis_bug.png`。

### 51. 训练、平滑 loss、评估与采样

实现通用训练历史、不能整除窗口长度的平滑函数、eval 评估和 8 字符滚动采样。历史统一保存 raw CE
及对应 step；`plot_wave_loss(steps, losses, smoothed_steps, smoothed)` 必须给尾块平滑值使用其真实结束 step，
绘图时可再计算/标注 `log10(loss)`。训练历史还必须逐步记录每个参数的
update:data，而不是训练结束后补一个点。训练 API 要透传 `decay_at/decay_factor`，并把实际 schedule 写入配置。
`evaluate_model()` 和 `sample_names()` 各自都要独立切换到 eval（返回时模型保持 eval）；训练必须真的改变参数，
固定 generator 采样要可复现、字符合法、context 正确滚动、受 max length 约束，且返回字符串不含终止 `.`。
生成 `wavenet_loss.png` 和
`wavenet_samples.txt`。

### 52. 最终实验与全套内部诊断

实验集合名称固定为 `flat_context_3`、`flat_context_8`、`hierarchical_small_before_bn_fix`、
`hierarchical_small_bn_fixed`、`hierarchical_scaled`，分别对应 3 字符扁平、8 字符扁平、带错误轴的树状小模型、
修复 3D BN 的同配置模型、76,579 参数模型。五组必须通过同一 experiment harness 实际运行并保存配置、
schedule、参数量和 train/dev 指标。必须生成：

- `wavenet_model_comparison.png`
- `wavenet_experiments.json`
- `wavenet_diagnostics.json`
- `wavenet_activation_histograms.png`
- `wavenet_activation_gradient_histograms.png`
- `wavenet_update_ratios.png`

诊断必须在真正的层次模型上运行，覆盖前两层 3D Tanh 和最后一层 2D Tanh；反向前先清空训练残留梯度。
重新统计 activation mean/std、饱和率、activation gradient、parameter gradient，以及训练全过程的 update:data。
`WAVENET_REQUIRED_EXPERIMENTS` 和 `WAVENET_REQUIRED_ARTIFACTS` 是不可缩减的固定清单；
`wavenet_experiments.json` 必须恰好包含上述五组实验。最终再执行 `python check.py --artifacts`，缺任何固定工件都不算完成。

---

# 最终工件清单

`artifacts/` 中以下文件全部属于必做验收：

```text
bigram_name_lengths.png
bigram_counts.png
bigram_next_char.png
bigram_smoothing.png
bigram_model_diff.png
bigram_samples.txt
mlp_lr_range.png
mlp_training.png
mlp_capacity.png
mlp_experiments.json
mlp_embeddings.png
mlp_prediction_frequency.png
mlp_entropy.png
mlp_samples.txt
init_logits.png
tanh_histograms.png
tanh_saturation.png
variance_propagation.png
bn_batch_coupling.png
bn_folding.png
activation_histograms.png
activation_summary.png
activation_gradient_histograms.png
parameter_gradient_histograms.png
update_ratios.png
diagnostics_dashboard.png
zero_init_pathology.png
diagnostics.json
manual_dlogits.png
manual_grad_diff.png
manual_training.png
manual_samples.txt
wavenet_shapes.csv
wavenet_receptive_field.png
wavenet_bn_axis_bug.png
wavenet_loss.png
wavenet_samples.txt
wavenet_model_comparison.png
wavenet_experiments.json
wavenet_diagnostics.json
wavenet_activation_histograms.png
wavenet_activation_gradient_histograms.png
wavenet_update_ratios.png
```
