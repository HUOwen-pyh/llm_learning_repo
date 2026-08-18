# 任务清单

每个任务只引入一个核心概念。命令均在 `micrograd_course` 目录中运行。

| 任务 | 小目标 | 主要文件 | 验收命令 |
|---:|---|---|---|
| 01 | 用有限差分理解导数 | `micrograd/engine.py` | `python check.py 1` |
| 02 | 完成最小 `Value` 表示 | `micrograd/engine.py` | `python check.py 2` |
| 03 | 实现加法、乘法和计算图关系 | `micrograd/engine.py` | `python check.py 3` |
| 04 | 遍历计算图的节点与边 | `micrograd/viz.py` | `python check.py 4` |
| 05 | 手算 `a*b+c` 的反向传播 | `micrograd/engine.py` | `python check.py 5` |
| 06 | 为加法和乘法保存局部反向函数 | `micrograd/engine.py` | `python check.py 6` |
| 07 | 实现 `tanh` 及其局部梯度 | `micrograd/engine.py` | `python check.py 7` |
| 08 | 用拓扑排序实现完整 `backward()` | `micrograd/engine.py` | `python check.py 8` |
| 09 | 修复共享节点的梯度累积 | `micrograd/engine.py` | `python check.py 9` |
| 10 | 补齐幂、指数、减法、除法等运算 | `micrograd/engine.py` | `python check.py 10` |
| 11 | 用有限差分检查自动梯度 | `micrograd/engine.py` | `python check.py 11` |
| 12 | 实现单个 `Neuron` | `micrograd/nn.py` | `python check.py 12` |
| 13 | 实现一层 `Layer` | `micrograd/nn.py` | `python check.py 13` |
| 14 | 实现多层感知器 `MLP` | `micrograd/nn.py` | `python check.py 14` |
| 15 | 实现平方误差 loss | `micrograd/training.py` | `python check.py 15` |
| 16 | 完成一次训练步骤 | `micrograd/training.py` | `python check.py 16` |
| 17 | 把训练步骤组成循环 | `micrograd/training.py` | `python check.py 17` |
| 18 | 训练课程中的完整 MLP | 全部 | `python check.py all` |

## 第一阶段：标量自动微分

### 01. 数值导数

实现 `numerical_derivative(fn, x, h)`。先用
`(fn(x+h)-fn(x))/h` 建立直觉，也可以使用更准确的中心差分。

验收：函数 `3*x**2 - 4*x + 5` 在 `x=3` 处的导数应接近 `14`。

### 02. 最小 Value

阅读已经给出的 `Value.__init__`，实现 `__repr__`。确认一个节点同时保存：

- `data`：前向计算的标量；
- `grad`：最终输出对该节点的导数；
- `_prev`：生成它的直接输入节点；
- `_op`：生成它的运算；
- `_backward`：该运算的局部反向函数。

### 03. 前向运算与计算图

实现 `Value.__add__` 和 `Value.__mul__` 的前向部分。结果节点必须记录两个输入节点和运算符。

验收：`Value(2) * Value(-3) + Value(10)` 的结果为 `4`，并且可以沿 `_prev` 找回完整表达式。

### 04. 遍历计算图

实现 `trace(root)`，返回所有节点和有向边。使用集合避免同一节点被重复访问。

验收：`a*b+c` 应有 5 个值节点、4 条依赖边。

### 05. 手算反向传播

完成 `manual_backprop_ab_plus_c`，不要调用 `backward()`。从输出梯度 `1` 开始，手动填写中间节点和叶节点的梯度。

验收：当 `a=2, b=-3, c=10` 时，三个输入的梯度分别为 `-3、2、1`。

### 06. 局部 backward

回到 `__add__` 和 `__mul__`，为结果节点绑定 `_backward` 闭包。闭包读取结果节点的上游梯度，并按链式法则传给父节点。

这一关只测试无分支的图；第 09 关会专门检查梯度累积。

### 07. tanh

实现 `Value.tanh()` 的前向值和局部反向函数。局部导数可以由输出 `t` 写成 `1 - t**2`。

### 08. 自动 backward

实现 `Value.backward()`：

1. 从当前输出节点深度优先遍历；
2. 生成父节点在前、输出节点在后的拓扑序；
3. 将输出梯度设为 `1`；
4. 逆序调用每个节点的 `_backward()`。

### 09. 梯度累积

测试 `y=x+x` 和 `y=x*x+x`。一个节点通过多条路径影响输出时，各条路径的梯度贡献必须相加，不能互相覆盖。

验收：`x=3, y=x*x+x` 时 `x.grad == 7`。

### 10. 完整标量运算

实现反向加/乘、取负、减法、幂、除法和 `exp()`。尽量用已经实现的基本运算组合出复合运算。

验收表达式：

```python
a = Value(3.0)
y = (2 + a) * (a - 1) / 2 + a**2
```

应得到 `y.data == 14`、`a.grad == 9.5`。

### 11. 自动梯度检查

实现 `gradient_check(fn, inputs)`：一边运行 `backward()` 获得解析梯度，一边逐个扰动输入获得数值梯度，返回两者的配对结果。

验收：复杂表达式中每个输入的两种梯度误差小于 `1e-4`。

## 第二阶段：搭建神经网络

### 12. Neuron

实现权重、偏置、前向计算和 `parameters()`：

```text
activation = w·x + b
output = tanh(activation)
```

`Neuron(nin)` 必须恰好有 `nin + 1` 个参数。
为保证练习结果可复现，权重使用 `random.uniform(-1, 1)`，偏置初始化为 `Value(0.0)`。

### 13. Layer

一层是若干个互不共享参数的神经元。`Layer(3, 4)` 接收 3 个输入并返回 4 个输出，总参数量为 `4*(3+1)=16`，而且这 16 个参数必须是不同的对象。

遵循视频约定：只有一个输出时返回单个 `Value`，否则返回列表。

### 14. MLP

将多层首尾相连。`MLP(3, [4, 4, 1])` 的层宽为 `3 -> 4 -> 4 -> 1`，总参数数目应为 `41`。
还要处理“一节点中间层”：当前一层返回单个 `Value` 而后面仍有层时，先把它重新包装成单元素序列；例如 `MLP(3, [1, 2])` 仍应正常运行。

## 第三阶段：训练

### 15. 平方误差

实现所有样本平方误差之和：

```text
loss = sum((prediction - target) ** 2)
```

不要读取 `.data` 来计算 loss，否则会切断计算图。

### 16. 一次训练步骤

完成 `train_step`，严格按以下顺序执行：

1. 前向计算和 loss；
2. 将所有参数梯度清零；
3. `loss.backward()`；
4. `parameter.data += -learning_rate * parameter.grad`。

### 17. 训练循环

实现 `fit`，重复调用训练步骤并返回每一步的 loss。先用一个神经元和一个样本确认训练后的 loss 相比初始值显著下降。

### 18. 最终完整任务

使用课程数据训练 `MLP(3, [4, 4, 1])`：

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]
```

最终验收：loss 显著下降，4 个预测值的正负号与标签一致。完成后运行 `python demo_train.py` 查看训练过程。
