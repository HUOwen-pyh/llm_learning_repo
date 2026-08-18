# 从零实现 micrograd

这是第一课《The spelled-out intro to neural networks and backpropagation》的动手框架。
你会在同一套代码上依次完成 18 个小任务，最终得到：

- 一个标量自动微分引擎；
- `Neuron -> Layer -> MLP` 神经网络模块；
- 损失函数、梯度清零和梯度下降；
- 一个能拟合课程中 4 条样本的完整 MLP。

## 开始

要求 Python 3.10 或更新版本，不需要安装第三方库。

```powershell
cd micrograd_course
python check.py --list
python check.py 1
```

第一次运行会显示 `[待完成] 01` 并返回非零退出码，这是预期行为；完成对应
`TODO` 后才会显示 `[通过]`。

完成第 1 关后运行：

```powershell
python check.py 2
```

`check.py N` 会检查第 1 关到第 N 关，因此前面的功能不会被后面的修改破坏。

完成全部任务后运行：

```powershell
python check.py all
python demo_train.py
```

## 文件结构

```text
micrograd_course/
├─ TASKS.md                 # 18 个任务及验收标准
├─ check.py                 # 分关自动检查
├─ demo_train.py            # 最终训练入口
└─ micrograd/
   ├─ __init__.py
   ├─ engine.py             # Value、反向传播、梯度检查
   ├─ nn.py                 # Neuron、Layer、MLP
   ├─ training.py           # loss 和训练循环
   └─ viz.py                # 计算图遍历
```

建议每次只做一个任务：先读 [TASKS.md](TASKS.md) 中的目标，再搜索对应的
`TODO NN`，修改后运行该关检查。

## 最终要理解的训练循环

```text
前向计算 -> 计算 loss -> 清空旧梯度 -> backward -> 更新参数 -> 重复
```

框架故意没有提供关键实现答案；所有未完成位置都用 `TODO` 和
`NotImplementedError` 标出。
