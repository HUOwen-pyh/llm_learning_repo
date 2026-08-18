"""使用 Value 搭建 Neuron、Layer 和 MLP。"""

from __future__ import annotations

import random
from collections.abc import Sequence

from .engine import Value


ScalarInput = Value | int | float


class Module:
    """所有神经网络模块的共同接口。"""

    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        """清空参数梯度；任务 16 会验证这个方法。"""
        for parameter in self.parameters():
            parameter.grad = 0.0


class Neuron(Module):
    def __init__(self, nin: int) -> None:
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x: Sequence[ScalarInput]) -> Value:
        v = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return v.tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: Sequence[ScalarInput]) -> Value | list[Value]:
        out = [n(x) for n in self.neurons]
        return out if len(self.neurons) != 1 else out[0]

    def parameters(self) -> list[Value]:
        out = [x for n in self.neurons for x in n.parameters()]
        return out


class MLP(Module):
    def __init__(self, nin: int, nouts: Sequence[int]) -> None:
        nums = [nin] + nouts
        self.layers = [Layer(nums[i], nums[i+1]) for i in range(len(nouts))]

    def __call__(self, x: Sequence[ScalarInput]) -> Value | list[Value]:
        out = x
        for layer in self.layers:
            tmp = layer(out)
            out = tmp if isinstance(tmp, list) else [tmp]
        return out if len(self.layers[-1].neurons) != 1 else out[0]

    def parameters(self) -> list[Value]:
        out = [x for layer in self.layers for x in layer.parameters()]
        return out
