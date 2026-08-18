"""课程风格的轻量层，不继承 torch.nn.Module。"""

from __future__ import annotations

from typing import Sequence

import torch


class Linear:
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        generator: torch.Generator,
        bias: bool = True,
        gain: float = 1.0,
    ) -> None:
        # TODO 28/29；任务 44 会正式验证任意前导维度。
        raise NotImplementedError("完成任务 28/29/44：Linear.__init__")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 28/29/44
        raise NotImplementedError("完成任务 28/29/44：Linear.__call__")

    def parameters(self) -> list[torch.Tensor]:
        # TODO 28/29
        raise NotImplementedError("完成任务 28/29：Linear.parameters")


class BatchNorm1d:
    """Channel-last BatchNorm used throughout the course.

    ``gamma``, ``beta``, ``running_mean`` and ``running_var`` are all
    one-dimensional tensors of shape ``(dim,)``.  This single convention
    broadcasts over both ``(B, C)`` and the later ``(B, T, C)`` inputs.
    Training uses the unbiased/sample variance (``correction=1``).
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        # TODO 25：四个逐通道张量均使用 shape (dim,)，不要混用 (1, dim)。
        raise NotImplementedError("完成任务 25：BatchNorm1d.__init__")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 25/26：实现 2D 训练/推理路径。
        # TODO 50：扩展到 (B,T,C)，统计轴为 (0,1)。
        raise NotImplementedError("完成任务 25/26/50：BatchNorm1d.__call__")

    def parameters(self) -> list[torch.Tensor]:
        # TODO 25
        raise NotImplementedError("完成任务 25：BatchNorm1d.parameters")

    def train(self, mode: bool = True) -> BatchNorm1d:
        # TODO 26
        raise NotImplementedError("完成任务 26：BatchNorm1d.train")

    def eval(self) -> BatchNorm1d:
        # TODO 26
        raise NotImplementedError("完成任务 26：BatchNorm1d.eval")


class Tanh:
    def __init__(self) -> None:
        self.out: torch.Tensor | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 29
        raise NotImplementedError("完成任务 29：Tanh.__call__")

    def parameters(self) -> list[torch.Tensor]:
        return []


class Embedding:
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        generator: torch.Generator,
    ) -> None:
        # TODO 44
        raise NotImplementedError("完成任务 44：Embedding.__init__")

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        # TODO 44
        raise NotImplementedError("完成任务 44：Embedding.__call__")

    def parameters(self) -> list[torch.Tensor]:
        # TODO 44
        raise NotImplementedError("完成任务 44：Embedding.parameters")


class FlattenConsecutive:
    def __init__(self, n: int) -> None:
        self.n = n
        self.out: torch.Tensor | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 45
        raise NotImplementedError("完成任务 45：FlattenConsecutive.__call__")

    def parameters(self) -> list[torch.Tensor]:
        return []


class Sequential:
    def __init__(self, layers: Sequence[object]) -> None:
        self.layers = list(layers)
        self.out: torch.Tensor | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 29
        raise NotImplementedError("完成任务 29：Sequential.__call__")

    def parameters(self) -> list[torch.Tensor]:
        # TODO 29
        raise NotImplementedError("完成任务 29：Sequential.parameters")

    def train(self, mode: bool = True) -> Sequential:
        # TODO 29/46：向所有支持 train() 的子层传播模式。
        raise NotImplementedError("完成任务 29/46：Sequential.train")

    def eval(self) -> Sequential:
        # TODO 29/46
        raise NotImplementedError("完成任务 29/46：Sequential.eval")


class DeepMLP:
    def __init__(self, embedding: torch.Tensor, network: Sequential, block_size: int) -> None:
        self.embedding = embedding
        self.network = network
        self.block_size = block_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 29
        raise NotImplementedError("完成任务 29：DeepMLP.__call__")

    def parameters(self) -> list[torch.Tensor]:
        # TODO 29
        raise NotImplementedError("完成任务 29：DeepMLP.parameters")

    def train(self, mode: bool = True) -> DeepMLP:
        # TODO 29
        raise NotImplementedError("完成任务 29：DeepMLP.train")

    def eval(self) -> DeepMLP:
        # TODO 29
        raise NotImplementedError("完成任务 29：DeepMLP.eval")


def build_deep_mlp(
    vocab_size: int,
    block_size: int,
    n_embd: int,
    n_hidden: int,
    n_hidden_layers: int,
    generator: torch.Generator,
) -> DeepMLP:
    """构建 Linear(no bias)-BN-Tanh 堆叠和最后 Linear-BN。"""
    # TODO 29
    raise NotImplementedError("完成任务 29：build_deep_mlp")
