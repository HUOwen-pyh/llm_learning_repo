"""显式参数的字符 MLP，便于检查内部张量。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MLPConfig:
    vocab_size: int = 27
    block_size: int = 3
    n_embd: int = 10
    n_hidden: int = 200


@dataclass
class MLPParameters:
    C: torch.Tensor
    W1: torch.Tensor
    b1: torch.Tensor
    W2: torch.Tensor
    b2: torch.Tensor

    def tensors(self) -> list[torch.Tensor]:
        # TODO 14
        raise NotImplementedError("完成任务 14：MLPParameters.tensors")


@dataclass
class ForwardCache:
    emb: torch.Tensor
    embcat: torch.Tensor
    hpreact: torch.Tensor
    h: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class TensorLayout:
    """用于观察 Tensor view 的形状、stride 和连续性。"""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    is_contiguous: bool


def embedding_lookup(C: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # TODO 13
    raise NotImplementedError("完成任务 13：embedding_lookup")


def flatten_embeddings(emb: torch.Tensor) -> torch.Tensor:
    # TODO 13
    raise NotImplementedError("完成任务 13：flatten_embeddings")


def tensor_layout(tensor: torch.Tensor) -> TensorLayout:
    """读取 Tensor 布局，不复制或修改 Tensor。"""
    # TODO 13
    raise NotImplementedError("完成任务 13：tensor_layout")


def init_mlp(config: MLPConfig, generator: torch.Generator) -> MLPParameters:
    # TODO 14
    raise NotImplementedError("完成任务 14：init_mlp")


def clone_parameters(parameters: MLPParameters) -> MLPParameters:
    """复制数值和 storage，并创建 requires_grad 的新叶张量。

    clone 与原参数不得共享 storage，且每个 clone 都必须满足
    ``is_leaf``、``grad_fn is None`` 和 ``requires_grad``。
    """
    # TODO 14
    raise NotImplementedError("完成任务 14：clone_parameters")


def parameter_count(parameters: MLPParameters) -> int:
    # TODO 14
    raise NotImplementedError("完成任务 14：parameter_count")


def mlp_forward(
    parameters: MLPParameters,
    x: torch.Tensor,
    return_cache: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ForwardCache]:
    # TODO 15
    raise NotImplementedError("完成任务 15：mlp_forward")


def stable_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """手写稳定 CE；结果应与 F.cross_entropy 一致。"""
    # TODO 16
    raise NotImplementedError("完成任务 16：stable_cross_entropy")


@torch.no_grad()
def sample_mlp(
    parameters: MLPParameters,
    block_size: int,
    itos: dict[int, str],
    generator: torch.Generator,
    count: int = 10,
    max_length: int = 30,
) -> list[str]:
    """用长度为 ``block_size`` 的滚动上下文自回归采样。

    索引 0 是起止边界。返回值不包含边界字符；若没有采到边界，则在
    ``max_length`` 个字符后停止。
    """
    # TODO 20
    raise NotImplementedError("完成任务 20：sample_mlp")
