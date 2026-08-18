"""Lecture 5：张量级手写反向传播。"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class ManualParameters:
    C: torch.Tensor
    W1: torch.Tensor
    b1: torch.Tensor
    W2: torch.Tensor
    b2: torch.Tensor
    bngain: torch.Tensor
    bnbias: torch.Tensor

    def tensors(self) -> list[torch.Tensor]:
        return [getattr(self, field.name) for field in fields(self)]


@dataclass(frozen=True)
class ManualBatchNormStats:
    """训练后用于手写 MLP 推理的固定 BatchNorm 统计。"""

    mean: torch.Tensor
    var: torch.Tensor


@dataclass
class ManualForwardCache:
    emb: torch.Tensor
    embcat: torch.Tensor
    hprebn: torch.Tensor
    bnmean: torch.Tensor
    bndiff: torch.Tensor
    bndiff2: torch.Tensor
    bnvar: torch.Tensor
    bnvar_inv: torch.Tensor
    bnraw: torch.Tensor
    hpreact: torch.Tensor
    h: torch.Tensor
    logits: torch.Tensor
    logit_maxes: torch.Tensor
    norm_logits: torch.Tensor
    counts: torch.Tensor
    counts_sum: torch.Tensor
    counts_sum_inv: torch.Tensor
    probs: torch.Tensor
    logprobs: torch.Tensor
    loss: torch.Tensor


@dataclass
class GradComparison:
    name: str
    shape_matches: bool
    exact: bool
    allclose: bool
    max_difference: float


def init_manual_parameters(
    vocab_size: int,
    block_size: int,
    n_embd: int,
    n_hidden: int,
    generator: torch.Generator,
) -> ManualParameters:
    # TODO 34：按官方 debug 初始化保留非零小随机 bias/gain/beta，避免零值掩盖错误。
    # bngain/bnbias 的 shape 固定为 (1, n_hidden)。
    raise NotImplementedError("完成任务 34：init_manual_parameters")


def manual_forward(
    parameters: ManualParameters,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-5,
) -> ManualForwardCache:
    # TODO 34：CE 必须显式减 row max，BN 使用样本方差 /(n-1)。
    # 为 checker 的 autograd oracle，对 requires_grad=True 的非叶 cache Tensor 调用 retain_grad()。
    # manual_train_step 位于 no_grad 环境，此时不要调用 retain_grad()。
    raise NotImplementedError("完成任务 34：manual_forward")


@torch.no_grad()
def calibrate_manual_batchnorm(
    parameters: ManualParameters,
    x: torch.Tensor,
) -> ManualBatchNormStats:
    """在完整训练输入上计算第一层 pre-BN 的 mean/sample variance。"""
    # TODO 42：x 的 batch 必须至少包含两个样本；统计 shape 为 (1, n_hidden)。
    raise NotImplementedError("完成任务 42：calibrate_manual_batchnorm")


@torch.no_grad()
def manual_inference_logits(
    parameters: ManualParameters,
    x: torch.Tensor,
    stats: ManualBatchNormStats,
    eps: float = 1e-5,
) -> torch.Tensor:
    """使用固定 BN statistics 前向；必须安全支持 batch size 1。"""
    # TODO 42：这里只返回 logits，不需要伪造 targets/loss。
    raise NotImplementedError("完成任务 42：manual_inference_logits")


def compare_grad(
    name: str,
    manual: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> GradComparison:
    # TODO 34：先强制检查 shape，再检查 exact/allclose/maxdiff。
    raise NotImplementedError("完成任务 34：compare_grad")


def backward_cross_entropy_atomic(
    cache: ManualForwardCache,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # TODO 35：返回 dlogprobs/dprobs/dcounts/dcounts_sum/
    # dcounts_sum_inv/dnorm_logits/dlogit_maxes/dlogits。
    raise NotImplementedError("完成任务 35：backward_cross_entropy_atomic")


def backward_cross_entropy_fused(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    # TODO 36
    raise NotImplementedError("完成任务 36：backward_cross_entropy_fused")


def linear_backward(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO 37
    raise NotImplementedError("完成任务 37：linear_backward")


def tanh_backward(dout: torch.Tensor, tanh_output: torch.Tensor) -> torch.Tensor:
    # TODO 37
    raise NotImplementedError("完成任务 37：tanh_backward")


def batchnorm_backward_atomic(
    dout: torch.Tensor,
    cache: ManualForwardCache,
    gamma: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # TODO 38：固定返回 dhprebn/dbngain/dbnbias/dbnraw/dbnvar_inv/
    # dbnvar/dbndiff2/dbndiff/dbnmean；每个 shape 必须与对应 forward Tensor 相同。
    raise NotImplementedError("完成任务 38：batchnorm_backward_atomic")


def batchnorm_backward_fused(
    dout: torch.Tensor,
    xhat: torch.Tensor,
    gamma: torch.Tensor,
    invstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO 39：与任务 38 相同的 unbiased variance 约定。
    raise NotImplementedError("完成任务 39：batchnorm_backward_fused")


def embedding_backward(
    demb: torch.Tensor,
    indices: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    # TODO 40：重复索引必须 scatter-add。
    raise NotImplementedError("完成任务 40：embedding_backward")


def manual_backward(
    parameters: ManualParameters,
    cache: ManualForwardCache,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # TODO 41：参数键固定为 C/W1/b1/W2/b2/bngain/bnbias；另返回
    # dlogits/dh/dhpreact/dhprebn/dembcat/demb，供逐节点审计。
    raise NotImplementedError("完成任务 41：manual_backward")


@torch.no_grad()
def manual_train_step(
    parameters: ManualParameters,
    x: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float,
) -> float:
    # TODO 42：禁止调用 loss.backward()/autograd.grad。
    raise NotImplementedError("完成任务 42：manual_train_step")
