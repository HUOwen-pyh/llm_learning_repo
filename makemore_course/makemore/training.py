"""MLP 的 minibatch、学习率扫描、训练和评估。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .mlp import MLPParameters, clone_parameters, mlp_forward


@dataclass
class TrainingHistory:
    # 所有横坐标都是“已经完成的参数更新次数”，从 1 开始。
    steps: list[int]
    losses: list[float]
    smoothed_steps: list[int]
    smoothed_losses: list[float]
    evaluation_steps: list[int]
    train_losses: list[float]
    dev_losses: list[float]


@dataclass
class LRRangeResult:
    learning_rates: list[float]
    log10_learning_rates: list[float]
    losses: list[float]


@dataclass
class ExperimentResult:
    name: str
    parameter_count: int
    train_loss: float
    dev_loss: float
    test_loss: float | None
    config: dict[str, int | float | str]


def moving_average(values: Sequence[float], window: int) -> list[float]:
    """按不重叠窗口求均值，最后不足 ``window`` 的尾块也必须保留。"""
    # TODO 19；任务 51 要求处理尾部不足一个窗口的情况。
    raise NotImplementedError("完成任务 19/51：moving_average")


def sample_minibatch(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO 17
    raise NotImplementedError("完成任务 17：sample_minibatch")


def mlp_train_step(
    parameters: MLPParameters,
    x: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float,
) -> float:
    """执行一次 SGD，并返回更新前的 minibatch CE。

    更新必须严格为 ``parameter -= learning_rate * parameter.grad``；函数
    返回时保留本轮梯度，便于教学检查。
    """
    # TODO 17
    raise NotImplementedError("完成任务 17：mlp_train_step")


@torch.no_grad()
def evaluate_mlp(parameters: MLPParameters, x: torch.Tensor, y: torch.Tensor) -> float:
    # TODO 19
    raise NotImplementedError("完成任务 19：evaluate_mlp")


def lr_range_test(
    parameters: MLPParameters,
    x: torch.Tensor,
    y: torch.Tensor,
    generator: torch.Generator,
    batch_size: int = 32,
    steps: int = 100,
    min_exponent: float = -3.0,
    max_exponent: float = 0.0,
) -> LRRangeResult:
    """在独立参数副本上做连续 LR range test，不得修改传入参数。

    使用包含两端点的等距 log10 学习率，共 ``steps`` 个；每一步抽取一个
    minibatch、记录更新前 loss，再以当前学习率更新同一个实验副本。
    """
    # TODO 18
    raise NotImplementedError("完成任务 18：lr_range_test")


def fit_mlp(
    parameters: MLPParameters,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_dev: torch.Tensor,
    y_dev: torch.Tensor,
    generator: torch.Generator,
    steps: int,
    batch_size: int,
    learning_rate: float,
    decay_at: int | None = None,
    decay_factor: float = 0.1,
    report_every: int = 100,
) -> TrainingHistory:
    """训练并返回所有曲线及各自明确的横坐标。

    ``steps`` 必须是 ``1..steps``。每次完成更新后，在更新次数能被
    ``report_every`` 整除时评估；最终一步即使不能整除也必须评估。
    ``evaluation_steps/train_losses/dev_losses`` 三者严格等长。
    ``smoothed_steps`` 是每个不重叠平滑窗口右端对应的更新次数。
    """
    # TODO 19
    raise NotImplementedError("完成任务 19：fit_mlp")


def character_frequencies(
    indices: torch.Tensor,
    vocab_size: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """返回总和为 1 的经验频率。

    若提供 ``ignore_index``，该类先被排除，再在剩余字符上归一化；这可
    保证真实与生成字符频率都按同一口径排除边界符。若过滤后为空，应
    返回全零向量。
    """
    # TODO 20
    raise NotImplementedError("完成任务 20：character_frequencies")


def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    # TODO 20
    raise NotImplementedError("完成任务 20：prediction_entropy")
