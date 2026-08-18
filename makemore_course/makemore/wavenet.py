"""Lecture 6：WaveNet-inspired 层次字符模型。"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .data import Vocabulary
from .layers import Sequential


ConfigValue = int | float | str | bool | None
BatchNormFactory = Callable[[int], object]


# 最终 demo 必须使用这两个固定清单，不能在调用 validate 时自行缩减。
WAVENET_REQUIRED_EXPERIMENTS: tuple[str, ...] = (
    "flat_context_3",
    "flat_context_8",
    "hierarchical_small_before_bn_fix",
    "hierarchical_small_bn_fixed",
    "hierarchical_scaled",
)

WAVENET_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "wavenet_shapes.csv",
    "wavenet_receptive_field.png",
    "wavenet_bn_axis_bug.png",
    "wavenet_loss.png",
    "wavenet_samples.txt",
    "wavenet_model_comparison.png",
    "wavenet_experiments.json",
    "wavenet_diagnostics.json",
    "wavenet_activation_histograms.png",
    "wavenet_activation_gradient_histograms.png",
    "wavenet_update_ratios.png",
)


@dataclass(frozen=True)
class WaveNetConfig:
    vocab_size: int = 27
    block_size: int = 8
    n_embd: int = 24
    n_hidden: int = 128
    group_size: int = 2
    output_scale: float = 0.1


@dataclass
class LayerTrace:
    index: int
    name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    parameter_count: int


@dataclass
class WaveTrainingHistory:
    """训练曲线及按固定间隔记录的每参数 update:data。"""

    steps: list[int]
    losses: list[float]
    learning_rates: list[float]
    update_steps: list[int]
    update_to_data_log10: dict[str, list[float | None]]


@dataclass
class WaveExperimentResult:
    name: str
    parameter_count: int
    train_loss: float
    dev_loss: float
    seconds: float
    config: dict[str, ConfigValue]
    history: WaveTrainingHistory


def model_parameter_count(model: Sequential) -> int:
    # TODO 47
    raise NotImplementedError("完成任务 47：model_parameter_count")


def build_flat_model(
    vocab_size: int,
    block_size: int,
    n_embd: int,
    n_hidden: int,
    generator: torch.Generator,
    output_scale: float = 0.1,
) -> Sequential:
    """构建 Embedding -> Flatten(all) -> Linear(no bias) -> BN -> Tanh -> Linear。"""
    # TODO 47
    raise NotImplementedError("完成任务 47：build_flat_model")


def build_hierarchical_model(
    config: WaveNetConfig,
    generator: torch.Generator,
    batchnorm_factory: BatchNormFactory | None = None,
) -> Sequential:
    """每阶段依次使用 FlattenConsecutive、无 bias Linear、BN、Tanh。"""
    # TODO 48：根据 block_size/group_size 自动生成，不硬编码三层。
    # batchnorm_factory 让最终实验能用同一 builder 对照错误轴版本与修复版本。
    raise NotImplementedError("完成任务 48：build_hierarchical_model")


def trace_shapes(model: Sequential, x: torch.Tensor) -> list[LayerTrace]:
    # TODO 46
    raise NotImplementedError("完成任务 46：trace_shapes")


def receptive_field_stages(
    block_size: int = 8,
    group_size: int = 2,
) -> list[list[tuple[int, ...]]]:
    # TODO 49
    raise NotImplementedError("完成任务 49：receptive_field_stages")


def train_model(
    model: Sequential,
    x: torch.Tensor,
    y: torch.Tensor,
    generator: torch.Generator,
    steps: int,
    batch_size: int,
    learning_rate: float,
    decay_at: int | None = None,
    decay_factor: float = 0.1,
    record_update_every: int = 1,
) -> WaveTrainingHistory:
    # TODO 51：losses 保存原始 CE；learning_rates 每步记录实际 lr。
    # 当 step % record_update_every == 0 时，以 p0、p1……为稳定名称记录
    # log10(std(lr * grad) / std(parameter))；零方差项显式记录 None。
    # TODO 51
    raise NotImplementedError("完成任务 51：train_model")


@torch.no_grad()
def evaluate_model(model: Sequential, x: torch.Tensor, y: torch.Tensor) -> float:
    # TODO 51
    raise NotImplementedError("完成任务 51：evaluate_model")


@torch.no_grad()
def sample_names(
    model: Sequential,
    vocab: Vocabulary,
    block_size: int,
    generator: torch.Generator,
    count: int = 10,
    max_length: int = 30,
) -> list[str]:
    """滚动固定长度上下文采样；返回字符串不包含索引 0 的边界字符。"""
    # TODO 51
    raise NotImplementedError("完成任务 51：sample_names")


def run_wave_experiment(
    name: str,
    model: Sequential,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_dev: torch.Tensor,
    y_dev: torch.Tensor,
    generator: torch.Generator,
    steps: int,
    batch_size: int,
    learning_rate: float,
    config: dict[str, ConfigValue],
    decay_at: int | None = None,
    decay_factor: float = 0.1,
    record_update_every: int = 100,
) -> WaveExperimentResult:
    # TODO 52：必须把实际 learning-rate schedule 写入 result.config，
    # 并把 train_model 返回的完整 history 放入 result.history。
    raise NotImplementedError("完成任务 52：run_wave_experiment")


def missing_required_wave_experiments(
    results: Sequence[WaveExperimentResult],
) -> list[str]:
    """按固定顺序返回最终实验套件中缺失的实验名。"""
    # TODO 52
    raise NotImplementedError("完成任务 52：missing_required_wave_experiments")


def validate_wave_artifacts(directory: str | Path) -> list[str]:
    """使用不可由 demo 调用方缩减的 WaveNet 工件清单。"""
    from .reporting import validate_artifacts

    return validate_artifacts(directory, WAVENET_REQUIRED_ARTIFACTS)
