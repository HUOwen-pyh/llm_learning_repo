"""计数 Bigram、神经 Bigram 与采样。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .data import Vocabulary, build_bigram_pairs


@dataclass
class BigramHistory:
    steps: list[int]
    losses: list[float]


@dataclass
class SmoothingResult:
    strengths: list[float]
    train_nll: list[float]
    dev_nll: list[float]


def count_bigrams(words: Sequence[str], vocab: Vocabulary) -> torch.Tensor:
    x, y = build_bigram_pairs(words, vocab)
    l = len(vocab.stoi)
    res = torch.zeros(l, l, dtype=torch.long)
    for i in range(x.shape[0]):
        res[x[i], y[i]] += 1
    return res


def normalize_counts(counts: torch.Tensor, smoothing: float = 1.0) -> torch.Tensor:
    t = torch.zeros_like(counts, dtype=torch.float)
    l = counts.shape[0]
    for i in range(l):
        s = counts[i].sum() + l * smoothing
        for j in range(l):
            t[i, j] = (counts[i, j] + smoothing) / s

    return t


def bigram_nll(probs: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """返回平均负对数似然。"""
    # TODO 06
    raise NotImplementedError("完成任务 06：bigram_nll")


def perplexity(nll: float | torch.Tensor) -> float:
    # TODO 06
    raise NotImplementedError("完成任务 06：perplexity")


@torch.no_grad()
def sample_bigram(
    probs: torch.Tensor,
    vocab: Vocabulary,
    generator: torch.Generator,
    count: int = 10,
    max_length: int = 30,
) -> list[str]:
    """从边界索引 0 开始采样。

    返回字符串不包含终止边界；若始终采不到边界，每个结果必须恰好在
    ``max_length`` 个普通字符后停止。
    """
    # TODO 07
    raise NotImplementedError("完成任务 07：sample_bigram")


def neural_bigram_logits(
    weights: torch.Tensor,
    x: torch.Tensor,
    use_one_hot: bool = True,
) -> torch.Tensor:
    """支持 one_hot@W 和 W[x] 两种等价路径。"""
    # TODO 09
    raise NotImplementedError("完成任务 09：neural_bigram_logits")


def manual_softmax(logits: torch.Tensor) -> torch.Tensor:
    """沿最后一维计算数值稳定的 softmax（先减去逐行最大值）。"""
    # TODO 09
    raise NotImplementedError("完成任务 09：manual_softmax")


def neural_bigram_loss(
    weights: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    regularization: float = 0.0,
) -> torch.Tensor:
    # TODO 09
    raise NotImplementedError("完成任务 09：neural_bigram_loss")


def train_neural_bigram(
    weights: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    learning_rate: float,
    regularization: float = 0.0,
) -> BigramHistory:
    """执行指定次数的全批量 SGD。

    ``history.steps`` 为从 1 开始的已完成更新次数；对应 loss 是该次更新
    之前的目标值。更新方向必须是 ``W -= learning_rate * W.grad``。
    """
    # TODO 10
    raise NotImplementedError("完成任务 10：train_neural_bigram")


def count_trigrams(
    words: Sequence[str],
    vocab: Vocabulary,
) -> torch.Tensor:
    """返回 shape=(V,V,V) 的 trigram 计数。

    前两维是两个上下文字符，最后一维是目标字符。每个单词从
    ``('.', '.')`` 上下文开始，并为最后的边界字符建立一个训练样本。
    """
    # TODO 10
    raise NotImplementedError("完成任务 10：count_trigrams")


def normalize_trigram_counts(
    counts: torch.Tensor,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """加伪计数后，仅沿目标字符轴（最后一维）归一化 3D 计数。"""
    # TODO 10
    raise NotImplementedError("完成任务 10：normalize_trigram_counts")


def trigram_nll(
    probs: torch.Tensor,
    words: Sequence[str],
    vocab: Vocabulary,
) -> torch.Tensor:
    """在单词序列上计算 trigram 模型的平均 NLL。"""
    # TODO 10
    raise NotImplementedError("完成任务 10：trigram_nll")


def smoothing_search(
    train_words: Sequence[str],
    dev_words: Sequence[str],
    vocab: Vocabulary,
    strengths: Sequence[float],
) -> SmoothingResult:
    """只用 ``train_words`` 拟合 trigram 计数并扫描伪计数强度。

    每个强度都必须分别返回 train/dev 平均 NLL；dev 数据不得加入计数，
    本函数也不得查看 test 数据。
    """
    # TODO 10
    raise NotImplementedError("完成任务 10：smoothing_search")


def probability_difference(
    count_probs: torch.Tensor,
    neural_probs: torch.Tensor,
) -> torch.Tensor:
    """返回两种模型条件概率的绝对差。"""
    # TODO 10
    raise NotImplementedError("完成任务 10：probability_difference")
