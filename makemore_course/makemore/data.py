"""语料、词表与上下文数据集。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


@dataclass(frozen=True)
class Vocabulary:
    stoi: dict[str, int]
    itos: dict[int, str]
    boundary: str = "."

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]
    
    def decode(self, indices: Sequence[int]) -> str:
        return "".join([self.itos[i] for i in indices])

    @property
    def size(self) -> int:
        return len(self.stoi)


def load_words(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [line.strip().lower() for line in file if line.strip()]


def corpus_stats(words: Sequence[str]) -> dict[str, int | float | list[str]]:
    lens = [len(s) for s in words]
    return {
        "count": len(lens),
        "max_length": max(lens),
        "min_length": min(lens),
        "mean_length": sum(lens) / len(lens),
        "characters": sorted(set("".join(words)))
    }


def build_vocab(words: Sequence[str], boundary: str = ".") -> Vocabulary:
    vl = [boundary] + sorted(set("".join(words)))
    stoi, itos = {}, {}

    for i, s in enumerate(vl):
        stoi[s] = i
        itos[i] = s

    return Vocabulary(stoi, itos)


def split_words(
    words: Sequence[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    """先按单词打乱，再切分 train/dev/test。

    契约：不得修改传入序列或全局 ``random`` 状态；使用独立的
    ``random.Random(seed)``。切分点与课程一致：
    ``n1=int(n*train_ratio)``、``n2=int(n*(train_ratio+dev_ratio))``；
    三段依次为 ``[:n1]``、``[n1:n2]``、``[n2:]``。
    ``train_ratio``、``dev_ratio`` 必须非负且两者之和小于 1。
    """
    # TODO 11
    raise NotImplementedError("完成任务 11：split_words")


def build_bigram_pairs(
    words: Sequence[str],
    vocab: Vocabulary,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把每个单词连同首尾边界转换成 bigram x/y。"""
    # TODO 03
    raise NotImplementedError("完成任务 03：build_bigram_pairs")


def build_context_dataset(
    words: Sequence[str],
    vocab: Vocabulary,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """为每个目标字符构建固定长度的左侧上下文。

    每个单词独立地从 ``block_size`` 个边界索引开始；只在单词内部滑动
    context。``block_size`` 必须至少为 1，返回 long Tensor。
    """
    # TODO 12：实现通用上下文窗口。
    # TODO 43：确认同一实现正确支持 block_size=8。
    raise NotImplementedError("完成任务 12/43：build_context_dataset")
