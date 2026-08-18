"""损失函数与梯度下降训练。"""

from __future__ import annotations

from collections.abc import Sequence

from .engine import Value
from .nn import Module


def squared_error(
    predictions: Sequence[Value],
    targets: Sequence[float],
) -> Value:
    loss = sum([(p-t)**2 for p, t in zip(predictions, targets)])
    return loss


def forward_loss(
    model: Module,
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
) -> tuple[list[Value], Value]:
    y = []
    for x in xs:
        tmp = model(x)
        pred = tmp if isinstance(tmp, list) else [tmp]
        y += pred
    return (y, squared_error(y, ys))


def train_step(
    model: Module,
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    learning_rate: float,
) -> float:
    _, loss = forward_loss(model, xs, ys)
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    return loss.data


def fit(
    model: Module,
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    steps: int = 100,
    learning_rate: float = 0.05,
) -> list[float]:
    losses = []
    for _ in range(steps):
        losses.append(train_step(model, xs, ys, learning_rate))

    return losses

