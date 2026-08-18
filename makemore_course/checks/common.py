from __future__ import annotations

import math
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float = 1e-6, message: str = "") -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(message or f"期望 {expected}，实际 {actual}")


def finite_tensor(tensor, message: str = "张量包含 NaN/Inf") -> None:
    import torch
    require(bool(torch.isfinite(tensor).all()), message)


def figure_has_content(result, minimum_axes: int = 1, every_axis: bool = True) -> None:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    require(
        isinstance(result, tuple) and len(result) == 2,
        "绘图函数必须严格返回 (Figure, axes)",
    )
    fig, returned_axes = result
    require(isinstance(fig, Figure), "返回值第一项必须是 Figure")
    require(len(fig.axes) >= minimum_axes, f"图中至少需要 {minimum_axes} 个 axes")

    def flatten_axes(value):
        if isinstance(value, Axes):
            return [value]
        if hasattr(value, "flat"):
            return [item for item in value.flat]
        if isinstance(value, (list, tuple)):
            flattened = []
            for item in value:
                flattened.extend(flatten_axes(item))
            return flattened
        return []

    axes = flatten_axes(returned_axes)
    require(len(axes) >= minimum_axes, "返回的 axes 数量不足")
    require(all(axis.figure is fig for axis in axes), "返回的 axes 必须属于返回的 Figure")

    artist_counts = [
        len(axis.lines)
        + len(axis.patches)
        + len(axis.collections)
        + len(axis.images)
        + len(axis.tables)
        + len(axis.texts)
        for axis in axes
    ]
    if every_axis:
        require(
            all(count > 0 for count in artist_counts),
            "图表含空白 axes：每个返回的 axes 都必须绘制真实数据或标注",
        )
    else:
        require(sum(artist_counts) > 0, "图表为空：至少绘制一条线、柱、散点、图像、表格或标注")


def distinct_tensors(tensors) -> None:
    require(len({id(item) for item in tensors}) == len(tensors), "参数列表包含重复对象")
