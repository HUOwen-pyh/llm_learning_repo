from __future__ import annotations

import base64
import csv
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from checks.common import close, distinct_tensors, figure_has_content, require
from makemore.data import Vocabulary, build_context_dataset, build_vocab
from makemore.diagnostics import (
    activation_gradient_statistics,
    activation_statistics,
    parameter_statistics,
    retain_intermediate_gradients,
)
from makemore.layers import BatchNorm1d, Embedding, FlattenConsecutive, Linear, Sequential, Tanh
from makemore.plotting import (
    plot_activation_gradient_histograms,
    plot_activation_histograms,
    plot_bn_axis_bug,
    plot_receptive_fields,
    plot_update_ratios,
    plot_wave_experiments,
    plot_wave_loss,
)
from makemore.reporting import save_shape_table, validate_artifacts
from makemore.training import moving_average
from makemore.wavenet import (
    WAVENET_REQUIRED_ARTIFACTS,
    WAVENET_REQUIRED_EXPERIMENTS,
    WaveNetConfig,
    WaveTrainingHistory,
    build_flat_model,
    build_hierarchical_model,
    evaluate_model,
    missing_required_wave_experiments,
    model_parameter_count,
    receptive_field_stages,
    run_wave_experiment,
    sample_names,
    trace_shapes,
    train_model,
    validate_wave_artifacts,
)


EXPECTED_WAVENET_ARTIFACTS = (
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

EXPECTED_WAVENET_EXPERIMENTS = (
    "flat_context_3",
    "flat_context_8",
    "hierarchical_small_before_bn_fix",
    "hierarchical_small_bn_fixed",
    "hierarchical_scaled",
)


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _write_valid_artifact(path: Path) -> None:
    if path.suffix == ".png":
        path.write_bytes(_ONE_PIXEL_PNG)
    elif path.suffix == ".json":
        path.write_text('{"ok": true}', encoding="utf-8")
    elif path.suffix == ".csv":
        path.write_text("column\nvalue\n", encoding="utf-8")
    else:
        path.write_text("sample\n", encoding="utf-8")


def _figure(result):
    return result[0] if isinstance(result, tuple) else result


def _artist_count(axis) -> int:
    return sum(
        len(items)
        for items in (axis.lines, axis.patches, axis.images, axis.collections, axis.texts)
    )


def _require_plot_content(result, minimum_axes: int = 1, every_axis: bool = False) -> None:
    """Stage 5 的图不能只创建空 axes。"""
    figure_has_content(result, minimum_axes=minimum_axes)
    fig = _figure(result)
    counts = [_artist_count(axis) for axis in fig.axes]
    if every_axis:
        require(all(count > 0 for count in counts), "每个 axes 都必须包含实际绘图内容")
    else:
        require(sum(counts) > 0, "图中没有 line/patch/image/collection/text")


def _require_raises_value_error(callback, message: str) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError(message)


def _require_hierarchical_layout(model: Sequential, depth: int, group_size: int) -> None:
    expected_length = 1 + depth * 4 + 1
    require(len(model.layers) == expected_length, "层次模型层数错误")
    require(isinstance(model.layers[0], Embedding), "第一层必须是 Embedding")
    for stage in range(depth):
        offset = 1 + stage * 4
        flatten, linear, batchnorm, activation = model.layers[offset : offset + 4]
        require(isinstance(flatten, FlattenConsecutive), "每阶段必须先 FlattenConsecutive")
        require(flatten.n == group_size, "FlattenConsecutive 分组大小错误")
        require(isinstance(linear, Linear) and linear.bias is None, "隐藏 Linear 必须关闭 bias")
        require(isinstance(batchnorm, BatchNorm1d), "隐藏 Linear 后必须接 BatchNorm1d")
        require(isinstance(activation, Tanh), "每阶段必须以 Tanh 结束")
    final = model.layers[-1]
    require(isinstance(final, Linear) and final.bias is not None, "输出 Linear 必须保留 bias")
    require(torch.count_nonzero(final.weight).item() > 0, "输出权重不能全零")
    require(torch.allclose(final.bias, torch.zeros_like(final.bias)), "输出 bias 应初始化为 0")


def check_43() -> None:
    vocab = build_vocab(["abcdefgh"])
    x, y = build_context_dataset(["abcdefgh"], vocab, block_size=8)
    require(tuple(x.shape) == (9, 8) and tuple(y.shape) == (9,), "8 字符数据 shape 错误")
    require(torch.equal(x[0], torch.zeros(8, dtype=torch.long)), "首个上下文应全是边界符")
    expected_last = torch.tensor([vocab.stoi[ch] for ch in "abcdefgh"])
    require(torch.equal(x[-1], expected_last) and y[-1].item() == 0, "最后应使用完整 8 字符预测边界")


def check_44() -> None:
    g = torch.Generator().manual_seed(201)
    linear = Linear(10, 20, g)
    x = torch.randn(4, 8, 10, generator=torch.Generator().manual_seed(202))
    vectorized = linear(x)
    looped = torch.stack([linear(x[:, t]) for t in range(x.shape[1])], dim=1)
    require(tuple(vectorized.shape) == (4, 8, 20), "N-D Linear shape 错误")
    require(torch.allclose(vectorized, looped), "向量化 Linear 应等价于时间 for-loop")
    embedding = Embedding(27, 10, torch.Generator().manual_seed(203))
    indices = torch.randint(0, 27, (4, 8), generator=torch.Generator().manual_seed(204))
    out = embedding(indices)
    require(tuple(out.shape) == (4, 8, 10), "Embedding shape 错误")
    require(len(embedding.parameters()) == 1 and embedding.parameters()[0] is embedding.weight, "Embedding 参数应只有 weight")


def check_45() -> None:
    x = torch.arange(2 * 8 * 3).view(2, 8, 3)
    layer = FlattenConsecutive(2)
    out = layer(x)
    require(tuple(out.shape) == (2, 4, 6), "第一次 FlattenConsecutive shape 错误")
    require(torch.equal(out[0, 0], torch.cat([x[0, 0], x[0, 1]])), "相邻位置拼接顺序错误")
    out2 = layer(out)
    require(tuple(out2.shape) == (2, 2, 12), "第二次 shape 错误")
    out3 = layer(out2)
    require(tuple(out3.shape) == (2, 24), "最后时间轴应只 squeeze dim=1")
    single = layer(torch.arange(1 * 2 * 3).view(1, 2, 3))
    require(tuple(single.shape) == (1, 6), "batch size=1 时不能删除 batch 轴")


def check_46() -> None:
    model = Sequential([
        Embedding(7, 3, torch.Generator().manual_seed(205)),
        FlattenConsecutive(2),
        Linear(6, 5, torch.Generator().manual_seed(206), bias=False),
        BatchNorm1d(5),
        Tanh(),
        Linear(5, 7, torch.Generator().manual_seed(207)),
    ])
    x = torch.randint(0, 7, (4, 2), generator=torch.Generator().manual_seed(208))
    records = trace_shapes(model, x)
    expected_names = ["Embedding", "FlattenConsecutive", "Linear", "BatchNorm1d", "Tanh", "Linear"]
    expected_inputs = [(4, 2), (4, 2, 3), (4, 6), (4, 5), (4, 5), (4, 5)]
    expected_outputs = [(4, 2, 3), (4, 6), (4, 5), (4, 5), (4, 5), (4, 7)]
    expected_parameters = [21, 0, 30, 10, 0, 42]
    require(len(records) == len(model.layers), "shape trace 应覆盖每层")
    require([record.index for record in records] == list(range(6)), "shape trace index 错误")
    require([record.name for record in records] == expected_names, "shape trace 层名或层序错误")
    require([record.input_shape for record in records] == expected_inputs, "shape trace 输入形状错误")
    require([record.output_shape for record in records] == expected_outputs, "shape trace 输出形状错误")
    require([record.parameter_count for record in records] == expected_parameters, "shape trace 每层参数量错误")
    require(sum(record.parameter_count for record in records) == sum(p.nelement() for p in model.parameters()), "trace 参数量总和错误")
    model.eval()
    require(all(not layer.training for layer in model.layers if isinstance(layer, BatchNorm1d)), "eval 没有传播")
    model.train()
    require(all(layer.training for layer in model.layers if isinstance(layer, BatchNorm1d)), "train(True) 没有传播")
    distinct_tensors(model.parameters())
    with tempfile.TemporaryDirectory() as directory:
        path = save_shape_table(records, Path(directory) / "shapes.csv")
        require(path.exists() and path.stat().st_size > 20, "shape CSV 未生成")
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required_columns = {"index", "name", "input_shape", "output_shape", "parameter_count"}
        require(rows and required_columns <= set(rows[0]), "CSV 缺少完整 trace 表头")
        require(len(rows) == len(records), "CSV 必须逐层写出所有记录")
        for row, record in zip(rows, records):
            require(int(row["index"]) == record.index and row["name"] == record.name, "CSV 层序错误")
            require(row["input_shape"] == str(record.input_shape), "CSV input_shape 格式或内容错误")
            require(row["output_shape"] == str(record.output_shape), "CSV output_shape 格式或内容错误")
            require(int(row["parameter_count"]) == record.parameter_count, "CSV parameter_count 错误")


def check_47() -> None:
    model = build_flat_model(27, 8, 10, 200, torch.Generator().manual_seed(210))
    require(model_parameter_count(model) == 22097, "8 字符扁平基线参数量应为 22097")
    expected_types = [Embedding, FlattenConsecutive, Linear, BatchNorm1d, Tanh, Linear]
    require(len(model.layers) == len(expected_types), "扁平基线层数错误")
    require(all(isinstance(layer, kind) for layer, kind in zip(model.layers, expected_types)), "扁平基线层序错误")
    require(model.layers[1].n == 8, "扁平模型必须一次合并全部 8 个位置")
    require(model.layers[2].bias is None, "BatchNorm 前的 hidden Linear 必须关闭 bias")
    require(model.layers[-1].bias is not None, "输出 Linear 必须保留 bias")
    require(torch.allclose(model.layers[-1].bias, torch.zeros_like(model.layers[-1].bias)), "输出 bias 应初始化为 0")
    unscaled = build_flat_model(
        27,
        8,
        10,
        200,
        torch.Generator().manual_seed(210),
        output_scale=1.0,
    )
    require(
        torch.allclose(model.layers[-1].weight, unscaled.layers[-1].weight * 0.1),
        "输出层权重必须按 output_scale=0.1 缩小",
    )
    x = torch.randint(0, 27, (5, 8), generator=torch.Generator().manual_seed(211))
    require(tuple(model(x).shape) == (5, 27), "扁平基线输出 shape 错误")


def check_48() -> None:
    config = WaveNetConfig()
    model = build_hierarchical_model(config, torch.Generator().manual_seed(212))
    require(model_parameter_count(model) == 76579, "官方层次模型参数量应为 76579")
    _require_hierarchical_layout(model, depth=3, group_size=2)
    unscaled = build_hierarchical_model(
        replace(config, output_scale=1.0),
        torch.Generator().manual_seed(212),
    )
    require(
        torch.allclose(model.layers[-1].weight, unscaled.layers[-1].weight * 0.1),
        "层次模型输出权重必须按 output_scale 缩放",
    )
    x = torch.randint(0, 27, (2, 8), generator=torch.Generator().manual_seed(213))
    records = trace_shapes(model, x)
    expected_names = [
        "Embedding",
        "FlattenConsecutive", "Linear", "BatchNorm1d", "Tanh",
        "FlattenConsecutive", "Linear", "BatchNorm1d", "Tanh",
        "FlattenConsecutive", "Linear", "BatchNorm1d", "Tanh",
        "Linear",
    ]
    expected_shapes = [
        (2, 8, 24),
        (2, 4, 48),
        (2, 4, 128),
        (2, 4, 128),
        (2, 4, 128),
        (2, 2, 256),
        (2, 2, 128),
        (2, 2, 128),
        (2, 2, 128),
        (2, 256),
        (2, 128),
        (2, 128),
        (2, 128),
        (2, 27),
    ]
    expected_counts = [648, 0, 6144, 256, 0, 0, 32768, 256, 0, 0, 32768, 256, 0, 3483]
    require([record.name for record in records] == expected_names, "官方层次模型层序错误")
    require(
        [record.input_shape for record in records] == [(2, 8), *expected_shapes[:-1]],
        "官方层次模型逐层 input shape 错误",
    )
    require([record.output_shape for record in records] == expected_shapes, "官方层次模型逐层 shape 错误")
    require([record.parameter_count for record in records] == expected_counts, "官方层次模型逐层参数量错误")
    require(sum(record.parameter_count for record in records) == 76579, "官方 trace 参数量总和错误")

    for alternate, depth in [
        (WaveNetConfig(vocab_size=11, block_size=4, n_embd=5, n_hidden=7, group_size=2), 2),
        (WaveNetConfig(vocab_size=11, block_size=9, n_embd=5, n_hidden=7, group_size=3), 2),
    ]:
        seen_dims: list[int] = []

        def batchnorm_factory(dim: int) -> BatchNorm1d:
            seen_dims.append(dim)
            return BatchNorm1d(dim)

        alternate_model = build_hierarchical_model(
            alternate,
            torch.Generator().manual_seed(300 + alternate.block_size),
            batchnorm_factory=batchnorm_factory,
        )
        _require_hierarchical_layout(alternate_model, depth=depth, group_size=alternate.group_size)
        require(seen_dims == [alternate.n_hidden] * depth, "builder 必须通过 batchnorm_factory 构造每个 BN")
        alternate_x = torch.randint(
            0,
            alternate.vocab_size,
            (3, alternate.block_size),
            generator=torch.Generator().manual_seed(310 + alternate.block_size),
        )
        require(tuple(alternate_model(alternate_x).shape) == (3, alternate.vocab_size), "通用层次 builder 输出 shape 错误")

    invalid_configs = [
        WaveNetConfig(block_size=6, group_size=2),
        WaveNetConfig(block_size=8, group_size=1),
        WaveNetConfig(block_size=0, group_size=2),
    ]
    for invalid in invalid_configs:
        _require_raises_value_error(
            lambda invalid=invalid: build_hierarchical_model(invalid, torch.Generator().manual_seed(320)),
            f"非法 block/group 配置应抛出 ValueError：{invalid}",
        )


def check_49() -> None:
    for block_size, group_size, expected_widths in [
        (8, 2, [1, 2, 4, 8]),
        (4, 2, [1, 2, 4]),
        (9, 3, [1, 3, 9]),
    ]:
        stages = receptive_field_stages(block_size, group_size)
        require(len(stages) == len(expected_widths), "感受野阶段数错误")
        for stage, width in zip(stages, expected_widths):
            require(len(stage) == block_size // width, "每阶段节点数错误")
            require(all(len(group) == width for group in stage), "同一阶段的感受野宽度必须一致")
            expected_groups = [tuple(range(start, start + width)) for start in range(0, block_size, width)]
            require(stage == expected_groups, "感受野必须按连续、无重叠的位置分组")
        require(stages[-1] == [tuple(range(block_size))], "最终节点必须覆盖全部输入")
    stages = receptive_field_stages(8, 2)
    _require_plot_content(plot_receptive_fields(stages))
    for block_size, group_size in [(6, 2), (8, 1), (0, 2)]:
        _require_raises_value_error(
            lambda block_size=block_size, group_size=group_size: receptive_field_stages(block_size, group_size),
            "非法感受野配置应抛出 ValueError",
        )


def check_50() -> None:
    x = torch.randn(4, 3, 5, generator=torch.Generator().manual_seed(214))
    x[:, 1] += 10
    x[:, 2] -= 7
    bn = BatchNorm1d(5, momentum=0.25)
    old_mean = bn.running_mean.clone()
    old_var = bn.running_var.clone()
    out = bn(x)
    require(tuple(out.shape) == (4, 3, 5), "3D BN 不应改变 shape")
    require(torch.allclose(out.mean((0, 1)), torch.zeros(5), atol=1e-5), "3D BN 应在 (B,T) 上联合归一化")
    require(torch.allclose(out.var((0, 1), correction=1), torch.ones(5), atol=2e-4), "3D BN 全局样本方差应接近 1")
    batch_mean = x.mean((0, 1))
    batch_var = x.var((0, 1), correction=1)
    expected_running_mean = 0.75 * old_mean.reshape(-1) + 0.25 * batch_mean
    expected_running_var = 0.75 * old_var.reshape(-1) + 0.25 * batch_var
    require(bn.running_mean.numel() == 5 and bn.running_var.numel() == 5, "3D BN 每通道只能保存一组 running stats")
    require(torch.allclose(bn.running_mean.reshape(-1), expected_running_mean), "3D running_mean 统计轴错误")
    require(torch.allclose(bn.running_var.reshape(-1), expected_running_var), "3D running_var 统计轴错误")
    saved_mean = bn.running_mean.clone()
    saved_var = bn.running_var.clone()
    bn.eval()
    single = x[:1]
    eval_single = bn(single)
    eval_with_companions = bn(torch.cat([single, x[1:]], dim=0))[:1]
    require(tuple(eval_single.shape) == (1, 3, 5), "3D BN eval 不能删除 batch/time 轴")
    require(torch.allclose(eval_single, eval_with_companions), "3D BN eval 不应依赖 batch companions")
    running_mean = saved_mean.reshape(1, 1, -1)
    running_var = saved_var.reshape(1, 1, -1)
    gamma = bn.gamma.reshape(1, 1, -1)
    beta = bn.beta.reshape(1, 1, -1)
    manual_eval = (single - running_mean) / torch.sqrt(running_var + bn.eps) * gamma + beta
    require(torch.allclose(eval_single, manual_eval), "3D BN eval 必须使用 running statistics")
    require(torch.equal(saved_mean, bn.running_mean) and torch.equal(saved_var, bn.running_var), "3D BN eval 不得更新 buffers")
    wrong_mean = x.mean(0, keepdim=True)
    wrong_var = x.var(0, keepdim=True, correction=1)
    wrong = (x - wrong_mean) / torch.sqrt(wrong_var + 1e-5)
    correct_mean = x.mean((0, 1), keepdim=True)
    correct_var = x.var((0, 1), keepdim=True, correction=1)
    correct = (x - correct_mean) / torch.sqrt(correct_var + 1e-5)
    _require_plot_content(plot_bn_axis_bug(wrong, correct), minimum_axes=2, every_axis=True)


def _tiny_wave_model(seed: int = 220) -> Sequential:
    config = WaveNetConfig(vocab_size=3, block_size=8, n_embd=2, n_hidden=8, group_size=2)
    return build_hierarchical_model(config, torch.Generator().manual_seed(seed))


def _tiny_wave_data(seed: int = 221, count: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, 3, (count, 8), generator=torch.Generator().manual_seed(seed))
    # 一个很容易学习、同时确实依赖上下文的目标。
    y = x[:, -1].clone()
    return x, y


class _SamplingProbe:
    """产生 a、b、边界，并记录 sample_names 实际喂入的滚动上下文。"""

    def __init__(self) -> None:
        self.training = True
        self.contexts: list[torch.Tensor] = []
        self.out: torch.Tensor | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.contexts.append(x.detach().clone())
        next_index = (1, 2, 0)[min(len(self.contexts) - 1, 2)]
        self.out = torch.full((x.shape[0], 3), -100.0, device=x.device)
        self.out[:, next_index] = 100.0
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return []

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)


def check_51() -> None:
    model = _tiny_wave_model()
    x, y = _tiny_wave_data()
    model.eval()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    history = train_model(
        model,
        x,
        y,
        torch.Generator().manual_seed(222),
        steps=100,
        batch_size=16,
        learning_rate=0.08,
        decay_at=60,
        decay_factor=0.1,
        record_update_every=5,
    )
    require(isinstance(history, WaveTrainingHistory), "train_model 必须返回 WaveTrainingHistory")
    require(history.steps == list(range(100)), "训练 history.steps 错误")
    require(len(history.losses) == 100 and all(math.isfinite(item) for item in history.losses), "loss 历史错误")
    require(len(history.learning_rates) == 100, "必须逐步记录实际 learning rate")
    require(all(abs(close_lr - 0.08) < 1e-12 for close_lr in history.learning_rates[:60]), "衰减前 learning rate 错误")
    require(all(abs(close_lr - 0.008) < 1e-12 for close_lr in history.learning_rates[60:]), "衰减后 learning rate 错误")
    expected_update_steps = list(range(0, 100, 5))
    require(history.update_steps == expected_update_steps, "update:data 记录步数错误")
    expected_parameter_names = {f"p{i}" for i in range(len(model.parameters()))}
    require(set(history.update_to_data_log10) == expected_parameter_names, "update:data 必须覆盖每个参数")
    require(
        all(len(values) == len(expected_update_steps) for values in history.update_to_data_log10.values()),
        "每个参数的 update:data 历史长度错误",
    )
    recorded_ratios = [value for values in history.update_to_data_log10.values() for value in values if value is not None]
    require(recorded_ratios and all(math.isfinite(value) for value in recorded_ratios), "update:data 必须记录 finite 数值或 None")
    require(any(not torch.allclose(old, new) for old, new in zip(before, model.parameters())), "train_model 必须真的更新参数")
    require(all(layer.training for layer in model.layers if isinstance(layer, BatchNorm1d)), "train_model 必须进入 train 模式")
    require(sum(history.losses[-15:]) / 15 < sum(history.losses[:15]) / 15, "短程 loss 应下降")
    smoothed = moving_average(history.losses, 13)
    require(len(smoothed) == 8, "平滑必须保留不足一个窗口的尾部")
    smoothed_steps = [
        history.steps[min((index + 1) * 13, len(history.steps)) - 1]
        for index in range(len(smoothed))
    ]

    model.train()
    evaluated = evaluate_model(model, x, y)
    require(math.isfinite(evaluated), "评估 loss 必须有限")
    require(all(not layer.training for layer in model.layers if isinstance(layer, BatchNorm1d)), "evaluate_model 必须进入 eval 模式")
    expected_eval = F.cross_entropy(model(x), y).item()
    close(evaluated, expected_eval, tolerance=1e-6, message="evaluate_model 必须返回完整数据 CE")

    vocab = Vocabulary({".": 0, "a": 1, "b": 2}, {0: ".", 1: "a", 2: "b"})
    model.train()
    samples_a = sample_names(model, vocab, 8, torch.Generator().manual_seed(223), count=3, max_length=6)
    require(all(not layer.training for layer in model.layers if isinstance(layer, BatchNorm1d)), "sample_names 必须自行进入 eval 模式")
    model.train()
    samples_b = sample_names(model, vocab, 8, torch.Generator().manual_seed(223), count=3, max_length=6)
    require(samples_a == samples_b, "固定 generator 的层次模型采样必须可复现")
    require(len(samples_a) == 3 and all(len(item) <= 6 for item in samples_a), "WaveNet 采样数量/最大长度错误")
    require(all(character in {"a", "b"} for item in samples_a for character in item), "采样结果含词表外字符或边界符")

    probe = _SamplingProbe()
    probe_model = Sequential([probe])
    probe_samples = sample_names(
        probe_model,
        vocab,
        8,
        torch.Generator().manual_seed(224),
        count=1,
        max_length=6,
    )
    require(probe_samples == ["ab"], "采样应在边界前返回已生成字符，且不包含边界符")
    expected_contexts = [
        torch.zeros((1, 8), dtype=torch.long),
        torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]]),
        torch.tensor([[0, 0, 0, 0, 0, 0, 1, 2]]),
    ]
    require(len(probe.contexts) == 3, "采样未按预测次数调用模型")
    require(all(torch.equal(actual, expected) for actual, expected in zip(probe.contexts, expected_contexts)), "8 字符上下文没有正确滚动")
    require(not probe.training, "sample_names 必须在调用模型前切换 eval")

    loss_plot = plot_wave_loss(history.steps, history.losses, smoothed_steps, smoothed)
    _require_plot_content(loss_plot)
    require(sum(len(axis.lines) for axis in _figure(loss_plot).axes) >= 2, "loss 图必须同时画原始与平滑曲线")
    plotted_x = [list(line.get_xdata()) for axis in _figure(loss_plot).axes for line in axis.lines]
    require(any(values == history.steps for values in plotted_x), "原始 loss 曲线横坐标必须使用 history.steps")
    require(any(values == smoothed_steps for values in plotted_x), "平滑 loss 曲线必须使用对应的尾块 step")


def check_52() -> None:
    model = _tiny_wave_model(230)
    x, y = _tiny_wave_data(231, count=48)
    result = run_wave_experiment(
        "hierarchical_small_bn_fixed",
        model,
        x,
        y,
        x,
        y,
        torch.Generator().manual_seed(232),
        steps=10,
        batch_size=8,
        learning_rate=0.05,
        config={"block_size": 8},
        decay_at=5,
        decay_factor=0.2,
        record_update_every=2,
    )
    require(result.name == "hierarchical_small_bn_fixed" and result.parameter_count > 0, "实验报告字段错误")
    require(math.isfinite(result.train_loss) and math.isfinite(result.dev_loss), "实验 loss 必须有限")
    require(isinstance(result.history, WaveTrainingHistory), "实验结果必须携带训练 history")
    expected_schedule = {
        "learning_rate": 0.05,
        "decay_at": 5,
        "decay_factor": 0.2,
        "record_update_every": 2,
    }
    require(all(result.config.get(key) == value for key, value in expected_schedule.items()), "实验 config 必须记录实际 schedule")

    require(WAVENET_REQUIRED_EXPERIMENTS == EXPECTED_WAVENET_EXPERIMENTS, "不得缩减正式实验清单")
    suite = [replace(result, name=name) for name in WAVENET_REQUIRED_EXPERIMENTS]
    require(missing_required_wave_experiments(suite) == [], "完整五组实验不应报告缺失")
    require(
        missing_required_wave_experiments(suite[:-1]) == [WAVENET_REQUIRED_EXPERIMENTS[-1]],
        "实验套件校验必须按固定顺序报告缺项",
    )
    experiment_plot = plot_wave_experiments(suite)
    _require_plot_content(experiment_plot)
    require(
        sum(len(axis.lines) + len(axis.patches) + len(axis.collections) for axis in _figure(experiment_plot).axes) >= 2,
        "实验图必须实际比较 train/dev 指标",
    )

    # run_wave_experiment 可能留下最后一个训练 batch 的参数梯度；诊断必须从干净状态开始。
    parameters = model.parameters()
    for parameter in parameters:
        parameter.grad = None
    model.train()
    logits = model(x[:8])
    tanh_layers = [layer for layer in model.layers if isinstance(layer, Tanh)]
    require([layer.out.ndim for layer in tanh_layers] == [3, 3, 2], "最终诊断必须覆盖 3D 与 2D Tanh 输出")
    retain_intermediate_gradients(model)
    loss = F.cross_entropy(logits, y[:8])
    loss.backward()
    astats = activation_statistics(model)
    gstats = activation_gradient_statistics(model)
    pstats = parameter_statistics([(f"p{i}", p) for i, p in enumerate(parameters)], 0.05)
    require(len(astats) == 3 and all(item.saturation_ratio is not None for item in astats), "必须统计全部三个 Tanh 的激活与饱和率")
    require(len(gstats) == 3 and all(item.finite_ratio == 1.0 for item in gstats), "必须统计全部三个 Tanh 的激活梯度")
    require(len(pstats) == len(parameters), "parameter statistics 必须覆盖全部参数")
    activation_tensors = [(f"tanh_{i}", layer.out) for i, layer in enumerate(model.layers) if isinstance(layer, Tanh)]
    gradient_tensors = [(name, tensor.grad) for name, tensor in activation_tensors]
    require(all(tensor is not None for _, tensor in gradient_tensors), "诊断必须读取当前 forward 的 layer.out.grad")
    _require_plot_content(plot_activation_histograms(activation_tensors), minimum_axes=3, every_axis=True)
    _require_plot_content(plot_activation_gradient_histograms(gradient_tensors), minimum_axes=3, every_axis=True)
    update_history = {
        name: [value if value is not None else -9.0 for value in values]
        for name, values in result.history.update_to_data_log10.items()
        if values
    }
    require(update_history, "实验 history 必须提供训练过程中的 update:data 轨迹")
    require(all(len(values) == len(result.history.update_steps) for values in update_history.values()), "update:data 曲线长度错误")
    _require_plot_content(plot_update_ratios(update_history))

    require(WAVENET_REQUIRED_ARTIFACTS == EXPECTED_WAVENET_ARTIFACTS, "不得缩减正式工件清单")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in WAVENET_REQUIRED_ARTIFACTS[:-1]:
            _write_valid_artifact(root / name)
        require(
            validate_wave_artifacts(root) == [WAVENET_REQUIRED_ARTIFACTS[-1]],
            "固定工件校验必须发现唯一缺项",
        )
        _write_valid_artifact(root / WAVENET_REQUIRED_ARTIFACTS[-1])
        require(validate_wave_artifacts(root) == [], "完整固定工件清单不应报告缺失")

        invalid_names = ("broken.png", "empty.json", "bad.json", "header_only.csv", "blank.txt")
        (root / "broken.png").write_bytes(b"not a png")
        (root / "empty.json").write_text("{}", encoding="utf-8")
        (root / "bad.json").write_text("{", encoding="utf-8")
        (root / "header_only.csv").write_text("a,b\n", encoding="utf-8")
        (root / "blank.txt").write_text("  \n", encoding="utf-8")
        require(
            validate_artifacts(root, invalid_names) == list(invalid_names),
            "工件校验必须识别伪 PNG、空/损坏 JSON、无数据 CSV 与空白 TXT",
        )
