from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from checks.common import close, distinct_tensors, figure_has_content, finite_tensor, require
from makemore.diagnostics import (
    DiagnosticRun,
    GradientStats,
    ParameterStats,
    TensorStats,
    activation_gradient_statistics,
    activation_statistics,
    batch_coupling_difference,
    calibrate_batchnorm,
    fix_output_initialization,
    fold_batchnorm,
    initialization_report,
    kaiming_scale,
    parameter_statistics,
    retain_intermediate_gradients,
    tanh_saturation,
    tensor_stats,
    uniform_nll,
    variance_propagation,
    zero_initialization_pathology,
)
from makemore.layers import BatchNorm1d, Linear, Tanh, build_deep_mlp
from makemore.mlp import MLPConfig, clone_parameters, init_mlp, mlp_forward
from makemore.plotting import (
    plot_activation_gradient_histograms,
    plot_activation_histograms,
    plot_activation_summary,
    plot_batch_coupling,
    plot_bn_folding,
    plot_diagnostics_dashboard,
    plot_initialization_comparison,
    plot_parameter_gradient_histograms,
    plot_saturation_heatmap,
    plot_tanh_histograms,
    plot_update_ratios,
    plot_variance_propagation,
    plot_zero_init_pathology,
)


def _figure_artist_count(result) -> tuple[object, list[int]]:
    """比只数 axes 更严格：每个面板都必须真的画出数据。"""
    fig = result[0] if isinstance(result, tuple) else result
    counts = [
        len(axis.lines) + len(axis.patches) + len(axis.images) + len(axis.collections)
        for axis in fig.axes
    ]
    return fig, counts


def check_21() -> None:
    close(uniform_nll(27), math.log(27))
    logits = torch.zeros(8, 27)
    targets = torch.arange(8) % 27
    report = initialization_report(logits, targets)
    close(report.cross_entropy, math.log(27))
    close(report.logits_mean, 0.0)
    close(report.logits_std, 0.0)
    close(report.mean_max_probability, 1 / 27)
    close(report.mean_entropy, math.log(27))

    nontrivial = torch.tensor([[2.0, -1.0, 0.5], [-0.5, 0.25, 1.5]])
    nontrivial_targets = torch.tensor([0, 2])
    probabilities = nontrivial.softmax(1)
    expected = initialization_report(nontrivial, nontrivial_targets)
    close(expected.cross_entropy, F.cross_entropy(nontrivial, nontrivial_targets).item())
    close(expected.logits_mean, nontrivial.mean().item())
    close(expected.logits_std, nontrivial.std().item())
    close(expected.mean_max_probability, probabilities.max(1).values.mean().item())
    close(expected.mean_entropy, (-(probabilities * probabilities.log()).sum(1)).mean().item())


def check_22() -> None:
    config = MLPConfig(vocab_size=7, block_size=3, n_embd=4, n_hidden=16)
    naive = init_mlp(config, torch.Generator().manual_seed(21))
    fixed = clone_parameters(naive)
    fix_output_initialization(fixed, weight_scale=0.01, zero_bias=True)
    require(fixed.W2.std() < naive.W2.std() * 0.02, "W2 没有显著缩小")
    require(torch.equal(fixed.b2, torch.zeros_like(fixed.b2)), "b2 应清零")
    x = torch.randint(0, 7, (16, 3), generator=torch.Generator().manual_seed(22))
    y = torch.randint(0, 7, (16,), generator=torch.Generator().manual_seed(23))
    naive_logits = mlp_forward(naive, x)
    fixed_logits = mlp_forward(fixed, x)
    require(fixed_logits.std() < naive_logits.std(), "修复后 logits std 应更小")
    fixed_loss = F.cross_entropy(fixed_logits, y).item()
    require(abs(fixed_loss - math.log(7)) < abs(F.cross_entropy(naive_logits, y).item() - math.log(7)), "初始 loss 应更接近均匀基线")
    figure_has_content(plot_initialization_comparison(naive_logits, fixed_logits, [3.5, 3.2], [2.0, 1.95]), minimum_axes=2)


def check_23() -> None:
    hpre = torch.tensor([[-4.0, -0.5, 0.0, 0.5, 4.0]])
    h = torch.tanh(hpre)
    expected = (h.abs() > 0.97).float().mean().item()
    close(tanh_saturation(h, 0.97), expected)
    stats = tensor_stats("h", h, saturation_threshold=0.97)
    close(stats.mean, h.mean().item())
    close(stats.std, h.std().item())
    close(stats.minimum, h.min().item())
    close(stats.maximum, h.max().item())
    close(stats.saturation_ratio, expected)
    require(stats.finite_ratio == 1.0 and stats.minimum <= stats.maximum, "tensor stats 不完整")
    figure_has_content(plot_tanh_histograms(hpre, h), minimum_axes=2)
    figure_has_content(plot_saturation_heatmap(h, 0.97))


def check_24() -> None:
    close(kaiming_scale(25, 1.0), 0.2)
    close(kaiming_scale(25, 5 / 3), 1 / 3)
    points = variance_propagation(
        torch.randn(256, 32, generator=torch.Generator().manual_seed(24)),
        depth=4,
        gains=[1.0, 5 / 3, 3.0],
        generator=torch.Generator().manual_seed(25),
    )
    require(len(points) == 12, "应记录 gains × depth 个点")
    require({item.gain for item in points} == {1.0, 5 / 3, 3.0}, "gain 记录不完整")
    for gain in (1.0, 5 / 3, 3.0):
        group = [item for item in points if item.gain == gain]
        require([item.layer_index for item in group] == list(range(4)), "每个 gain 都应从同一输入独立记录 0..depth-1")
    require(all(math.isfinite(item.preactivation_std) and math.isfinite(item.activation_std) for item in points), "方差传播统计必须有限")
    require(all(0.0 <= item.saturation_ratio <= 1.0 for item in points), "饱和率必须位于 [0,1]")
    figure_has_content(plot_variance_propagation(points), minimum_axes=2)


def check_25() -> None:
    bn = BatchNorm1d(4)
    require(tuple(bn.gamma.shape) == tuple(bn.beta.shape) == (4,), "gamma/beta shape 固定为 (C,)")
    require(tuple(bn.running_mean.shape) == tuple(bn.running_var.shape) == (4,), "running stats shape 固定为 (C,)")
    x = torch.randn(32, 4, generator=torch.Generator().manual_seed(26)) * 3 + 5
    out = bn(x)
    require(tuple(out.shape) == (32, 4), "BatchNorm 不应改变 shape")
    require(torch.allclose(out.mean(0), torch.zeros(4), atol=1e-5), "训练态输出均值应接近 0")
    require(torch.allclose(out.var(0, correction=1), torch.ones(4), atol=2e-4), "训练态样本方差应接近 1")
    params = bn.parameters()
    require(len(params) == 2 and params[0] is bn.gamma and params[1] is bn.beta, "BN 参数只能是 gamma/beta")
    require(all(item.requires_grad for item in params), "gamma/beta 应可训练")


def check_26() -> None:
    bn = BatchNorm1d(3, momentum=0.25)
    x = torch.randn(16, 3, generator=torch.Generator().manual_seed(27)) + 4
    old_mean = bn.running_mean.clone()
    old_var = bn.running_var.clone()
    bn(x)
    expected_mean = 0.75 * old_mean + 0.25 * x.mean(0)
    expected_var = 0.75 * old_var + 0.25 * x.var(0, correction=1)
    require(bn.running_mean.shape == old_mean.shape and torch.allclose(bn.running_mean, expected_mean), "running_mean momentum 更新/shape 错误")
    require(bn.running_var.shape == old_var.shape and torch.allclose(bn.running_var, expected_var), "running_var momentum 更新/shape 错误")
    saved_mean = bn.running_mean.clone()
    saved_var = bn.running_var.clone()
    bn.eval()
    out1 = bn(x[:1])
    out2 = bn(torch.cat([x[:1], x[5:10]], dim=0))[:1]
    require(torch.allclose(out1, out2), "eval 中同一样本不应依赖 batch companions")
    require(torch.equal(saved_mean, bn.running_mean) and torch.equal(saved_var, bn.running_var), "eval 不应更新 buffers")
    bn.train()
    require(bn.training, "train() 应恢复训练模式")

def check_27() -> None:
    bn = BatchNorm1d(2)
    sample = torch.tensor([[1.0, -1.0]])
    a = torch.tensor([[0.0, 0.0], [2.0, 2.0], [3.0, 1.0]])
    b = torch.tensor([[10.0, 10.0], [12.0, 12.0], [8.0, 9.0]])
    report = batch_coupling_difference(bn, sample, a, b)
    require(report.output_with_companions_a.shape == sample.shape, "coupling 输出 A shape 错误")
    require(report.output_with_companions_b.shape == sample.shape, "coupling 输出 B shape 错误")
    measured = (report.output_with_companions_a - report.output_with_companions_b).abs().max().item()
    close(report.max_abs_difference, measured)
    require(report.max_abs_difference > 1e-3, "训练态 BN 应体现 batch coupling")
    x = torch.randn(32, 3, generator=torch.Generator().manual_seed(28))
    bias = torch.tensor([[4.0, -2.0, 1.0]])
    bn1, bn2 = BatchNorm1d(3), BatchNorm1d(3)
    out_with_bias = bn1(x + bias)
    out_without_bias = bn2(x)
    require(torch.allclose(out_with_bias, out_without_bias, atol=1e-5), "BN 应消除逐通道常数 bias")
    coupling_figure = plot_batch_coupling(report.output_with_companions_a, report.output_with_companions_b)
    figure_has_content(coupling_figure)
    _, counts = _figure_artist_count(coupling_figure)
    require(any(count > 0 for count in counts), "batch-coupling 图必须画出两组实际输出")


def check_28() -> None:
    x = torch.randn(20, 3, generator=torch.Generator().manual_seed(30))

    def verify(bias: bool, seed: int):
        linear = Linear(3, 4, torch.Generator().manual_seed(seed), bias=bias)
        bn = BatchNorm1d(4)
        with torch.no_grad():
            bn.running_mean.copy_(torch.tensor([1.0, -2.0, 0.5, 3.0]))
            bn.running_var.copy_(torch.tensor([4.0, 9.0, 1.0, 16.0]))
            bn.gamma.copy_(torch.tensor([1.5, -0.5, 2.0, 0.25]))
            bn.beta.copy_(torch.tensor([0.1, 0.2, -0.3, 1.0]))
        bn.eval()
        original_weight = linear.weight.detach().clone()
        original_bias = None if linear.bias is None else linear.bias.detach().clone()
        reference = bn(linear(x)).detach().clone()  # 必须在 fold 前计算。
        fused = fold_batchnorm(linear, bn)
        require(fused is not linear, "fold_batchnorm 必须返回新 Linear，不能原地改输入")
        require(torch.equal(linear.weight, original_weight), "fold 不得修改原 Linear.weight")
        if original_bias is None:
            require(linear.bias is None, "fold 不得给原本无 bias 的 Linear 添 bias")
        else:
            require(torch.equal(linear.bias, original_bias), "fold 不得修改原 Linear.bias")
        result = fused(x)
        max_difference = (reference - result).abs().max().item()
        require(max_difference < 1e-5, f"fold 后最大绝对误差过大: {max_difference}")
        return reference, result

    reference, result = verify(True, 29)
    verify(False, 291)
    figure_has_content(plot_bn_folding(reference, result))


def check_29() -> None:
    model = build_deep_mlp(27, 3, 10, 100, 5, torch.Generator().manual_seed(31))
    parameters = model.parameters()
    require(sum(item.nelement() for item in parameters) == 47024, "官方深层配置参数量应为 47024")
    distinct_tensors(parameters)
    require(all(parameter.requires_grad for parameter in parameters), "所有 DeepMLP 参数都应 requires_grad")
    x = torch.randint(0, 27, (16, 3), generator=torch.Generator().manual_seed(32))
    logits = model(x)
    require(tuple(logits.shape) == (16, 27), "DeepMLP 输出 shape 错误")
    tanh_layers = [layer for layer in model.network.layers if isinstance(layer, Tanh)]
    require(len(tanh_layers) == 5 and all(layer.out is not None for layer in tanh_layers), "应有 5 个保存 out 的 Tanh")
    model.eval()
    require(all(not layer.training for layer in model.network.layers if isinstance(layer, BatchNorm1d)), "eval 模式没有向下传播")
    model.train()
    require(all(layer.training for layer in model.network.layers if isinstance(layer, BatchNorm1d)), "train 模式没有向下传播")

    # DeepMLP 在本关才可用，因此完整数据 BN calibration 也在这里验收。
    calibration_model = build_deep_mlp(7, 3, 3, 8, 2, torch.Generator().manual_seed(270))
    tokens = torch.randint(0, 7, (32, 3), generator=torch.Generator().manual_seed(271))
    bn_layers = [layer for layer in calibration_model.network.layers if isinstance(layer, BatchNorm1d)]
    calibration_model.eval()
    calibrate_batchnorm(calibration_model, tokens)
    require(all(not layer.training for layer in bn_layers), "校准后必须恢复调用前的 eval 模式")
    calibrated = [(layer.running_mean.clone(), layer.running_var.clone()) for layer in bn_layers]
    require(all(torch.isfinite(mean).all() and torch.isfinite(var).all() for mean, var in calibrated), "校准统计必须 finite")

    calibration_model.train()
    with torch.no_grad():
        calibration_model(tokens)
    expected_stats = []
    for index, layer in enumerate(calibration_model.network.layers):
        if isinstance(layer, BatchNorm1d):
            incoming = calibration_model.network.layers[index - 1].out
            require(incoming is not None, "BN 前一层必须保存 out")
            axes = tuple(range(incoming.ndim - 1))
            expected_stats.append((incoming.mean(dim=axes), incoming.var(dim=axes, correction=1)))
    require(len(calibrated) == len(expected_stats), "必须校准所有 BatchNorm 层")
    for (actual_mean, actual_var), (wanted_mean, wanted_var) in zip(calibrated, expected_stats):
        require(actual_mean.shape == wanted_mean.shape and torch.allclose(actual_mean, wanted_mean, atol=1e-6), "calibration running_mean 不是完整数据精确统计")
        require(actual_var.shape == wanted_var.shape and torch.allclose(actual_var, wanted_var, atol=1e-6), "calibration running_var 不是完整数据样本方差")


def _deep_model_with_backward(seed: int = 33):
    model = build_deep_mlp(9, 3, 4, 12, 3, torch.Generator().manual_seed(seed))
    x = torch.randint(0, 9, (20, 3), generator=torch.Generator().manual_seed(seed + 1))
    y = torch.randint(0, 9, (20,), generator=torch.Generator().manual_seed(seed + 2))
    logits = model(x)
    retain_intermediate_gradients(model)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    return model, loss


def check_30() -> None:
    model = build_deep_mlp(9, 3, 4, 12, 3, torch.Generator().manual_seed(33))
    x = torch.randint(0, 9, (20, 3), generator=torch.Generator().manual_seed(34))
    model(x)
    stats = activation_statistics(model)
    require(len(stats) == 3, "应统计每个 Tanh 输出")
    require(all(item.saturation_ratio is not None for item in stats), "必须包含饱和率")
    tanh_layers = [layer for layer in model.network.layers if isinstance(layer, Tanh)]
    for stat, layer in zip(stats, tanh_layers):
        out = layer.out
        require(out is not None, "Tanh.out 不应为空")
        close(stat.mean, out.mean().item())
        close(stat.std, out.std().item())
        close(stat.minimum, out.min().item())
        close(stat.maximum, out.max().item())
        close(stat.finite_ratio, torch.isfinite(out).float().mean().item())
        close(stat.saturation_ratio, (out.abs() > 0.97).float().mean().item())
    tensors = [(f"tanh_{i}", layer.out) for i, layer in enumerate(model.network.layers) if isinstance(layer, Tanh)]
    figure_has_content(plot_activation_histograms(tensors), minimum_axes=3)
    figure_has_content(plot_activation_summary(stats), minimum_axes=2)


def check_31() -> None:
    model, _ = _deep_model_with_backward(35)
    stats = activation_gradient_statistics(model)
    require(len(stats) == 3, "每个 Tanh 都应有激活梯度统计")
    require(all(item.shape and item.finite_ratio == 1.0 for item in stats), "激活梯度必须 finite 且包含 shape")
    tanh_layers = [layer for layer in model.network.layers if isinstance(layer, Tanh)]
    for stat, layer in zip(stats, tanh_layers):
        gradient = layer.out.grad
        require(gradient is not None and stat.shape == tuple(gradient.shape), "激活梯度统计 shape 错误")
        close(stat.mean, gradient.mean().item())
        close(stat.std, gradient.std().item())
        close(stat.finite_ratio, torch.isfinite(gradient).float().mean().item())
    tensors = [(f"tanh_{i}", layer.out.grad) for i, layer in enumerate(model.network.layers) if isinstance(layer, Tanh)]
    require(all(tensor is not None for _, tensor in tensors), "必须读取 layer.out.grad")
    figure_has_content(plot_activation_gradient_histograms(tensors), minimum_axes=3)


def check_32() -> None:
    known = torch.tensor([[-2.0, 0.0, 2.0], [1.0, 3.0, 5.0]], requires_grad=True)
    known.grad = torch.tensor([[0.5, -1.0, 1.5], [2.0, -0.5, 0.25]])
    constant = torch.ones(2, 3, requires_grad=True)
    constant.grad = torch.full_like(constant, 2.0)
    synthetic = parameter_statistics([("known", known), ("constant", constant)], learning_rate=0.25)
    expected_grad_std = known.grad.std().item()
    expected_data_std = known.std().item()
    close(synthetic[0].data_std, expected_data_std)
    close(synthetic[0].grad_mean, known.grad.mean().item())
    close(synthetic[0].grad_std, expected_grad_std)
    close(synthetic[0].grad_to_data_ratio, expected_grad_std / expected_data_std)
    close(synthetic[0].update_to_data_log10, math.log10((0.25 * known.grad).std().item() / expected_data_std))
    require(synthetic[1].grad_to_data_ratio is None and synthetic[1].update_to_data_log10 is None, "零 data std 必须明确标为 N/A")

    model, _ = _deep_model_with_backward(38)
    named = [(f"p{i}", parameter) for i, parameter in enumerate(model.parameters())]
    stats = parameter_statistics(named, learning_rate=0.1)
    require(len(stats) == len(named), "每个参数都应有统计")
    require(any(item.update_to_data_log10 is not None for item in stats), "应计算 update:data")
    require(all(item.update_to_data_log10 is None or math.isfinite(item.update_to_data_log10) for item in stats), "update:data 不能静默产生 Inf")
    figure_has_content(plot_parameter_gradient_histograms(named))
    history = {
        item.name: [
            item.update_to_data_log10 if item.update_to_data_log10 is not None else -9.0,
            (item.update_to_data_log10 if item.update_to_data_log10 is not None else -9.0) + 0.1,
        ]
        for item in stats[:3]
    }
    result = plot_update_ratios(history, reference=-3.0)
    figure_has_content(result)
    fig = result[0] if isinstance(result, tuple) else result
    require(any(abs(line.get_ydata()[0] + 3.0) < 1e-6 for ax in fig.axes for line in ax.lines if len(line.get_ydata())), "图中应包含 y=-3 参考线")


def check_33() -> None:
    model = build_deep_mlp(7, 3, 3, 8, 2, torch.Generator().manual_seed(41))
    counts = [8, 6, 5, 4, 3, 1, 1]
    y = torch.cat([torch.full((count,), index, dtype=torch.long) for index, count in enumerate(counts)])
    x = torch.randint(0, 7, (len(y), 3), generator=torch.Generator().manual_seed(42))
    payload = zero_initialization_pathology(model, x, y, steps=200, learning_rate=1.0)
    required_keys = {"parameter_grad_norms", "prediction_rows", "marginal_probs", "losses"}
    require(required_keys <= set(payload), f"zero-init 报告缺少 {required_keys - set(payload)}")
    rows = payload["prediction_rows"]
    require(torch.is_tensor(rows) and rows.ndim == 2, "prediction_rows 应为二维 Tensor")
    require(torch.allclose(rows.sum(1), torch.ones(rows.shape[0]), atol=1e-6), "prediction_rows 必须是概率")
    require(torch.allclose(rows, rows[:1].expand_as(rows)), "全零隐藏权重时不同 context 的预测行应相同")
    marginal = payload["marginal_probs"]
    require(torch.is_tensor(marginal) and marginal.ndim == 1, "marginal_probs 应为一维 Tensor")
    empirical = torch.bincount(y, minlength=7).float() / len(y)
    require(torch.allclose(marginal, empirical), "marginal_probs 必须由 y 的经验频率计算")
    require((rows.mean(0) - empirical).abs().max().item() < 0.04, "全零网络训练后应退化为接近 unigram")
    grad_norms = payload["parameter_grad_norms"]
    require(isinstance(grad_norms, dict), "parameter_grad_norms 应为字典")
    expected_names = [f"p{index}" for index in range(len(model.parameters()))]
    require(list(grad_norms) == expected_names, "梯度范数键必须按 model.parameters() 顺序命名 p0,p1,...")
    require(all(abs(float(grad_norms[name])) < 1e-10 for name in expected_names[:-1]), "首轮隐藏/权重/gamma 梯度必须为零")
    require(float(grad_norms[expected_names[-1]]) > 1e-6, "首轮应只有最终 BN beta 获得非零梯度")
    require(len(payload["losses"]) == 200 and payload["losses"][-1] < payload["losses"][0], "zero-init 训练记录不完整")
    zero_figure = plot_zero_init_pathology(payload)
    figure_has_content(zero_figure, minimum_axes=2)

    configurations = {}
    for offset, name in enumerate(("gain_1", "gain_5_3", "gain_3", "batchnorm")):
        configurations[name] = DiagnosticRun(
            name=name,
            activations=[TensorStats(name="h", mean=0.0, std=0.4 + 0.1 * offset, minimum=-1.0, maximum=1.0, finite_ratio=1.0, saturation_ratio=0.01 * offset)],
            activation_gradients=[GradientStats(name="dh", mean=0.0, std=1e-3 * (offset + 1), finite_ratio=1.0, shape=(8, 8))],
            parameters=[ParameterStats(name="W", data_std=1.0, grad_mean=0.0, grad_std=0.01 * (offset + 1), grad_to_data_ratio=0.01 * (offset + 1), update_to_data_log10=-3.0 + 0.1 * offset)],
            update_history={"W": [-3.2 + 0.1 * offset, -3.0 + 0.1 * offset]},
        )
    dashboard = plot_diagnostics_dashboard(configurations)
    figure_has_content(dashboard, minimum_axes=4)
    fig, artist_counts = _figure_artist_count(dashboard)
    require(all(count > 0 for count in artist_counts[:4]), "dashboard 四个诊断面板都必须实际绘图")
    legend_labels = {
        text.get_text()
        for axis in fig.axes
        for legend in ([axis.get_legend()] if axis.get_legend() is not None else [])
        for text in legend.get_texts()
    }
    require(set(configurations) <= legend_labels, "dashboard 图例必须包含 gain=1、5/3、3 与 BatchNorm 四配置")
