from __future__ import annotations

import ast
import inspect
import math
import textwrap
from dataclasses import fields

import torch
import torch.nn.functional as F

from checks.common import close, figure_has_content, finite_tensor, require
from makemore.manual_backprop import (
    ManualForwardCache,
    ManualParameters,
    backward_cross_entropy_atomic,
    backward_cross_entropy_fused,
    batchnorm_backward_atomic,
    batchnorm_backward_fused,
    compare_grad,
    calibrate_manual_batchnorm,
    embedding_backward,
    init_manual_parameters,
    linear_backward,
    manual_backward,
    manual_forward,
    manual_inference_logits,
    manual_train_step,
    tanh_backward,
)
from makemore.plotting import (
    plot_dlogits,
    plot_gradient_differences,
    plot_manual_training,
)


def clone_manual(parameters: ManualParameters) -> ManualParameters:
    return ManualParameters(**{
        field.name: getattr(parameters, field.name).detach().clone().requires_grad_(True)
        for field in fields(parameters)
    })


def debug_batch(seed: int = 101):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 27, (8, 3), generator=g)
    y = torch.randint(0, 27, (8,), generator=g)
    return x, y


def _reference_init(
    vocab_size: int,
    block_size: int,
    n_embd: int,
    n_hidden: int,
    seed: int,
) -> ManualParameters:
    """独立固定官方 debug 初始化，避免全零值掩盖错误梯度。"""
    generator = torch.Generator().manual_seed(seed)
    parameters = ManualParameters(
        C=torch.randn((vocab_size, n_embd), generator=generator),
        W1=torch.randn((n_embd * block_size, n_hidden), generator=generator)
        * (5 / 3)
        / math.sqrt(n_embd * block_size),
        b1=torch.randn(n_hidden, generator=generator) * 0.1,
        W2=torch.randn((n_hidden, vocab_size), generator=generator) * 0.1,
        b2=torch.randn(vocab_size, generator=generator) * 0.1,
        bngain=torch.randn((1, n_hidden), generator=generator) * 0.1 + 1.0,
        bnbias=torch.randn((1, n_hidden), generator=generator) * 0.1,
    )
    for parameter in parameters.tensors():
        parameter.requires_grad_(True)
    return parameters


def _reference_forward(
    parameters: ManualParameters,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-5,
) -> ManualForwardCache:
    """不调用学生 manual_forward 的独立原子计算图。"""
    n = x.shape[0]
    emb = parameters.C[x]
    embcat = emb.reshape(emb.shape[0], -1)
    hprebn = embcat @ parameters.W1 + parameters.b1
    bnmean = hprebn.mean(0, keepdim=True)
    bndiff = hprebn - bnmean
    bndiff2 = bndiff**2
    bnvar = bndiff2.sum(0, keepdim=True) / (n - 1)
    bnvar_inv = (bnvar + eps) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = parameters.bngain * bnraw + parameters.bnbias
    h = torch.tanh(hpreact)
    logits = h @ parameters.W2 + parameters.b2
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdim=True)
    counts_sum_inv = counts_sum**-1
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[torch.arange(n), y].mean()
    cache = ManualForwardCache(
        emb=emb,
        embcat=embcat,
        hprebn=hprebn,
        bnmean=bnmean,
        bndiff=bndiff,
        bndiff2=bndiff2,
        bnvar=bnvar,
        bnvar_inv=bnvar_inv,
        bnraw=bnraw,
        hpreact=hpreact,
        h=h,
        logits=logits,
        logit_maxes=logit_maxes,
        norm_logits=norm_logits,
        counts=counts,
        counts_sum=counts_sum,
        counts_sum_inv=counts_sum_inv,
        probs=probs,
        logprobs=logprobs,
        loss=loss,
    )
    for field in fields(cache):
        tensor = getattr(cache, field.name)
        if tensor.requires_grad and not tensor.is_leaf:
            tensor.retain_grad()
    return cache


def _assert_no_grad_side_effects(parameters: ManualParameters, cache: ManualForwardCache | None = None) -> None:
    require(all(parameter.grad is None for parameter in parameters.tensors()), "学生手写路径不得填充 parameter.grad")
    if cache is not None:
        require(all(getattr(cache, field.name).grad is None for field in fields(cache)), "学生手写路径不得读取/填充 cache .grad")


def _forbid_autograd_calls(function) -> None:
    """课程级静态护栏；运行时 no-grad/无副作用检查仍是主验证。"""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Attribute) and target.attr == "backward":
            raise AssertionError(f"{function.__name__} 不得调用 backward()")
        if isinstance(target, ast.Attribute) and target.attr == "grad":
            owner = target.value
            if isinstance(owner, ast.Attribute) and owner.attr == "autograd":
                raise AssertionError(f"{function.__name__} 不得调用 autograd.grad()")


def check_34() -> None:
    seed = 100
    params = init_manual_parameters(27, 3, 10, 64, torch.Generator().manual_seed(seed))
    expected_params = _reference_init(27, 3, 10, 64, seed)
    require(sum(item.nelement() for item in params.tensors()) == 4137, "debug 网络参数量应为 4137")
    require(all(item.requires_grad and item.is_leaf for item in params.tensors()), "参数必须是 requires_grad 叶张量")
    for field in fields(params):
        actual = getattr(params, field.name)
        expected = getattr(expected_params, field.name)
        require(actual.shape == expected.shape and torch.equal(actual, expected), f"{field.name} 未按固定 seed 的 debug 公式初始化")
    require(torch.count_nonzero(params.b1) and torch.count_nonzero(params.b2), "debug bias 必须非零，不能掩盖反传错误")
    require(torch.count_nonzero(params.bnbias) and not torch.equal(params.bngain, torch.ones_like(params.bngain)), "BN gain/bias 必须使用非退化小随机初始化")
    x, y = debug_batch()
    cache = manual_forward(params, x, y)
    expected_cache = _reference_forward(clone_manual(params), x, y)
    require(tuple(cache.emb.shape) == (8, 3, 10), "emb shape 错误")
    require(tuple(cache.embcat.shape) == (8, 30), "embcat shape 错误")
    require(tuple(cache.hprebn.shape) == tuple(cache.h.shape) == (8, 64), "hidden shape 错误")
    require(tuple(cache.logits.shape) == (8, 27), "logits shape 错误")
    for field in fields(cache):
        actual = getattr(cache, field.name)
        expected = getattr(expected_cache, field.name)
        require(actual.shape == expected.shape, f"forward cache {field.name} shape 错误")
        require(torch.allclose(actual, expected, atol=1e-7, rtol=1e-5), f"forward cache {field.name} 与独立公式不一致")
    require(torch.allclose(cache.loss, F.cross_entropy(cache.logits, y), atol=1e-6), "原子 CE forward 与 PyTorch 不一致")
    result = compare_grad("same", torch.ones(2, 3), torch.ones(2, 3))
    require(result.shape_matches and result.exact and result.allclose, "相同梯度比较应通过")
    mismatch = compare_grad("shape", torch.ones(2, 3), torch.ones(6))
    require(not mismatch.shape_matches and not mismatch.allclose, "shape 不同不能因广播被误判")


def _oracle(seed: int = 110):
    base = init_manual_parameters(27, 3, 10, 64, torch.Generator().manual_seed(seed))
    tested_params = clone_manual(base)
    oracle_params = clone_manual(base)
    x, y = debug_batch(seed + 1)
    tested_cache = manual_forward(tested_params, x, y)
    oracle_cache = _reference_forward(oracle_params, x, y)
    oracle_cache.loss.backward()
    _assert_no_grad_side_effects(tested_params, tested_cache)
    return tested_params, x, y, tested_cache, oracle_params, oracle_cache


def check_35() -> None:
    tested_params, _, y, cache, _, oracle_cache = _oracle(110)
    grads = backward_cross_entropy_atomic(cache, y)
    names = [
        "logprobs",
        "probs",
        "counts",
        "counts_sum",
        "counts_sum_inv",
        "norm_logits",
        "logit_maxes",
        "logits",
    ]
    for name in names:
        key = f"d{name}"
        require(key in grads, f"缺少 {key}")
        reference = getattr(oracle_cache, name).grad
        require(reference is not None, f"manual_forward 应对 {name} retain_grad")
        require(grads[key].shape == reference.shape, f"{key} shape 错误")
        require(torch.allclose(grads[key], reference, atol=1e-7, rtol=1e-5), f"{key} 与 autograd 不一致")
    _assert_no_grad_side_effects(tested_params, cache)
    _forbid_autograd_calls(backward_cross_entropy_atomic)


def check_36() -> None:
    logits = torch.randn(6, 9, generator=torch.Generator().manual_seed(120), requires_grad=True)
    targets = torch.tensor([0, 3, 4, 8, 2, 2])
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    manual = backward_cross_entropy_fused(logits.detach(), targets)
    require(manual.shape == logits.shape, "dlogits shape 错误")
    require(torch.allclose(manual, logits.grad, atol=1e-7), "fused CE backward 不正确")
    require(torch.allclose(manual.sum(1), torch.zeros(6), atol=1e-7), "每行 dlogits 和应接近 0")
    require(torch.all(manual[torch.arange(len(targets)), targets] < 0), "正确类别的 CE logit 梯度应为负")
    extreme = torch.tensor([[10000.0, -10000.0, 0.0], [-9000.0, 9000.0, 1.0]])
    finite_tensor(backward_cross_entropy_fused(extreme, torch.tensor([0, 1])), "极端 logits 的 fused CE 梯度必须 finite")
    figure_has_content(plot_dlogits(manual, targets))


def check_37() -> None:
    x = torch.randn(5, 4, generator=torch.Generator().manual_seed(130), requires_grad=True)
    weight = torch.randn(4, 3, generator=torch.Generator().manual_seed(131), requires_grad=True)
    bias = torch.randn(3, generator=torch.Generator().manual_seed(132), requires_grad=True)
    out = x @ weight + bias
    dout = torch.randn_like(out)
    out.backward(dout)
    dx, dweight, dbias = linear_backward(dout, x.detach(), weight.detach())
    require(dx.shape == x.shape and dweight.shape == weight.shape and dbias.shape == bias.shape, "Linear backward shape 错误")
    require(torch.allclose(dx, x.grad) and torch.allclose(dweight, weight.grad) and torch.allclose(dbias, bias.grad), "Linear backward 不正确")
    raw = torch.randn(4, 5, generator=torch.Generator().manual_seed(133), requires_grad=True)
    h = torch.tanh(raw)
    upstream = torch.randn_like(h)
    h.backward(upstream)
    require(torch.allclose(tanh_backward(upstream, h.detach()), raw.grad), "tanh backward 不正确")


def check_38() -> None:
    tested_params, _, _, cache, oracle_params, oracle_cache = _oracle(140)
    require(oracle_cache.hpreact.grad is not None, "oracle hpreact grad 缺失")
    grads = batchnorm_backward_atomic(oracle_cache.hpreact.grad.detach(), cache, tested_params.bngain)
    references = {
        "dhprebn": oracle_cache.hprebn.grad,
        "dbngain": oracle_params.bngain.grad,
        "dbnbias": oracle_params.bnbias.grad,
        "dbnraw": oracle_cache.bnraw.grad,
        "dbnvar_inv": oracle_cache.bnvar_inv.grad,
        "dbnvar": oracle_cache.bnvar.grad,
        "dbndiff2": oracle_cache.bndiff2.grad,
        "dbndiff": oracle_cache.bndiff.grad,
        "dbnmean": oracle_cache.bnmean.grad,
    }
    for name, reference in references.items():
        require(name in grads, f"BN 原子梯度缺少 {name}")
        require(reference is not None, f"oracle {name} 缺失")
        require(grads[name].shape == reference.shape, f"{name} shape 错误，不能依赖广播")
        require(torch.allclose(grads[name], reference, atol=1e-7, rtol=1e-5), f"{name} 不正确")
    _assert_no_grad_side_effects(tested_params, cache)
    _forbid_autograd_calls(batchnorm_backward_atomic)


def check_39() -> None:
    cases = [(7, 5, 1e-5), (3, 4, 1e-3), (16, 2, 1e-8)]
    for case_index, (n, c, eps) in enumerate(cases):
        x = torch.randn(n, c, generator=torch.Generator().manual_seed(150 + case_index), requires_grad=True)
        gamma = (-torch.rand(1, c, generator=torch.Generator().manual_seed(160 + case_index))).requires_grad_()
        beta = torch.randn(1, c, generator=torch.Generator().manual_seed(170 + case_index), requires_grad=True)
        mean = x.mean(0, keepdim=True)
        var = ((x - mean) ** 2).sum(0, keepdim=True) / (n - 1)
        invstd = (var + eps) ** -0.5
        xhat = (x - mean) * invstd
        out = gamma * xhat + beta
        dout = torch.randn_like(out)
        out.backward(dout)
        dx, dgamma, dbeta = batchnorm_backward_fused(dout, xhat.detach(), gamma.detach(), invstd.detach())
        require(dx.shape == x.shape, "fused BN dx shape 错误")
        require(dgamma.shape == gamma.shape and dbeta.shape == beta.shape, "fused BN gamma/beta 梯度 shape 错误")
        require(torch.allclose(dx, x.grad, atol=1e-6, rtol=1e-5), f"fused BN dx 不正确: case {case_index}")
        require(torch.allclose(dgamma, gamma.grad, atol=1e-6, rtol=1e-5), f"fused BN dgamma 不正确: case {case_index}")
        require(torch.allclose(dbeta, beta.grad, atol=1e-6, rtol=1e-5), f"fused BN dbeta 不正确: case {case_index}")
        require(torch.allclose(dx.sum(0), torch.zeros(c), atol=2e-6), "BN dx 沿 batch 求和应接近 0")


def check_40() -> None:
    indices = torch.tensor([[1, 2, 1], [0, 1, 2]])
    demb = torch.arange(18, dtype=torch.float32).view(2, 3, 3)
    dC = embedding_backward(demb, indices, vocab_size=4)
    expected = torch.zeros(4, 3)
    expected.index_add_(0, indices.reshape(-1), demb.reshape(-1, 3))
    require(torch.equal(dC, expected), "重复 embedding index 的梯度必须累加")
    demb64 = demb.double()
    result64 = embedding_backward(demb64, indices, vocab_size=4)
    require(result64.dtype == demb64.dtype and result64.device == demb64.device, "embedding backward 必须保留 dtype/device")


def check_41() -> None:
    tested_params, x, y, cache, oracle_params, oracle_cache = _oracle(160)
    grads = manual_backward(tested_params, cache, x, y)
    comparisons = []
    for field in fields(tested_params):
        name = field.name
        require(name in grads, f"manual_backward 缺少 {name}")
        reference = getattr(oracle_params, name).grad
        comparison = compare_grad(name, grads[name], reference)
        require(comparison.shape_matches and comparison.allclose, f"{name} 梯度不正确")
        comparisons.append(comparison)
    node_references = {
        "dlogits": oracle_cache.logits.grad,
        "dh": oracle_cache.h.grad,
        "dhpreact": oracle_cache.hpreact.grad,
        "dhprebn": oracle_cache.hprebn.grad,
        "dembcat": oracle_cache.embcat.grad,
        "demb": oracle_cache.emb.grad,
    }
    for name, reference in node_references.items():
        require(name in grads, f"manual_backward 缺少关键节点 {name}")
        require(reference is not None and grads[name].shape == reference.shape, f"{name} shape 错误")
        require(torch.allclose(grads[name], reference, atol=1e-7, rtol=1e-5), f"{name} 梯度不正确")
    _assert_no_grad_side_effects(tested_params, cache)
    _forbid_autograd_calls(manual_backward)
    figure_has_content(plot_gradient_differences(comparisons))


def check_42() -> None:
    base = init_manual_parameters(9, 3, 4, 12, torch.Generator().manual_seed(170))
    manual_params = clone_manual(base)
    auto_params = clone_manual(base)
    g = torch.Generator().manual_seed(171)
    x = torch.randint(0, 9, (16, 3), generator=g)
    y = torch.randint(0, 9, (16,), generator=g)
    manual_loss = manual_train_step(manual_params, x, y, learning_rate=0.03)
    _assert_no_grad_side_effects(manual_params)
    cache = _reference_forward(auto_params, x, y)
    auto_loss = cache.loss.item()
    cache.loss.backward()
    with torch.no_grad():
        for parameter in auto_params.tensors():
            parameter -= 0.03 * parameter.grad
    close(manual_loss, auto_loss)
    for manual_parameter, auto_parameter in zip(manual_params.tensors(), auto_params.tensors()):
        require(torch.allclose(manual_parameter, auto_parameter, atol=1e-6), "一步手写更新与 autograd 不一致")

    losses = [manual_train_step(manual_params, x, y, learning_rate=0.03) for _ in range(40)]
    _assert_no_grad_side_effects(manual_params)
    require(sum(losses[-5:]) / 5 < sum(losses[:5]) / 5, "短程手写训练 loss 应下降")
    _forbid_autograd_calls(manual_train_step)

    stats = calibrate_manual_batchnorm(manual_params, x)
    emb = manual_params.C[x]
    embcat = emb.reshape(emb.shape[0], -1)
    hprebn = embcat @ manual_params.W1 + manual_params.b1
    expected_mean = hprebn.mean(0, keepdim=True)
    expected_var = hprebn.var(0, keepdim=True, correction=1)
    require(stats.mean.shape == expected_mean.shape and torch.allclose(stats.mean, expected_mean), "manual BN calibration mean 错误")
    require(stats.var.shape == expected_var.shape and torch.allclose(stats.var, expected_var), "manual BN calibration sample variance 错误")
    single = manual_inference_logits(manual_params, x[:1], stats)
    companions = manual_inference_logits(manual_params, x[:5], stats)[:1]
    require(single.shape == (1, 9) and torch.isfinite(single).all(), "固定 BN 统计的单样本推理必须 finite")
    require(torch.allclose(single, companions, atol=1e-6), "manual eval 中同一样本不应依赖 companions")
    reference_h = torch.tanh(manual_params.bngain * (hprebn[:1] - stats.mean) * (stats.var + 1e-5) ** -0.5 + manual_params.bnbias)
    reference_logits = reference_h @ manual_params.W2 + manual_params.b2
    require(torch.allclose(single, reference_logits, atol=1e-6), "manual inference logits 未使用校准统计")
    figure_has_content(plot_manual_training(losses))
