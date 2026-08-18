from __future__ import annotations

import json
import math
import random
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from checks.common import close, distinct_tensors, figure_has_content, finite_tensor, require
from makemore.data import build_context_dataset, build_vocab, load_words, split_words
from makemore.mlp import (
    MLPConfig,
    MLPParameters,
    clone_parameters,
    embedding_lookup,
    flatten_embeddings,
    init_mlp,
    mlp_forward,
    parameter_count,
    sample_mlp,
    stable_cross_entropy,
    tensor_layout,
)
from makemore.plotting import (
    plot_capacity,
    plot_embeddings,
    plot_entropy_distribution,
    plot_frequency_comparison,
    plot_lr_range,
    plot_mlp_training,
)
from makemore.reporting import save_json
from makemore.training import (
    ExperimentResult,
    character_frequencies,
    evaluate_mlp,
    fit_mlp,
    lr_range_test,
    mlp_train_step,
    moving_average,
    prediction_entropy,
    sample_minibatch,
)


OFFICIAL_NAMES = Path(__file__).resolve().parents[1] / "data" / "names.txt"


def tiny_dataset(block_size: int = 3):
    words = ["emma", "ava", "liam", "noah", "mia"]
    vocab = build_vocab(words)
    x, y = build_context_dataset(words, vocab, block_size)
    return words, vocab, x, y


def check_11() -> None:
    words = [f"name{i}" for i in range(20)]
    original_words = list(words)
    global_random_state = random.getstate()
    tr1, de1, te1 = split_words(words, seed=42)
    require(words == original_words, "split_words 不得原地打乱输入")
    require(random.getstate() == global_random_state, "split_words 不得污染全局 random 状态")
    tr2, de2, te2 = split_words(words, seed=42)
    require((tr1, de1, te1) == (tr2, de2, te2), "相同 seed 划分必须可复现")
    require(len(tr1) == 16 and len(de1) == 2 and len(te1) == 2, "80/10/10 数量不正确")
    require(not (set(tr1) & set(de1) or set(tr1) & set(te1) or set(de1) & set(te1)), "集合发生泄漏")
    require(sorted(tr1 + de1 + te1) == sorted(words), "划分后丢失或重复单词")
    try:
        split_words(words, train_ratio=0.9, dev_ratio=0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("非法 split ratio 必须抛出 ValueError")


def check_12() -> None:
    vocab = build_vocab(["emma"])
    x, y = build_context_dataset(["emma"], vocab, block_size=3)
    require(tuple(x.shape) == (5, 3) and tuple(y.shape) == (5,), "emma 应产生 5 个样本")
    expected_first = torch.tensor([0, 0, 0])
    expected_last = torch.tensor([vocab.stoi["m"], vocab.stoi["m"], vocab.stoi["a"]])
    require(torch.equal(x[0], expected_first), "首个上下文应全是边界符")
    require(torch.equal(x[-1], expected_last) and y[-1].item() == 0, "最后一个样本应预测边界符")

    if OFFICIAL_NAMES.exists():
        words = load_words(OFFICIAL_NAMES)
        official_vocab = build_vocab(words)
        train_words, dev_words, test_words = split_words(words, seed=42)
        xtr, ytr = build_context_dataset(train_words, official_vocab, block_size=3)
        xdev, ydev = build_context_dataset(dev_words, official_vocab, block_size=3)
        xtest, ytest = build_context_dataset(test_words, official_vocab, block_size=3)
        require(tuple(xtr.shape) == (182441, 3) and tuple(ytr.shape) == (182441,), "官方 train shape 错误")
        require(tuple(xdev.shape) == (22902, 3) and tuple(ydev.shape) == (22902,), "官方 dev shape 错误")
        require(tuple(xtest.shape) == (22803, 3) and tuple(ytest.shape) == (22803,), "官方 test shape 错误")


def check_13() -> None:
    C = torch.arange(35, dtype=torch.float32).view(7, 5)
    x = torch.tensor([[0, 1, 2], [3, 4, 5]])
    emb = embedding_lookup(C, x)
    require(tuple(emb.shape) == (2, 3, 5), "embedding shape 不正确")
    require(torch.equal(emb[0, 1], C[1]), "embedding lookup 值不正确")
    flat = flatten_embeddings(emb)
    require(tuple(flat.shape) == (2, 15), "只能展平 context 和 embedding 轴")
    require(torch.equal(flat[0], emb[0].reshape(-1)), "展平顺序错误")
    require(
        flat.untyped_storage().data_ptr() == emb.untyped_storage().data_ptr(),
        "连续 embedding 的 flatten 应是共享 storage 的 view",
    )
    emb_layout = tensor_layout(emb)
    flat_layout = tensor_layout(flat)
    require(emb_layout.shape == tuple(emb.shape), "layout.shape 错误")
    require(emb_layout.stride == tuple(emb.stride()), "layout.stride 错误")
    require(emb_layout.is_contiguous == emb.is_contiguous(), "layout contiguous 标记错误")
    require(flat_layout.shape == (2, 15) and flat_layout.is_contiguous, "flat layout 错误")


def check_14() -> None:
    config = MLPConfig(vocab_size=27, block_size=3, n_embd=10, n_hidden=200)
    params = init_mlp(config, torch.Generator().manual_seed(2147483647))
    require(parameter_count(params) == 11897, "官方配置参数量应为 11897")
    tensors = params.tensors()
    require(len(tensors) == 5, "参数列表应为 C/W1/b1/W2/b2")
    distinct_tensors(tensors)
    require(all(item.requires_grad and item.is_leaf and item.grad_fn is None for item in tensors), "所有参数都应是 requires_grad 叶张量")
    clone = clone_parameters(params)
    cloned_tensors = clone.tensors()
    require(all(torch.equal(a, b) for a, b in zip(tensors, cloned_tensors)), "clone 数值应相同")
    require(all(a is not b for a, b in zip(tensors, cloned_tensors)), "clone 必须创建新对象")
    require(
        all(item.requires_grad and item.is_leaf and item.grad_fn is None for item in cloned_tensors),
        "clone 必须是新的 requires_grad 叶张量",
    )
    require(
        all(a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr() for a, b in zip(tensors, cloned_tensors)),
        "clone 不得与原参数共享 storage",
    )
    original_first = tensors[0].detach().clone()
    with torch.no_grad():
        cloned_tensors[0].add_(1.0)
    require(torch.equal(tensors[0], original_first), "修改 clone 不得改变原参数")


def check_15() -> None:
    config = MLPConfig(vocab_size=7, block_size=3, n_embd=4, n_hidden=8)
    params = init_mlp(config, torch.Generator().manual_seed(7))
    x = torch.randint(0, 7, (5, 3), generator=torch.Generator().manual_seed(1))
    logits, cache = mlp_forward(params, x, return_cache=True)
    require(tuple(cache.emb.shape) == (5, 3, 4), "emb shape 错误")
    require(tuple(cache.embcat.shape) == (5, 12), "embcat shape 错误")
    require(tuple(cache.hpreact.shape) == tuple(cache.h.shape) == (5, 8), "hidden shape 错误")
    require(tuple(logits.shape) == tuple(cache.logits.shape) == (5, 7), "logits shape 错误")
    require(torch.allclose(cache.h, torch.tanh(cache.hpreact)), "h 必须是 tanh(hpreact)")
    expected_emb = params.C[x]
    expected_embcat = expected_emb.reshape(5, -1)
    expected_hpreact = expected_embcat @ params.W1 + params.b1
    expected_h = torch.tanh(expected_hpreact)
    expected_logits = expected_h @ params.W2 + params.b2
    require(torch.equal(cache.emb, expected_emb), "cache.emb 值错误")
    require(torch.equal(cache.embcat, expected_embcat), "cache.embcat 值或展平顺序错误")
    require(torch.allclose(cache.hpreact, expected_hpreact), "第一层 Linear 值错误")
    require(torch.allclose(logits, expected_logits), "输出层 logits 值错误")
    require(torch.allclose(mlp_forward(params, x), logits), "不返回 cache 时 logits 必须一致")


def check_16() -> None:
    logits = torch.tensor([[1000.0, 999.0, -1000.0], [-1000.0, 1000.0, 999.0]])
    targets = torch.tensor([0, 2])
    loss = stable_cross_entropy(logits, targets)
    finite_tensor(loss)
    require(torch.allclose(loss, F.cross_entropy(logits, targets)), "稳定 CE 与 PyTorch 不一致")
    require(bool(torch.isinf(logits.exp()).any()), "测试 logits 应能展示直接 exp 的溢出")


def check_17() -> None:
    _, _, x, y = tiny_dataset()
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    xb1, yb1 = sample_minibatch(x, y, 8, g1)
    xb2, yb2 = sample_minibatch(x, y, 8, g2)
    require(torch.equal(xb1, xb2) and torch.equal(yb1, yb2), "minibatch 必须可复现")
    config = MLPConfig(vocab_size=int(y.max()) + 1, block_size=3, n_embd=3, n_hidden=12)
    params = init_mlp(config, torch.Generator().manual_seed(6))
    reference = clone_parameters(params)
    reference_loss = F.cross_entropy(mlp_forward(reference, xb1), yb1)
    reference_grads = torch.autograd.grad(reference_loss, reference.tensors())
    learning_rate = 0.01
    expected_values = [
        parameter.detach() - learning_rate * gradient.detach()
        for parameter, gradient in zip(reference.tensors(), reference_grads)
    ]
    loss = mlp_train_step(params, xb1, yb1, learning_rate=learning_rate)
    require(math.isfinite(loss), "train_step 必须返回有限 loss")
    close(loss, reference_loss.item())
    require(all(item.grad is not None for item in params.tensors()), "所有参数应得到梯度")
    for actual, expected, expected_grad in zip(params.tensors(), expected_values, reference_grads):
        require(torch.allclose(actual, expected), "参数必须严格按 parameter -= lr*grad 更新")
        require(torch.allclose(actual.grad, expected_grad), "train_step 返回时应保留本轮正确梯度")


def check_18() -> None:
    _, _, x, y = tiny_dataset()
    config = MLPConfig(vocab_size=int(y.max()) + 1, block_size=3, n_embd=2, n_hidden=8)
    params = init_mlp(config, torch.Generator().manual_seed(8))
    original_values = [item.detach().clone() for item in params.tensors()]
    result = lr_range_test(
        params,
        x,
        y,
        torch.Generator().manual_seed(9),
        batch_size=8,
        steps=8,
        min_exponent=-3,
        max_exponent=-1,
    )
    require(
        len(result.losses) == len(result.learning_rates) == len(result.log10_learning_rates) == 8,
        "LR range 三组记录必须等长",
    )
    require(all(a < b for a, b in zip(result.learning_rates, result.learning_rates[1:])), "学习率应递增")
    require(all(math.isfinite(item) for item in result.losses), "LR loss 必须有限")
    require(all(torch.equal(before, after) for before, after in zip(original_values, params.tensors())), "LR sweep 不得修改传入参数")
    close(result.log10_learning_rates[0], -3.0)
    close(result.log10_learning_rates[-1], -1.0)
    for learning_rate, exponent in zip(result.learning_rates, result.log10_learning_rates):
        close(learning_rate, 10**exponent)
    figure_has_content(plot_lr_range(result))


def check_19() -> None:
    words, vocab, _, _ = tiny_dataset()
    train_words, dev_words, test_words = split_words(words, seed=17, train_ratio=0.6, dev_ratio=0.2)
    require(not (set(train_words) & set(dev_words)), "必须先按单词切分，不能切分展开后的 context")
    require(len(test_words) == 1, "tiny split 应保留独立 test word")
    xtr, ytr = build_context_dataset(train_words, vocab, block_size=3)
    xdev, ydev = build_context_dataset(dev_words, vocab, block_size=3)
    config = MLPConfig(vocab_size=vocab.size, block_size=3, n_embd=3, n_hidden=12)
    params = init_mlp(config, torch.Generator().manual_seed(10))
    history = fit_mlp(
        params,
        xtr,
        ytr,
        xdev,
        ydev,
        torch.Generator().manual_seed(11),
        steps=12,
        batch_size=8,
        learning_rate=0.03,
        report_every=4,
    )
    require(history.steps == list(range(1, 13)), "history.steps 应为 1..训练更新次数")
    require(len(history.losses) == 12, "训练历史长度错误")
    require(moving_average([1, 2, 3, 4, 5], 2) == [1.5, 3.5, 5.0], "moving_average 应按不重叠窗口并保留尾块")
    require(
        len(history.smoothed_steps) == len(history.smoothed_losses) > 0,
        "平滑 loss 必须带有等长横坐标",
    )
    require(history.smoothed_steps[-1] == 12, "平滑曲线最后一个横坐标应覆盖最终更新")
    require(
        history.evaluation_steps == [4, 8, 12]
        and len(history.train_losses) == len(history.dev_losses) == 3,
        "train/dev 评估必须在 report_every 和最终更新处记录并严格对齐",
    )
    evaluated = evaluate_mlp(params, xdev, ydev)
    close(evaluated, F.cross_entropy(mlp_forward(params, xdev), ydev).item())
    figure_has_content(plot_mlp_training(history))
    results = [
        ExperimentResult("small", parameter_count(params), 2.5, 2.7, None, {"hidden": 12}),
        ExperimentResult("large", parameter_count(params) * 2, 2.2, 2.4, None, {"hidden": 24}),
    ]
    figure_has_content(plot_capacity(results))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "experiments.json"
        returned = save_json(results, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(returned == path and [item["name"] for item in payload] == ["small", "large"], "save_json 必须支持 dataclass 序列")


def check_20() -> None:
    _, vocab, x, y = tiny_dataset()
    config = MLPConfig(vocab_size=vocab.size, block_size=3, n_embd=2, n_hidden=10)
    params = init_mlp(config, torch.Generator().manual_seed(12))
    samples = sample_mlp(
        params,
        block_size=3,
        itos=vocab.itos,
        generator=torch.Generator().manual_seed(13),
        count=4,
        max_length=8,
    )
    require(len(samples) == 4 and all(len(item) <= 8 for item in samples), "采样数量/最大长度错误")

    cycle_vocab = build_vocab(["a"])
    cycle_params = MLPParameters(
        C=torch.tensor([[-1.0], [1.0]], requires_grad=True),
        W1=torch.tensor([[1.0]], requires_grad=True),
        b1=torch.zeros(1, requires_grad=True),
        W2=torch.tensor([[100.0, -100.0]], requires_grad=True),
        b2=torch.zeros(2, requires_grad=True),
    )
    cycle_1 = sample_mlp(
        cycle_params,
        block_size=1,
        itos=cycle_vocab.itos,
        generator=torch.Generator().manual_seed(21),
        count=3,
        max_length=5,
    )
    cycle_2 = sample_mlp(
        cycle_params,
        block_size=1,
        itos=cycle_vocab.itos,
        generator=torch.Generator().manual_seed(21),
        count=3,
        max_length=5,
    )
    require(cycle_1 == cycle_2 == ["a", "a", "a"], "采样必须更新滚动 context、遇边界停止且固定 seed 可复现")

    always_a = MLPParameters(
        C=torch.zeros((2, 1), requires_grad=True),
        W1=torch.zeros((1, 1), requires_grad=True),
        b1=torch.zeros(1, requires_grad=True),
        W2=torch.zeros((1, 2), requires_grad=True),
        b2=torch.tensor([-100.0, 100.0], requires_grad=True),
    )
    limited = sample_mlp(
        always_a,
        block_size=1,
        itos=cycle_vocab.itos,
        generator=torch.Generator().manual_seed(22),
        count=2,
        max_length=4,
    )
    require(limited == ["aaaa", "aaaa"], "采不到边界时必须严格遵守 max_length")

    exact_frequency = character_frequencies(torch.tensor([0, 1, 1, 3]), vocab_size=4)
    require(torch.allclose(exact_frequency, torch.tensor([0.25, 0.5, 0.0, 0.25])), "字符频率公式错误")
    ignored_frequency = character_frequencies(torch.tensor([0, 1, 1, 3]), vocab_size=4, ignore_index=0)
    require(torch.allclose(ignored_frequency, torch.tensor([0.0, 2 / 3, 0.0, 1 / 3])), "排除边界后的频率必须重新归一化")

    observed = character_frequencies(y, vocab.size, ignore_index=0)
    generated_indices = torch.tensor([vocab.stoi.get(ch, 0) for word in samples for ch in word] or [0])
    generated = character_frequencies(generated_indices, vocab.size, ignore_index=0)
    close(observed.sum().item(), 1.0)
    require(close_or_zero(generated), "生成字符频率应归一化；全空样本时允许全零")
    logits = mlp_forward(params, x[:6])
    entropy = prediction_entropy(logits)
    require(tuple(entropy.shape) == (6,), "每个样本应有一个 entropy")
    expected_entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    require(torch.allclose(entropy, expected_entropy), "预测熵公式错误")
    uniform_entropy = prediction_entropy(torch.zeros((2, 4)))
    require(torch.allclose(uniform_entropy, torch.full((2,), math.log(4.0))), "四类均匀预测熵应为 log(4)")
    figure_has_content(plot_embeddings(params.C, vocab.itos))
    figure_has_content(plot_frequency_comparison(observed, generated, vocab.itos))
    figure_has_content(plot_entropy_distribution(entropy))


def close_or_zero(frequencies: torch.Tensor) -> bool:
    total = frequencies.sum().item()
    return math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6) or math.isclose(total, 0.0, abs_tol=1e-6)
