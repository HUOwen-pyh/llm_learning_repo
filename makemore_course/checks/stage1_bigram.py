from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from checks.common import close, figure_has_content, finite_tensor, require
from makemore.bigram import (
    bigram_nll,
    count_bigrams,
    count_trigrams,
    manual_softmax,
    neural_bigram_logits,
    neural_bigram_loss,
    normalize_counts,
    normalize_trigram_counts,
    perplexity,
    probability_difference,
    sample_bigram,
    smoothing_search,
    train_neural_bigram,
    trigram_nll,
)
from makemore.data import build_bigram_pairs, build_vocab, corpus_stats, load_words
from makemore.plotting import (
    plot_bigram_heatmap,
    plot_name_lengths,
    plot_next_char_distribution,
    plot_probability_difference,
    plot_smoothing_search,
)
from makemore.reporting import save_lines


WORDS = ["ab", "ac"]
OFFICIAL_NAMES = Path(__file__).resolve().parents[1] / "data" / "names.txt"


def reference_trigram_probs(words, vocab, smoothing: float) -> torch.Tensor:
    counts = torch.zeros((vocab.size, vocab.size, vocab.size), dtype=torch.float64)
    for word in words:
        context = [0, 0]
        for char in word + vocab.boundary:
            target = vocab.stoi[char]
            counts[context[0], context[1], target] += 1
            context = [context[1], target]
    probs = counts + smoothing
    return probs / probs.sum(dim=-1, keepdim=True)


def reference_trigram_nll(probs, words, vocab) -> float:
    log_likelihoods = []
    for word in words:
        context = [0, 0]
        for char in word + vocab.boundary:
            target = vocab.stoi[char]
            log_likelihoods.append(probs[context[0], context[1], target].log())
            context = [context[1], target]
    return -torch.stack(log_likelihoods).mean().item()


def check_01() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "words.txt"
        path.write_text(" Alice \n\nBOB\ncarol\n", encoding="utf-8")
        words = load_words(path)
    require(words == ["alice", "bob", "carol"], "应清除空行、空白并转小写")
    stats = corpus_stats(words)
    require(stats["count"] == 3, "count 不正确")
    require(stats["min_length"] == 3 and stats["max_length"] == 5, "长度统计不正确")
    close(stats["mean_length"], 13 / 3)
    require(stats["characters"] == sorted(set("alicebobcarol")), "字符集合应稳定排序")


def check_02() -> None:
    vocab = build_vocab(WORDS)
    require(vocab.stoi["."] == 0, "边界符必须为 0")
    require(list(vocab.stoi) == [".", "a", "b", "c"], "其余字符应排序")
    encoded = vocab.encode("cab")
    require(vocab.decode(encoded) == "cab", "encode/decode 必须可逆")
    require(vocab.size == 4, "词表大小不正确")


def check_03() -> None:
    vocab = build_vocab(["ab"])
    x, y = build_bigram_pairs(["ab"], vocab)
    expected_x = torch.tensor([0, vocab.stoi["a"], vocab.stoi["b"]])
    expected_y = torch.tensor([vocab.stoi["a"], vocab.stoi["b"], 0])
    require(x.dtype == torch.long and y.dtype == torch.long, "x/y 必须是 long")
    require(torch.equal(x, expected_x) and torch.equal(y, expected_y), "bigram 对不正确")


def check_04() -> None:
    vocab = build_vocab(WORDS)
    counts = count_bigrams(WORDS, vocab)
    require(counts.dtype == torch.long, "计数矩阵必须是 long")
    require(tuple(counts.shape) == (4, 4), "计数矩阵 shape 不正确")
    require(counts.sum().item() == 6, "总 bigram 数应为 sum(len(word)+1)")
    require(counts[0, vocab.stoi["a"]].item() == 2, "重复 bigram 没有累加")

    if OFFICIAL_NAMES.exists():
        official_words = load_words(OFFICIAL_NAMES)
        official_vocab = build_vocab(official_words)
        official_counts = count_bigrams(official_words, official_vocab)
        require(len(official_words) == 32033, "官方 names.txt 应包含 32033 个名字")
        require(official_vocab.size == 27, "官方词表应为边界符加 26 个字母")
        require(official_counts.sum().item() == 228146, "官方 bigram 总数应为 228146")
        require(
            official_counts[official_vocab.stoi["."], official_vocab.stoi["a"]].item() == 4410,
            "官方 .->a 计数应为 4410",
        )
        require(
            official_counts[official_vocab.stoi["a"], official_vocab.stoi["."]].item() == 6640,
            "官方 a->. 计数应为 6640",
        )


def check_05() -> None:
    counts = torch.tensor([[0, 1], [3, 0]], dtype=torch.long)
    probs = normalize_counts(counts, smoothing=1.0)
    require(tuple(probs.shape) == (2, 2), "归一化不应改变 shape")
    require(bool((probs > 0).all()), "smoothing 后概率必须为正")
    require(torch.allclose(probs.sum(1), torch.ones(2)), "每行概率和必须为 1")
    require(torch.allclose(probs[0], torch.tensor([0.5, 0.5])), "第一行归一化错误")
    require(torch.allclose(probs[1], torch.tensor([0.8, 0.2])), "第二行归一化错误")


def check_06() -> None:
    probs = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    x = torch.tensor([0, 1])
    y = torch.tensor([1, 0])
    expected = (-math.log(0.75) - math.log(0.5)) / 2
    nll = bigram_nll(probs, x, y)
    close(nll.item(), expected)
    close(perplexity(nll), math.exp(expected))
    impossible = bigram_nll(
        torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        torch.tensor([0]),
        torch.tensor([1]),
    )
    require(bool(torch.isinf(impossible)), "真实目标概率为 0 时 NLL 应为正无穷")


def check_07() -> None:
    vocab = build_vocab(["a"])
    probs = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    require(sample_bigram(probs, vocab, g1, count=3) == ["a", "a", "a"], "采样链不正确")
    require(sample_bigram(probs, vocab, g2, count=3) == ["a", "a", "a"], "固定 seed 应可复现")
    never_stops = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    limited = sample_bigram(
        never_stops,
        vocab,
        torch.Generator().manual_seed(9),
        count=2,
        max_length=4,
    )
    require(limited == ["aaaa", "aaaa"], "采不到边界时必须严格遵守 max_length")


def check_08() -> None:
    vocab = build_vocab(WORDS)
    counts = count_bigrams(WORDS, vocab)
    probs = normalize_counts(counts, smoothing=1.0)
    figure_has_content(plot_name_lengths(WORDS))
    figure_has_content(plot_bigram_heatmap(counts, vocab.itos))
    figure_has_content(plot_next_char_distribution(probs, vocab.stoi["a"], vocab.itos))


def check_09() -> None:
    weights = torch.tensor([[1.0, -1.0], [0.5, 0.25]], requires_grad=True)
    x = torch.tensor([0, 1, 0])
    y = torch.tensor([1, 0, 0])
    one_hot_logits = neural_bigram_logits(weights, x, use_one_hot=True)
    lookup_logits = neural_bigram_logits(weights, x, use_one_hot=False)
    require(torch.allclose(one_hot_logits, lookup_logits), "one-hot matmul 必须等价于 W[x]")
    probs = manual_softmax(one_hot_logits)
    require(torch.allclose(probs.sum(1), torch.ones(3)), "softmax 行和不为 1")
    loss = neural_bigram_loss(weights, x, y)
    require(torch.allclose(loss, F.cross_entropy(one_hot_logits, y)), "神经 Bigram loss 不正确")
    extreme_logits = torch.tensor([[1000.0, 999.0, -1000.0]])
    extreme_probs = manual_softmax(extreme_logits)
    finite_tensor(extreme_probs)
    require(
        torch.allclose(extreme_probs, F.softmax(extreme_logits, dim=-1)),
        "manual_softmax 必须在极端 logits 上仍与 PyTorch 一致",
    )
    regularized = neural_bigram_loss(weights, x, y, regularization=0.25)
    expected_regularized = F.cross_entropy(one_hot_logits, y) + 0.25 * weights.square().mean()
    require(torch.allclose(regularized, expected_regularized), "L2 正则项应为 strength*mean(W^2)")


def check_10() -> None:
    vocab = build_vocab(WORDS)
    x, y = build_bigram_pairs(WORDS, vocab)
    one_step_weights = torch.zeros((vocab.size, vocab.size), requires_grad=True)
    one_step_loss = neural_bigram_loss(one_step_weights, x, y)
    (one_step_grad,) = torch.autograd.grad(one_step_loss, (one_step_weights,))
    expected_one_step = one_step_weights.detach() - 0.2 * one_step_grad.detach()
    one_step_history = train_neural_bigram(one_step_weights, x, y, steps=1, learning_rate=0.2)
    require(torch.allclose(one_step_weights, expected_one_step), "神经 Bigram 必须沿负梯度方向更新")
    close(one_step_history.losses[0], one_step_loss.item())

    weights = torch.zeros((vocab.size, vocab.size), requires_grad=True)
    initial = neural_bigram_loss(weights, x, y).item()
    history = train_neural_bigram(weights, x, y, steps=80, learning_rate=0.2)
    require(len(history.losses) == 80, "history 长度不正确")
    require(history.steps == list(range(1, 81)), "history.steps 应表示已完成的更新次数")
    require(history.losses[-1] < initial, "训练后 loss 应下降")
    trigrams = count_trigrams(WORDS, vocab)
    require(tuple(trigrams.shape) == (4, 4, 4), "trigram shape 不正确")
    require(trigrams.sum().item() == 6, "trigram 总数不正确")
    require(trigrams[0, 0, vocab.stoi["a"]].item() == 2, "起始上下文 ..->a 计数错误")
    require(trigrams[0, vocab.stoi["a"], vocab.stoi["b"]].item() == 1, ".a->b 计数错误")
    require(trigrams[0, vocab.stoi["a"], vocab.stoi["c"]].item() == 1, ".a->c 计数错误")
    require(trigrams[vocab.stoi["a"], vocab.stoi["b"], 0].item() == 1, "ab->. 计数错误")

    trigram_probs = normalize_trigram_counts(trigrams, smoothing=1.0)
    require(tuple(trigram_probs.shape) == (4, 4, 4), "trigram 概率 shape 不应改变")
    require(
        torch.allclose(trigram_probs.sum(dim=-1), torch.ones((4, 4))),
        "trigram 必须只沿最后一个目标字符轴归一化",
    )
    expected_nll = reference_trigram_nll(trigram_probs, WORDS, vocab)
    close(trigram_nll(trigram_probs, WORDS, vocab).item(), expected_nll)

    strengths = [0.1, 1.0, 10.0]
    train_words, dev_words = ["ab"], ["ac"]
    result = smoothing_search(train_words, dev_words, vocab, strengths)
    require(result.strengths == strengths, "smoothing 结果必须保留输入强度及顺序")
    require(
        len(result.strengths) == len(result.train_nll) == len(result.dev_nll) == 3,
        "每个 smoothing 强度都必须同时记录 train/dev NLL",
    )
    for index, strength in enumerate(strengths):
        reference_probs = reference_trigram_probs(train_words, vocab, strength)
        close(result.train_nll[index], reference_trigram_nll(reference_probs, train_words, vocab))
        close(result.dev_nll[index], reference_trigram_nll(reference_probs, dev_words, vocab))

    diff = probability_difference(torch.full((4, 4), 0.25), torch.eye(4))
    require(tuple(diff.shape) == (4, 4) and bool((diff >= 0).all()), "概率差应非负且 shape 不变")
    require(
        torch.allclose(diff, (torch.full((4, 4), 0.25) - torch.eye(4)).abs()),
        "概率差必须逐元素取绝对值",
    )
    figure_has_content(plot_smoothing_search(result))
    figure_has_content(plot_probability_difference(diff))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "samples.txt"
        returned = save_lines(["anna", "bob"], path)
        require(returned == path and path.read_text(encoding="utf-8") == "anna\nbob\n", "save_lines 必须逐行保存 UTF-8 文本")
