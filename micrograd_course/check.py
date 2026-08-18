"""按编号检查 micrograd 练习；仅使用 Python 标准库。"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections.abc import Callable

from micrograd import (
    Layer,
    MLP,
    Neuron,
    Value,
    fit,
    forward_loss,
    gradient_check,
    manual_backprop_ab_plus_c,
    numerical_derivative,
    squared_error,
    trace,
    train_step,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


TASK_TITLES = [
    "数值导数",
    "最小 Value",
    "前向运算与计算图",
    "遍历计算图",
    "手算反向传播",
    "局部 backward",
    "tanh",
    "自动 backward",
    "梯度累积",
    "完整标量运算",
    "自动梯度检查",
    "Neuron",
    "Layer",
    "MLP",
    "平方误差",
    "一次训练步骤",
    "训练循环",
    "最终完整 MLP",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_close(
    actual: float,
    expected: float,
    *,
    tolerance: float = 1e-6,
    message: str = "",
) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        detail = message or f"期望 {expected}，实际 {actual}"
        raise AssertionError(detail)


def check_01() -> None:
    fn = lambda x: 3 * x**2 - 4 * x + 5
    require_close(numerical_derivative(fn, 3.0), 14.0, tolerance=1e-4)
    require_close(numerical_derivative(lambda x: x**2, -2.0), -4.0, tolerance=1e-4)


def check_02() -> None:
    value = Value(2.0, label="x")
    require_close(value.data, 2.0)
    require_close(value.grad, 0.0)
    require(value._prev == set(), "叶节点的 _prev 应为空")
    text = repr(value)
    require("2" in text and "x" in text, "repr 应包含 data 和 label")
    require("grad" in text.lower() and "0" in text, "repr 应包含 grad")


def check_03() -> None:
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    product = a * b
    out = product + c
    require_close(out.data, 4.0)
    require(product._op == "*" and out._op == "+", "结果节点应记录运算符")
    require(product._prev == {a, b}, "乘法节点应记录两个输入节点")
    require(out._prev == {product, c}, "加法节点应记录两个输入节点")


def check_04() -> None:
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    product = a * b
    out = product + c
    nodes, edges = trace(out)
    require(len(nodes) == 5, f"a*b+c 应有 5 个值节点，实际 {len(nodes)}")
    require(len(edges) == 4, f"a*b+c 应有 4 条边，实际 {len(edges)}")
    require((a, product) in edges and (product, out) in edges, "边方向应为输入节点 -> 结果节点")


def check_05() -> None:
    grads = manual_backprop_ab_plus_c(2.0, -3.0, 10.0)
    expected = {"out": 1.0, "product": 1.0, "a": -3.0, "b": 2.0, "c": 1.0}
    require(set(grads) == set(expected), f"返回键应为 {set(expected)}")
    for key, value in expected.items():
        require_close(grads[key], value, message=f"{key} 的手算梯度不正确")


def check_06() -> None:
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    product = a * b
    out = product + c
    out.grad = 1.0
    out._backward()
    product._backward()
    require_close(a.grad, -3.0)
    require_close(b.grad, 2.0)
    require_close(c.grad, 1.0)
    require_close(product.grad, 1.0)


def check_07() -> None:
    x = Value(0.5)
    out = x.tanh()
    require_close(out.data, math.tanh(0.5))
    out.grad = 1.0
    out._backward()
    require_close(x.grad, 1.0 - math.tanh(0.5) ** 2)


def check_08() -> None:
    a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
    loss = (a * b + c) * f
    loss.backward()
    require_close(a.grad, 6.0)
    require_close(b.grad, -4.0)
    require_close(c.grad, -2.0)
    require_close(f.grad, 4.0)
    require_close(loss.grad, 1.0)


def check_09() -> None:
    x = Value(3.0)
    (x + x).backward()
    require_close(x.grad, 2.0, message="x+x 的梯度贡献必须相加")

    x = Value(3.0)
    (x * x + x).backward()
    require_close(x.grad, 7.0, message="共享节点梯度应为 2*x+1=7")


def check_10() -> None:
    a = Value(3.0)
    out = (2 + a) * (a - 1) / 2 + a**2
    require_close(out.data, 14.0)
    out.backward()
    require_close(a.grad, 9.5)

    x = Value(0.25)
    exp_x = x.exp()
    exp_x.backward()
    require_close(exp_x.data, math.exp(0.25))
    require_close(x.grad, math.exp(0.25))

    b = Value(4.0)
    require_close((2 * b).data, 8.0)
    require_close((10 - b).data, 6.0)
    require_close((8 / b).data, 2.0)


def check_11() -> None:
    pairs = gradient_check(
        lambda a, b, c: ((a * b + c).tanh() + a**2) / b,
        [1.25, -0.75, 0.2],
    )
    require(len(pairs) == 3, "每个输入都应返回一组梯度")
    for index, (automatic, numerical) in enumerate(pairs):
        require_close(
            automatic,
            numerical,
            tolerance=1e-4,
            message=f"第 {index} 个输入的自动梯度与数值梯度不一致",
        )


def check_12() -> None:
    random.seed(2024)
    neuron = Neuron(2)
    require(hasattr(neuron, "w") and hasattr(neuron, "b"), "Neuron 应创建 w 和 b")
    require(len(neuron.w) == 2, "Neuron(2) 应创建两个权重")
    require(all(isinstance(weight, Value) for weight in neuron.w), "每个权重都应是 Value")
    require(isinstance(neuron.b, Value), "偏置应是 Value")
    require(all(-1.0 <= weight.data <= 1.0 for weight in neuron.w), "权重应初始化在 [-1,1]")
    require_close(neuron.b.data, 0.0, message="偏置应初始化为 0")
    neuron_parameters = neuron.parameters()
    require(len(neuron_parameters) == 3, "Neuron(2) 应有 3 个参数")
    require(len({id(item) for item in neuron_parameters}) == 3, "Neuron 参数不应共享对象")
    require(
        {id(item) for item in neuron_parameters} == {id(item) for item in [*neuron.w, neuron.b]},
        "parameters() 应返回实际的权重和偏置",
    )

    neuron.w = [Value(0.5), Value(-1.0)]
    neuron.b = Value(0.25)
    out = neuron([2.0, 3.0])
    require(isinstance(out, Value), "Neuron 应返回单个 Value")
    require_close(out.data, math.tanh(-1.75))
    require(len(neuron.parameters()) == 3, "Neuron(2) 应有 3 个参数")


def check_13() -> None:
    layer = Layer(3, 4)
    outputs = layer([1.0, 2.0, 3.0])
    require(isinstance(outputs, list) and len(outputs) == 4, "Layer(3,4) 应返回 4 个输出")
    require(all(isinstance(item, Value) for item in outputs), "每个输出都应是 Value")
    parameters = layer.parameters()
    require(len(parameters) == 16, "Layer(3,4) 应有 16 个参数")
    require(len({id(item) for item in parameters}) == 16, "不同神经元不应共享参数")

    single = Layer(3, 1)([1.0, 2.0, 3.0])
    require(isinstance(single, Value), "单输出 Layer 应直接返回 Value")


def check_14() -> None:
    model = MLP(3, [4, 4, 1])
    out = model([2.0, 3.0, -1.0])
    require(isinstance(out, Value), "最终单输出 MLP 应返回 Value")
    require(len(model.parameters()) == 41, "MLP(3,[4,4,1]) 应有 41 个参数")

    narrow_model = MLP(3, [1, 2])
    narrow_out = narrow_model([2.0, 3.0, -1.0])
    require(isinstance(narrow_out, list) and len(narrow_out) == 2, "MLP 应支持单节点中间层")


def check_15() -> None:
    p1, p2 = Value(0.5), Value(-0.5)
    loss = squared_error([p1, p2], [1.0, -1.0])
    require_close(loss.data, 0.5)
    loss.backward()
    require_close(p1.grad, -1.0)
    require_close(p2.grad, 1.0)


def check_16() -> None:
    neuron = Neuron(1)
    neuron.w = [Value(0.0)]
    neuron.b = Value(0.0)
    for parameter in neuron.parameters():
        parameter.grad = 99.0
    neuron.zero_grad()
    require(all(parameter.grad == 0.0 for parameter in neuron.parameters()), "zero_grad 未清空梯度")

    _, before = forward_loss(neuron, [[1.0]], [1.0])
    reported = train_step(neuron, [[1.0]], [1.0], learning_rate=0.1)
    _, after = forward_loss(neuron, [[1.0]], [1.0])
    require_close(reported, before.data)
    require(after.data < before.data, "正确的一次梯度下降应使这个简单样本的 loss 下降")


def check_17() -> None:
    neuron = Neuron(1)
    neuron.w = [Value(0.0)]
    neuron.b = Value(0.0)
    losses = fit(neuron, [[1.0]], [1.0], steps=25, learning_rate=0.1)
    require(len(losses) == 25, "fit 应返回每一步的 loss")
    require(losses[-1] < losses[0] * 0.05, "简单训练循环结束后的 loss 应显著下降")


def check_18() -> None:
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    random.seed(1337)
    model = MLP(3, [4, 4, 1])
    losses = fit(model, xs, ys, steps=120, learning_rate=0.05)
    predictions, final_loss = forward_loss(model, xs, ys)
    require(final_loss.data < losses[0] * 0.1, "最终 MLP 的 loss 没有显著下降")
    signs = [1.0 if item.data > 0 else -1.0 for item in predictions]
    require(signs == ys, f"预测符号 {signs} 与标签 {ys} 不一致")


CHECKS: list[Callable[[], None]] = [
    check_01,
    check_02,
    check_03,
    check_04,
    check_05,
    check_06,
    check_07,
    check_08,
    check_09,
    check_10,
    check_11,
    check_12,
    check_13,
    check_14,
    check_15,
    check_16,
    check_17,
    check_18,
]


def parse_target(raw: str) -> int:
    if raw.lower() == "all":
        return len(CHECKS)
    try:
        target = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("任务编号必须是 1-18 或 all") from exc
    if not 1 <= target <= len(CHECKS):
        raise argparse.ArgumentTypeError("任务编号必须是 1-18")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="逐关检查 micrograd 实现")
    parser.add_argument("target", nargs="?", default="all", help="检查到第几关，或 all")
    parser.add_argument("--list", action="store_true", help="列出全部任务")
    args = parser.parse_args()

    if args.list:
        for index, title in enumerate(TASK_TITLES, start=1):
            print(f"{index:02d}. {title}")
        return

    try:
        target = parse_target(args.target)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    for index, check in enumerate(CHECKS[:target], start=1):
        title = TASK_TITLES[index - 1]
        try:
            check()
        except NotImplementedError as exc:
            print(f"[待完成] {index:02d}. {title}: {exc}")
            raise SystemExit(1) from exc
        except Exception as exc:
            print(f"[未通过] {index:02d}. {title}: {exc}")
            raise SystemExit(1) from exc
        else:
            print(f"[通过]   {index:02d}. {title}")

    print(f"\n已通过前 {target} 个任务。")


if __name__ == "__main__":
    main()
