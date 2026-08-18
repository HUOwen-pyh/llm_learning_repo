"""第三阶段最终组合任务：系统诊断 MLP 内部健康状况。"""

from __future__ import annotations

from _common import data_path, parse_args, prepare_artifacts


def run(full: bool, seed: int) -> None:
    output = prepare_artifacts()
    path = data_path(full)

    # TODO FINAL-DIAGNOSTICS：
    # 1. naive/fixed 初始化对照；
    # 2. tanh preactivation/activation/saturation；
    # 3. Kaiming gain 与方差传播；
    # 4. BN batch coupling、eval、folding；
    # 5. 深层网络 activation、activation-grad、parameter-grad、update:data；
    # 6. 全零初始化尸检；
    # 7. 保存 TASKS.md 中全部诊断 PNG 和 diagnostics.json。
    raise NotImplementedError("组合任务：生成 MLP 内部诊断 dashboard")


if __name__ == "__main__":
    args = parse_args("MLP 内部诊断最终组合任务")
    run(args.full, args.seed)

