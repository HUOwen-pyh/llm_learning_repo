"""第二阶段最终组合任务：训练 MLP 并输出评估、embedding 和频率分析。"""

from __future__ import annotations

from _common import data_path, parse_args, prepare_artifacts


def run(full: bool, seed: int) -> None:
    output = prepare_artifacts()
    path = data_path(full)

    # TODO FINAL-MLP：
    # 1. 按单词划分并构造 block_size=3 数据；
    # 2. LR range test；
    # 3. 主模型训练与 train/dev/test 评估；
    # 4. 至少两组容量实验；
    # 5. embedding、entropy、频率对照与采样；
    # 6. 生成所有 mlp_* PNG/JSON/TXT。
    raise NotImplementedError("组合任务：生成全部 MLP 图表、报告与样本")


if __name__ == "__main__":
    args = parse_args("MLP 最终组合任务")
    run(args.full, args.seed)

