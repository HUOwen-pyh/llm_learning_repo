"""第一阶段最终组合任务。核心函数完成后，在这里生成全部 Bigram 工件。"""

from __future__ import annotations

from _common import data_path, parse_args, prepare_artifacts


def run(full: bool, seed: int) -> None:
    output = prepare_artifacts()
    path = data_path(full)

    # TODO FINAL-BIGRAM：
    # 1. load_words/build_vocab；
    #    本阶段尚未实现任务 11 的 split_words。smoothing 实验请在本文件中
    #    用 words 的显式、不重叠切片构造 train/dev，勿依赖后续阶段 TODO；
    # 2. 计数、归一化、NLL、采样；
    # 3. 训练神经 Bigram；
    # 4. trigram 与 smoothing search；
    # 5. 调用 plotting/reporting 生成 TASKS.md 中全部 bigram_* 工件。
    raise NotImplementedError("组合任务：生成全部 Bigram 图表、报告与样本")


if __name__ == "__main__":
    args = parse_args("Bigram 最终组合任务")
    run(args.full, args.seed)
